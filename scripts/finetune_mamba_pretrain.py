#!/usr/bin/env python3
"""Fine-tune the shared Mamba pretraining encoder on a low-data target.

This script reports validation metrics only.  The target's original training
prefix is truncated to ``--few-shot-fraction``; validation rows remain outside
that prefix.  Optional sensor dropout and noise are validation-time stressors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import load_numeric_array_file
from models import MambaTSPretrain, MambaTSPretrainChannelwise, MambaTSPretrainChannelContext


class ForecastWindows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError("forecast split has no valid windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def load_values(path, fmt):
    if fmt == "csv":
        frame = pd.read_csv(path)
        return frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    return load_numeric_array_file(path, data_format=fmt)


def model_from_args(args, channels):
    model_class = {
        "pooled": MambaTSPretrain.Model,
        "channelwise": MambaTSPretrainChannelwise.Model,
        "channel_context": MambaTSPretrainChannelContext.Model,
    }[args.encoder_mode]
    return model_class(SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, patch_len=args.patch_len,
        patch_stride=args.patch_stride, d_model=args.d_model, d_ff=args.d_ff,
        dropout=0.1, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        enc_in=channels, c_out=channels,
    ))


def load_encoder(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    target = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in target and target[key].shape == value.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    restored = sorted(compatible)
    if "mamba.A_log" not in " ".join(restored) and not any(key.startswith("mamba.") for key in restored):
        raise ValueError("checkpoint does not contain a compatible pretrained Mamba encoder")
    return {"restored_keys": restored, "missing_keys": missing, "unexpected_keys": unexpected}


def evaluate(model, loader, device, channel_dropout, noise_std, seed, data_interval_minutes=None):
    model.eval()
    errors = []
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if channel_dropout:
                keep = torch.rand(x.shape[0], 1, x.shape[-1], device=device, generator=generator) >= channel_dropout
                x = x * keep.to(x.dtype)
            if noise_std:
                scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                x = x + noise_std * scale * torch.randn(x.shape, device=device, generator=generator, dtype=x.dtype)
            kwargs = {"data_interval_minutes": data_interval_minutes} if data_interval_minutes is not None else {}
            errors.append(F.mse_loss(model(x, **kwargs), y, reduction="none").mean(dim=(1, 2)).cpu())
    return float(torch.cat(errors).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="PATH to target data")
    parser.add_argument("--format", default="csv", choices=["csv", "npz", "txt"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--encoder-mode", choices=["pooled", "channelwise", "channel_context"], default="pooled")
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25], required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    parser.add_argument("--channel-dropout", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--data-interval-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3401)
    args = parser.parse_args()
    cpu_limit = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu_limit:
        torch.set_num_threads(int(cpu_limit))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("train/validation fractions must form chronological splits")
    if not 0 <= args.channel_dropout < 1 or args.noise_std < 0:
        raise ValueError("invalid stressor magnitude")
    if args.data_interval_minutes is not None and args.data_interval_minutes <= 0:
        raise ValueError("data_interval_minutes must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = load_values(Path(args.dataset), args.format)
    train_end = int(len(raw) * args.train_fraction)
    val_end = int(len(raw) * (args.train_fraction + args.val_fraction))
    few_end = int(train_end * args.few_shot_fraction)
    if few_end <= args.seq_len + args.pred_len:
        raise ValueError("few-shot prefix is too short for one forecast window")
    mean = raw[:few_end].mean(axis=0, keepdims=True)
    std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    train_starts = np.arange(0, few_end - args.seq_len - args.pred_len + 1)
    if args.max_train_windows is not None and len(train_starts) > args.max_train_windows:
        train_starts = np.linspace(0, len(train_starts) - 1, args.max_train_windows, dtype=np.int64)
    # Validation targets must begin at the original validation boundary.
    val_start_min = max(0, train_end - args.seq_len)
    val_starts = np.arange(val_start_min, val_end - args.seq_len - args.pred_len + 1)
    val_starts = val_starts[val_starts + args.seq_len >= train_end]
    train_loader = DataLoader(ForecastWindows(values, args.seq_len, args.pred_len, train_starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ForecastWindows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size, shuffle=False)
    model = model_from_args(args, raw.shape[1]).cuda()
    checkpoint_info = None if args.checkpoint is None else load_encoder(model, args.checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history = []
    best_state, best_val = None, float("inf")
    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_encoder_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not (frozen and name.startswith(("patch_embedding", "position_embedding", "encoder_norm", "mamba", "final_norm", "repr_")))
        model.train(); losses = []
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            kwargs = {"data_interval_minutes": args.data_interval_minutes} if args.encoder_mode == "channel_context" and args.data_interval_minutes is not None else {}
            loss = F.mse_loss(model(x, **kwargs), y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        val_mse = evaluate(model, val_loader, "cuda", args.channel_dropout, args.noise_std, args.seed + epoch, args.data_interval_minutes if args.encoder_mode == "channel_context" else None)
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val_mse, "encoder_frozen": frozen})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "channels": int(raw.shape[1]), "few_train_rows": int(few_end), "original_train_rows": int(train_end), "validation_rows": int(val_end - train_end), "best_validation_mse": best_val, "history": history, "checkpoint_load": checkpoint_info, "test_metrics_produced": False}
    torch.save({"model": best_state, **metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "best_validation_mse": best_val, "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
