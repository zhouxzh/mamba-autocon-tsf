#!/usr/bin/env python3
"""Low-data target adaptation from a frozen AutoCon MambaTSv3 source core.

Only source state keys whose names and tensor shapes are identical in the
target model are restored.  This explicitly excludes channel-count dependent
embeddings/context, target horizon heads, and persistence scales.  Target test
rows are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
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

from models import MambaTSv3


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("no valid forecast windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config(args, channels):
    return SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, enc_in=channels, c_out=channels,
        d_model=args.d_model, d_ff=args.d_ff, patch_len=args.patch_len,
        patch_stride=args.patch_stride, dropout=0.1, e_layers=1,
        mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand, mamba_headdim=args.mamba_headdim,
        mamba_ts_variant=getattr(args, "mamba_ts_variant", "v3"), mamba_ts_mixer_type=args.mamba_ts_mixer_type,
        mamba_ts_context_rank=args.context_rank,
        mamba_ts_channel_agnostic_scan=args.channel_agnostic_scan, AutoCon=True,
        mamba_ts_repr_forecast_fusion=bool(getattr(args, "mamba_ts_repr_forecast_fusion", False)),
        mamba_ts_repr_forecast_channelwise=bool(getattr(args, "mamba_ts_repr_forecast_channelwise", False)),
        mamba_ts_shared_channel_summary=bool(getattr(args, "mamba_ts_shared_channel_summary", False)),
        mamba_ts_channel_set_pooling=bool(getattr(args, "mamba_ts_channel_set_pooling", False)),
        mamba_ts_channel_prototype_pooling=bool(getattr(args, "mamba_ts_channel_prototype_pooling", False)),
        mamba_ts_channel_prototype_count=int(getattr(args, "mamba_ts_channel_prototype_count", 4)),
        mamba_ts_shared_postscan_mixer=bool(getattr(args, "mamba_ts_shared_postscan_mixer", False)),
        mamba_ts_shared_postscan_mixer_rank=int(getattr(args, "mamba_ts_shared_postscan_mixer_rank", 4)),
        mamba_ts_acf_channel_descriptor=bool(getattr(args, "mamba_ts_acf_channel_descriptor", False)),
        mamba_ts_channel_adapter=bool(getattr(args, "mamba_ts_channel_adapter", False)),
        mamba_ts_channel_adapter_rank=int(getattr(args, "mamba_ts_channel_adapter_rank", 8)),
        mamba_ts_target_mlp_adapter=bool(getattr(args, "mamba_ts_target_mlp_adapter", False)),
        mamba_ts_target_mlp_adapter_rank=int(getattr(args, "mamba_ts_target_mlp_adapter_rank", 32)),
        mamba_ts_revin_affine=bool(getattr(args, "mamba_ts_revin_affine", False)),
        AutoCon_wnorm="MeanStd", task_name="long_term_forecast",
    )


def model_output(model, x):
    output = model(x, return_repr=False)
    return output[0] if isinstance(output, tuple) else output


def transfer_core(model, checkpoints: list[Path], transfer_prediction_head: bool = False,
                  head_transfer_mode: str = "tile"):
    states = []
    prediction_heads = []
    for checkpoint in checkpoints:
        raw = torch.load(checkpoint, map_location="cpu")
        states.append(raw.get("model", raw))
        head = raw.get("pretrain_prediction_head")
        if head is not None:
            prediction_heads.append(head)
    target = model.state_dict()
    transferable_prefixes = ("patch_embedding.", "patch_position_embedding", "encoder.", "final_norm.", "repr_", "ts_shared_channel_summary_", "ts_channel_set_pooling_", "ts_channel_prototype_", "ts_lite_", "ts_acf_channel_descriptor_", "ts_shared_postscan_mixer.")
    common = set.intersection(*(set(state) for state in states))
    loaded = {}
    for key in common:
        values = [state[key] for state in states]
        if (key != "ts_acf_descriptor_values" and key.startswith(transferable_prefixes)
                and key in target and all(value.shape == target[key].shape for value in values)):
            loaded[key] = torch.stack([value.float() for value in values]).mean(dim=0).to(dtype=target[key].dtype)
    model.load_state_dict(loaded, strict=False)
    if not any(key.startswith("encoder.") for key in loaded):
        raise ValueError("source checkpoint has no compatible MambaTSv3 encoder core")
    transferred_head = False
    if transfer_prediction_head:
        if not prediction_heads:
            raise ValueError("transfer-prediction-head requires a next-patch pretraining checkpoint")
        weights = [head["weight"] for head in prediction_heads]
        biases = [head["bias"] for head in prediction_heads]
        target_head = model.forecast_head
        if not all(weight.shape[1] == target_head.weight.shape[1] for weight in weights):
            raise ValueError("pretraining prediction head input width does not match target forecast head")
        weight = torch.stack([value.float() for value in weights]).mean(dim=0)
        bias = torch.stack([value.float() for value in biases]).mean(dim=0)
        if head_transfer_mode == "exact" and weight.shape != target_head.weight.shape:
            raise ValueError("exact prediction-head transfer requires source and target horizons to match")
        repeats = 1 if head_transfer_mode == "exact" else (target_head.weight.shape[0] + weight.shape[0] - 1) // weight.shape[0]
        with torch.no_grad():
            target_head.weight.copy_(weight.repeat((repeats, 1))[:target_head.weight.shape[0]].to(target_head.weight))
            target_head.bias.copy_(bias.repeat(repeats)[:target_head.bias.shape[0]].to(target_head.bias))
        transferred_head = True
    return {"checkpoints": [str(path) for path in checkpoints], "sha256": [digest(path) for path in checkpoints], "loaded_keys": sorted(loaded), "prediction_head_transferred": transferred_head, "prediction_head_transfer_mode": head_transfer_mode if transferred_head else None}


def json_compatible(value):
    """Convert argparse values such as repeated Path options into JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--mamba-d-state", type=int, default=8)
    parser.add_argument("--mamba-d-conv", type=int, default=2)
    parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--mamba-headdim", type=int, default=None,
                        help="Mamba mixer head width; defaults to d-model to match source pretraining")
    parser.add_argument("--mamba-ts-mixer-type", choices=["mamba1", "mamba2"], default="mamba1")
    parser.add_argument("--context-rank", type=int, default=16)
    parser.add_argument("--channel-agnostic-scan", action="store_true")
    parser.add_argument("--mamba-ts-repr-forecast-fusion", action="store_true",
                        help="zero-gated AutoCon representation-to-forecast residual from the existing scan")
    parser.add_argument("--mamba-ts-repr-forecast-channelwise", action="store_true",
                        help="use shared per-channel representation projection; requires forecast fusion")
    parser.add_argument("--mamba-ts-shared-channel-summary", action="store_true",
                        help="inject channel-count-agnostic mean/std patch summary before the shared scan")
    parser.add_argument("--mamba-ts-channel-set-pooling", action="store_true",
                        help="learn permutation-equivariant channel pooling before the shared scan")
    parser.add_argument("--mamba-ts-channel-prototype-pooling", action="store_true")
    parser.add_argument("--mamba-ts-channel-prototype-count", type=int, default=4)
    parser.add_argument("--transfer-prediction-head", action="store_true")
    parser.add_argument("--prediction-head-transfer-mode", choices=["tile", "exact"], default="tile")
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25], required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--freeze-core-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3501)
    args = parser.parse_args()
    if args.mamba_headdim is None:
        args.mamba_headdim = args.d_model
    if args.mamba_headdim <= 0:
        raise ValueError("mamba-headdim must be positive")
    if args.mamba_ts_repr_forecast_channelwise and not args.mamba_ts_repr_forecast_fusion:
        raise ValueError("mamba-ts-repr-forecast-channelwise requires mamba-ts-repr-forecast-fusion")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = pd.read_csv(args.dataset).iloc[:, 1:].to_numpy(dtype=np.float32)
    train_end = int(len(raw) * args.train_fraction)
    val_end = int(len(raw) * (args.train_fraction + args.val_fraction))
    few_end = int(train_end * args.few_shot_fraction)
    mean = raw[:few_end].mean(axis=0, keepdims=True)
    std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    train_starts = np.arange(0, few_end - args.seq_len - args.pred_len + 1)
    if len(train_starts) > args.max_train_windows:
        train_starts = np.linspace(0, len(train_starts) - 1, args.max_train_windows, dtype=np.int64)
    val_starts = np.arange(max(0, train_end - args.seq_len), val_end - args.seq_len - args.pred_len + 1)
    val_starts = val_starts[val_starts + args.seq_len >= train_end]
    train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, train_starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    model = MambaTSv3.Model(config(args, raw.shape[1])).cuda()
    transfer = None if args.source_checkpoint is None else transfer_core(
        model, args.source_checkpoint, args.transfer_prediction_head,
        args.prediction_head_transfer_mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        freeze = transfer is not None and epoch < args.freeze_core_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not (freeze and name.startswith(("patch_embedding", "patch_position_embedding", "encoder.", "final_norm", "repr_")))
        model.train(); losses = []
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = F.mse_loss(model_output(model, x), y)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader:
                errors.append(F.mse_loss(model_output(model, x.cuda()), y.cuda(), reduction="none").mean(dim=(1,2)).cpu())
        val = float(torch.cat(errors).mean())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val, "core_frozen": freeze})
        if val < best:
            best, best_state = val, {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = {"best_validation_mse": best, "history": history, "transfer": transfer, "target_channels": int(raw.shape[1]), "few_train_rows": int(few_end), "original_train_rows": int(train_end), "validation_rows": int(val_end-train_end), "test_metrics_produced": False}
    torch.save({"model": best_state, "config": vars(args), **meta}, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({"config": json_compatible(vars(args)), **meta}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
