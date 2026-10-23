#!/usr/bin/env python3
"""Pretrain the channel-count agnostic temporal Mamba with AutoCon.

Each dataset is loaded and batched separately, but one model and optimizer are
shared across all datasets.  ACF priors are fitted from each dataset's train
prefix only.  The script never opens validation or test rows.
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

from layers.losses import ACFRelationLoss
from models import MambaTSPretrain, MambaTSPretrainChannelwise, MambaTSPretrainChannelContext
from utils.acf_guidance import fit_signed_acf_prior
from data_provider.data_loader import load_numeric_array_file
from data_provider.samplers import ACFLagPairBatchSampler


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int, max_windows: int | None = None,
                 contiguous_windows: bool = False, next_patch: bool = False,
                 patch_len: int | None = None):
        if values.ndim != 2 or values.shape[0] <= seq_len:
            raise ValueError("training values must be [time, channels] with enough rows")
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.next_patch = bool(next_patch)
        self.patch_len = int(patch_len or 0)
        if self.next_patch and self.patch_len <= 0:
            raise ValueError("next_patch task requires a positive patch_len")
        count = len(self.values) - self.seq_len - (self.patch_len if self.next_patch else 0) + 1
        self.starts = np.arange(count, dtype=np.int64)
        if max_windows is not None and len(self.starts) > int(max_windows):
            self.starts = (
                self.starts[:int(max_windows)] if contiguous_windows
                else np.linspace(0, count - 1, int(max_windows), dtype=np.int64)
            )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        x = self.values[start:start + self.seq_len]
        if self.next_patch:
            return start, x, self.values[start + self.seq_len:start + self.seq_len + self.patch_len]
        return start, x


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_values(path: Path, data_format: str) -> np.ndarray:
    if data_format == "csv":
        frame = pd.read_csv(path)
        if frame.shape[1] < 2:
            raise ValueError("CSV must contain a timestamp column and channels")
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        values = load_numeric_array_file(path, data_format=data_format)
    if not np.isfinite(values).all():
        raise ValueError("dataset contains non-finite values: {}".format(path))
    return values


def _dataset_spec(raw: str) -> tuple[str, Path, str, float | None]:
    parts = raw.split(":")
    if len(parts) not in {2, 3, 4}:
        raise ValueError("dataset spec must be NAME:PATH[:csv|npz|txt][:interval_minutes]")
    name, path = parts[:2]
    fmt = parts[2] if len(parts) >= 3 else ("csv" if path.lower().endswith(".csv") else "auto")
    interval = float(parts[3]) if len(parts) == 4 else None
    if interval is not None and interval <= 0:
        raise ValueError("dataset interval_minutes must be positive")
    return name, Path(path).expanduser().resolve(), fmt, interval


def _fit_dataset(path: Path, fmt: str, seq_len: int, train_fraction: float,
                 max_windows: int | None, contiguous_windows: bool, pretrain_task: str,
                 patch_len: int):
    raw = _load_values(path, fmt)
    split = int(len(raw) * train_fraction)
    train_raw = raw[:split]
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True)
    values = (train_raw - mean) / np.sqrt(std * std + 1e-5)
    prior = fit_signed_acf_prior(
        train_raw, max_lag=min(720, len(train_raw) - 1), top_k=8,
        min_lag=1, min_abs_acf=0.05, confidence_z=1.96,
        min_separation=2, detrend="mean", channel_chunk_size=64,
    )
    acf = prior.acf.mean(axis=0).astype(np.float32)
    selected = list(prior.common_lags(top_k=8))
    selected = [lag for lag in selected if lag < len(train_raw)] or [1]
    dataset = WindowDataset(
        values, seq_len, max_windows=max_windows, contiguous_windows=contiguous_windows,
        next_patch=pretrain_task == "next_patch", patch_len=patch_len,
    )
    return dataset, acf, selected, {
        "name": path.name, "path": str(path), "sha256": _sha256(path),
        "raw_rows": int(len(raw)), "train_rows": int(split), "channels": int(raw.shape[1]),
        "selected_lags": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="NAME:PATH[:csv|npz|txt], repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--encoder-mode", choices=["pooled", "channelwise", "channel_context"], default="pooled")
    parser.add_argument("--pretrain-task", choices=["masked_reconstruction", "next_patch"], default="masked_reconstruction")
    parser.add_argument("--next-patch-loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--max-windows", type=int, default=20000)
    parser.add_argument("--relation-batching", choices=["random", "lag_pairs"], default="random")
    parser.add_argument("--contiguous-windows", action="store_true")
    parser.add_argument("--mamba-lambda", type=float, default=0.001)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
        raise RuntimeError("CUDA is required for the Mamba pretraining script")
    if not 0.0 < args.train_fraction < 1.0 or not 0.0 <= args.mask_ratio < 1.0:
        raise ValueError("train_fraction must be in (0,1), mask_ratio in [0,1)")
    if args.relation_batching == "lag_pairs" and not args.contiguous_windows:
        raise ValueError("lag_pairs requires --contiguous-windows so selected lags retain origin meaning")
    if args.relation_batching == "lag_pairs" and args.batch_size % 2:
        raise ValueError("lag_pairs requires an even batch size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    specs = [_dataset_spec(raw) for raw in args.dataset]
    prepared = []
    for name, path, fmt, interval in specs:
        dataset, acf, selected, metadata = _fit_dataset(
            path, fmt, args.seq_len, args.train_fraction, args.max_windows, args.contiguous_windows,
            args.pretrain_task, args.patch_len,
        )
        metadata["name"] = name
        metadata["interval_minutes"] = interval
        prepared.append((dataset, acf, selected, metadata))

    config = SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.seq_len, patch_len=args.patch_len,
        patch_stride=args.patch_stride, d_model=args.d_model, d_ff=args.d_ff,
        dropout=0.1, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
    )
    model_class = {
        "pooled": MambaTSPretrain.Model,
        "channelwise": MambaTSPretrainChannelwise.Model,
        "channel_context": MambaTSPretrainChannelContext.Model,
    }[args.encoder_mode]
    model = model_class(config).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history = []
    for epoch in range(args.epochs):
        epoch_losses = []
        active_fractions = []
        for dataset, acf, selected, metadata in prepared:
            if args.relation_batching == "lag_pairs" and args.mamba_lambda:
                legal_lags = [lag for lag in selected if lag < len(dataset)] or [1]
                sampler = ACFLagPairBatchSampler(
                    dataset, batch_size=args.batch_size, lags=legal_lags,
                    drop_last=False, seed=args.seed + epoch, lag_balance="balanced",
                )
                loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
            else:
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            relation = None if args.mamba_lambda == 0 else ACFRelationLoss(
                acf, temperature=0.2, relation_temperature=0.25,
                sign_weight=0.25, sign_margin=0.1, relation_scope="both",
                selected_lags=selected, lag_tolerance=0, target_mode="hard_selected",
                soft_min_abs=0.05, diagnostic_min_abs=0.05, min_lag=0,
            ).cuda()
            for packed in loader:
                if args.pretrain_task == "next_patch":
                    starts, batch, future_patch = packed
                else:
                    starts, batch = packed
                    future_patch = None
                batch = batch.cuda(non_blocking=True)
                model_kwargs = (
                    {"data_interval_minutes": metadata["interval_minutes"]}
                    if args.encoder_mode == "channel_context" and metadata["interval_minutes"] is not None
                    else {}
                )
                if args.pretrain_task == "next_patch":
                    if args.encoder_mode != "channel_context":
                        raise ValueError("next_patch pretraining currently requires encoder-mode=channel_context")
                    states, normalized, mean, scale = model.encode(batch, **model_kwargs)
                    outputs = {"representation": model.representation(states)}
                    prediction = model.predict_next_patch(states)
                    target = future_patch.cuda(non_blocking=True).transpose(1, 2)
                    target = (target - mean.transpose(1, 2)) / scale.transpose(1, 2)
                    rec_loss = (
                        F.mse_loss(prediction, target)
                        if args.next_patch_loss == "mse"
                        else F.smooth_l1_loss(prediction, target)
                    )
                else:
                    outputs = model(batch, pretrain=True, mask=(torch.rand(batch.shape[0], model.patch_count, device="cuda") < args.mask_ratio), **model_kwargs)
                    target = outputs["reconstruction_target"]
                    mask = outputs["mask"]
                    rec_error = (outputs["reconstruction"] - target).square().mean(dim=-1)
                    if rec_error.ndim == 3:
                        mask = mask[:, None, :].expand_as(rec_error)
                    rec_loss = rec_error.masked_select(mask).mean() if mask.any() else rec_error.mean() * 0.0
                if relation is None:
                    autocon_loss = rec_loss.detach() * 0.0
                else:
                    features = F.normalize(outputs["representation"], dim=-1)
                    local_loss, global_loss = relation(features, starts.long().cuda())
                    autocon_loss = (local_loss.mean(dim=1) + global_loss).mean() / 2.0
                    active_fractions.append(float(relation.last_global_contrastive_active_anchor_fraction.detach().cpu()))
                loss = rec_loss + float(args.mamba_lambda) * autocon_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append({"dataset": metadata["name"], "rec": float(rec_loss.detach()), "autocon": float(autocon_loss.detach()), "loss": float(loss.detach())})
        mean_loss = float(np.mean([item["loss"] for item in epoch_losses])) if epoch_losses else float("nan")
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "batches": len(epoch_losses), "global_relation_active_fraction": float(np.mean(active_fractions)) if active_fractions else 0.0})
        print(json.dumps(history[-1], sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {"config": config_payload, "datasets": [item[3] for item in prepared], "history": history}
    torch.save({"model": model.state_dict(), **payload}, args.output)
    (args.output.with_suffix(".json")).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "epochs": args.epochs, "datasets": len(prepared)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
