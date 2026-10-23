#!/usr/bin/env python3
"""Pretrain a shared PatchTST encoder using train-only ACF forecast relations."""
from __future__ import annotations

import argparse, hashlib, json, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from data_provider.data_loader import load_numeric_array_file
from data_provider.samplers import ACFLagPairBatchSampler
from layers.losses import ChannelPatchLagContrastiveLoss, ForecastLagPairContrastiveLoss
from models.PatchTSTACFPretrain import Model
from utils.acf_guidance import fit_signed_acf_prior


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, max_windows, lags):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        starts = np.arange(len(values) - seq_len - pred_len + 1, dtype=np.int64)
        if not len(starts): raise ValueError("source prefix is too short")
        self.starts = self._select(starts, max_windows, lags)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)

    @staticmethod
    def _select(starts, max_windows, lags):
        if len(starts) <= max_windows: return starts
        usable = [int(lag) for lag in sorted(set(lags)) if 0 < int(lag) < len(starts)] or [1]
        selected = set()
        for index in range(max_windows // 2):
            lag = usable[index % len(usable)]
            anchor = int(round(index * (len(starts) - lag - 1) / max(1, max_windows // 2 - 1)))
            selected.update((int(starts[anchor]), int(starts[anchor + lag])))
        for start in np.linspace(0, len(starts) - 1, max_windows * 2, dtype=np.int64):
            if len(selected) >= max_windows: break
            selected.add(int(start))
        return np.asarray(sorted(selected)[:max_windows], dtype=np.int64)

    def __len__(self): return len(self.starts)
    def __getitem__(self, index):
        start = int(self.starts[index])
        return start, self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def spec(value):
    name, path, fmt = value.split(":", 2); return name, Path(path).resolve(), fmt
def load_values(path, fmt):
    if fmt == "csv":
        return pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=np.float32)
    return load_numeric_array_file(path, data_format=fmt)
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
def peaks(prior):
    result = np.zeros((prior.n_channels, 8), dtype=np.int64)
    for channel in range(prior.n_channels):
        selected = prior.channel_lags(channel, top_k=8); result[channel, :len(selected)] = selected
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--calibration-source")
    parser.add_argument("--calibration-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96); parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24); parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=32); parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4); parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=.6); parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--max-windows", type=int, default=192)
    parser.add_argument("--relation-loss-fraction", type=float, default=.1)
    parser.add_argument("--calibration-relation-loss-fraction", type=float, default=.3)
    parser.add_argument("--channel-patch-relation-fraction", type=float, default=0.)
    parser.add_argument("--calibration-channel-patch-relation-fraction", type=float, default=0.)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if (args.calibration_source is None) != (args.calibration_fraction is None): raise ValueError("calibration source/fraction must pair")
    if any(not 0 <= value <= 1 for value in (args.relation_loss_fraction, args.calibration_relation_loss_fraction, args.channel_patch_relation_fraction, args.calibration_channel_patch_relation_fraction)): raise ValueError("loss fractions must be in [0,1]")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu: torch.set_num_threads(int(cpu))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    source_specs = [(spec(item), "source", None) for item in args.source]
    if args.calibration_source: source_specs.append((spec(args.calibration_source), "target_calibration", args.calibration_fraction))
    sources = []
    for (name, path, fmt), role, fraction in source_specs:
        raw = load_values(path, fmt)
        rows = int(len(raw) * args.train_fraction * (fraction if role == "target_calibration" else 1.0))
        train = raw[:rows]
        prior = fit_signed_acf_prior(train, max_lag=min(720, rows - 1), top_k=8, min_lag=1, min_abs_acf=.05, confidence_z=1.96, min_separation=2, detrend="mean", channel_chunk_size=64)
        channel_peaks = peaks(prior); lags = sorted({int(lag) for lag in channel_peaks.ravel() if lag > 0}) or [1]
        sources.append({"name": name, "dataset": Windows(train, args.seq_len, args.pred_len, args.max_windows, lags), "acf": prior.acf.astype(np.float32), "peaks": channel_peaks, "lags": lags, "fraction": args.calibration_relation_loss_fraction if role == "target_calibration" else args.relation_loss_fraction, "channel_fraction": args.calibration_channel_patch_relation_fraction if role == "target_calibration" else args.channel_patch_relation_fraction, "metadata": {"name": name, "path": str(path), "sha256": digest(path), "role": role, "train_rows": rows, "channels": int(raw.shape[1]), "relation_loss_fraction": args.calibration_relation_loss_fraction if role == "target_calibration" else args.relation_loss_fraction, "channel_patch_relation_fraction": args.calibration_channel_patch_relation_fraction if role == "target_calibration" else args.channel_patch_relation_fraction}})
    model = Model(args.seq_len, args.pred_len, args.patch_len, args.patch_stride, args.d_model, args.d_ff, args.n_heads, args.e_layers).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history = []
    for epoch in range(args.epochs):
        losses=[]; cp_losses=[]; cp_active=[]; relation_active=[]
        for source in sources:
            loader = DataLoader(source["dataset"], batch_sampler=ACFLagPairBatchSampler(source["dataset"], args.batch_size, source["lags"], seed=args.seed + epoch, origins=source["dataset"].starts, lag_balance="balanced"), num_workers=0)
            forecast_relation = ForecastLagPairContrastiveLoss(source["acf"], selected_lags=source["lags"], channel_selected_lags=source["peaks"]).cuda()
            channel_relation = ChannelPatchLagContrastiveLoss(source["acf"], selected_lags=source["lags"], channel_selected_lags=source["peaks"]).cuda() if source["channel_fraction"] else None
            for origins, x, future in loader:
                x, future, origins = x.cuda(), future.cuda(), origins.long().cuda()
                normalized, (mean, scale) = model._normalize(x)
                prediction, encoded = model.normalized_forecast(normalized)
                target = (future - mean) / scale
                predictive = F.smooth_l1_loss(prediction, target)
                forecast_loss = forecast_relation(prediction, origins).mean()
                forecast_weight = source["fraction"] * predictive.detach() / forecast_loss.detach().abs().clamp_min(1e-6)
                if channel_relation is not None:
                    cp_loss = channel_relation(encoded, origins).mean()
                    cp_weight = source["channel_fraction"] * predictive.detach() / cp_loss.detach().abs().clamp_min(1e-6)
                    cp_losses.append(float(cp_loss.detach())); cp_active.append(float(channel_relation.last_active_anchor_fraction))
                else:
                    cp_loss = predictive.detach() * 0.; cp_weight = cp_loss
                loss = predictive + forecast_weight * forecast_loss + cp_weight * cp_loss
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach())); relation_active.append(float(forecast_relation.last_active_anchor_fraction))
        row = {"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "forecast_relation_active_fraction": float(np.mean(relation_active)), "channel_patch_relation_active_fraction": float(np.mean(cp_active)) if cp_active else 0., "mean_channel_patch_relation_loss": float(np.mean(cp_losses)) if cp_losses else 0.}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "config": config, "sources": [source["metadata"] for source in sources], "history": history, "architecture": "shared_channel_independent_patchtst_encoder", "mamba_scan_count": 0, "test_metrics_produced": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({key: value for key, value in payload.items() if key != "model"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "test_metrics_produced": False}, sort_keys=True))


if __name__ == "__main__": main()
