#!/usr/bin/env python3
"""Audit train-only forecast-InfoNCE pair geometry before a GPU experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.samplers import ACFLagPairBatchSampler
from layers.losses import ForecastLagPairContrastiveLoss
from scripts.pretrain_mambatsv3_core_autocon import (
    Windows, _channel_peak_matrix, _coverage_pair_lags, _load_values, _source_spec,
)
from utils.acf_guidance import fit_signed_acf_prior


def _profile(name: str, path: Path, data_format: str, args):
    raw = _load_values(path, data_format)
    train = raw[:int(len(raw) * args.train_fraction)]
    prior = fit_signed_acf_prior(
        train, max_lag=min(args.max_lag, len(train) - 1), top_k=8,
        min_lag=1, min_abs_acf=.05, confidence_z=1.96, min_separation=2,
        detrend="mean", channel_chunk_size=64,
    )
    peaks = _channel_peak_matrix(prior, top_k=8)
    pair_lags, support = _coverage_pair_lags(peaks, args.min_channel_positive_support)
    dataset = Windows(
        train, args.seq_len, args.max_windows, next_patch=True,
        patch_len=args.pred_len, relation_lags=pair_lags,
        relation_pair_policy="contrastive_cluster",
    )
    sampler = ACFLagPairBatchSampler(
        dataset, batch_size=args.batch_size, lags=pair_lags,
        lag_balance="balanced", seed=args.seed, origins=dataset.starts,
        batch_packing="local",
    )
    relation = ForecastLagPairContrastiveLoss(
        prior.acf.astype(np.float32),
        selected_lags=sorted(set(peaks.ravel()) - {0}), channel_selected_lags=peaks,
    )
    active, spans = [], []
    # The dummy forecast isolates pair geometry from any learned representation.
    generator = torch.Generator().manual_seed(args.seed)
    for origins, _values, _future in DataLoader(dataset, batch_sampler=sampler):
        forecast = torch.randn(len(origins), args.pred_len, raw.shape[1], generator=generator)
        relation(forecast, origins)
        active.append(float(relation.last_active_anchor_fraction))
        spans.append(int(origins.max() - origins.min()))
    return {
        "name": name,
        "path": str(path),
        "data_format": data_format,
        "train_rows": int(len(train)),
        "channels": int(raw.shape[1]),
        "pair_lags": pair_lags,
        "pair_lag_support": {str(lag): support[lag] for lag in pair_lags},
        "active_anchor_fraction_mean": float(np.mean(active)),
        "active_anchor_fraction_min": float(np.min(active)),
        "batch_span_max": int(max(spans)),
        "batch_count": int(len(active)),
        "test_metrics_produced": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME:PATH:csv|npz|txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=192)
    parser.add_argument("--train-fraction", type=float, default=.6)
    parser.add_argument("--max-lag", type=int, default=720)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-channel-positive-support", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=5510)
    args = parser.parse_args()
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("batch-size must be even and at least four for local InfoNCE negatives")
    profiles = [_profile(name, path, data_format, args) for name, path, data_format in (
        _source_spec(item) for item in args.source
    )]
    result = {"profiles": profiles, "test_metrics_produced": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
