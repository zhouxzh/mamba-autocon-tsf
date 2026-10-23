#!/usr/bin/env python3
"""Select pretraining sources using only a target few-shot training prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import load_numeric_array_file
from utils.acf_guidance import fit_signed_acf_prior


def _load(path: Path, fmt: str):
    values = pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=np.float32) if fmt == "csv" else load_numeric_array_file(path, data_format=fmt)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite [time, channels] values: {}".format(path))
    return values


def _spec(value: str):
    name, path, fmt = value.split(":", 2)
    return name, Path(path).resolve(), fmt


def _digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _signature(values):
    prior = fit_signed_acf_prior(values, max_lag=min(720, len(values) - 1), top_k=8, min_lag=1, min_abs_acf=.05, confidence_z=1.96, min_separation=2, detrend="mean", channel_chunk_size=64)
    acf = prior.acf.mean(axis=0)
    signature = np.concatenate((acf[1:337], np.asarray([np.abs(prior.acf[:, 1:337]).mean(), prior.acf[:, 1:337].std()], dtype=np.float32)))
    signature /= np.linalg.norm(signature).clip(min=1e-12)
    return signature.astype(np.float32), list(prior.common_lags(top_k=8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="NAME:PATH:csv|npz|txt")
    parser.add_argument("--source", action="append", required=True, help="NAME:PATH:csv|npz|txt")
    parser.add_argument("--few-shot-fraction", type=float, choices=[.1, .25], required=True)
    parser.add_argument("--train-fraction", type=float, default=.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or not 0 < args.train_fraction < 1:
        raise ValueError("invalid source-selection geometry")
    target_name, target_path, target_fmt = _spec(args.target)
    target_values = _load(target_path, target_fmt)
    target_rows = int(len(target_values) * args.train_fraction * args.few_shot_fraction)
    target_signature, target_lags = _signature(target_values[:target_rows])
    ranking = []
    for item in args.source:
        name, path, fmt = _spec(item)
        if name == target_name:
            raise ValueError("target must not appear in the source pool")
        values = _load(path, fmt)
        source_rows = int(len(values) * args.train_fraction)
        signature, lags = _signature(values[:source_rows])
        ranking.append({"name": name, "path": str(path), "format": fmt, "sha256": _digest(path), "source_train_rows": source_rows, "similarity": float(np.dot(target_signature, signature)), "source_lags": lags})
    ranking.sort(key=lambda item: (-item["similarity"], item["name"]))
    result = {"target": {"name": target_name, "path": str(target_path), "sha256": _digest(target_path), "few_shot_rows": target_rows, "target_lags": target_lags}, "few_shot_fraction": args.few_shot_fraction, "train_fraction": args.train_fraction, "ranking": ranking, "selected": ranking[:args.top_k], "main_validation_accessed": False, "test_metrics_produced": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
