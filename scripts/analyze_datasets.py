#!/usr/bin/env python
"""Create a reproducible, train-only audit of the datasets shipped with this repo.

The script never writes to the source data directory.  It records file hashes,
split sizes, missingness, and ACF summaries for raw, first-difference/log-return,
absolute-difference, and squared-difference signals.  ACF statistics are fit on
the training segment only, matching the forecasting data loaders.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


DATASETS = {
    "ETTh1": ("dataset/ETT-small/ETTh1.csv", "OT", "ett_hour"),
    "ETTh2": ("dataset/ETT-small/ETTh2.csv", "OT", "ett_hour"),
    "ETTm1": ("dataset/ETT-small/ETTm1.csv", "OT", "ett_minute"),
    "ETTm2": ("dataset/ETT-small/ETTm2.csv", "OT", "ett_minute"),
    "weather": ("dataset/weather/weather.csv", "OT", "custom"),
    "exchange_rate": ("dataset/exchange_rate/exchange_rate.csv", "OT", "custom"),
    # The shipped CSV names the target ``OT``.  ``ILI`` was a stale alias and
    # caused this audit to report a target that the forecasting loader never
    # uses (the fallback happened to recover the last numeric column).
    "national_illness": ("dataset/illness/national_illness.csv", "OT", "custom"),
}


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_bounds(n_rows, kind, seq_len):
    if kind == "ett_hour":
        train_end = min(12 * 30 * 24, n_rows)
        val_end = min(12 * 30 * 24 + 4 * 30 * 24, n_rows)
    elif kind == "ett_minute":
        train_end = min(12 * 30 * 24 * 4, n_rows)
        val_end = min(12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, n_rows)
    else:
        train_end = int(n_rows * 0.6)
        val_end = n_rows - int(n_rows * 0.2)
    return train_end, val_end, n_rows


def safe_signal(values, name):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if name == "raw":
        return values
    if name == "diff":
        return np.diff(values)
    if name == "log_return":
        positive = np.maximum(values, np.finfo(np.float64).tiny)
        return np.diff(np.log(positive))
    diff = np.diff(values)
    if name == "abs_diff":
        return np.abs(diff)
    if name == "squared_diff":
        return diff ** 2
    raise ValueError(name)


def acf_summary(values, max_lag):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4 or np.std(values) == 0:
        return {"n": int(values.size), "acf1": np.nan, "acf24": np.nan,
                "acf96": np.nan, "significant_lags": 0, "mean": float(np.mean(values)) if values.size else np.nan,
                "std": float(np.std(values)) if values.size else np.nan, "acf": []}
    lag = min(int(max_lag), values.size - 1)
    values_acf = acf(values, nlags=lag, fft=True, adjusted=False)
    threshold = 1.96 / np.sqrt(values.size)
    return {
        "n": int(values.size),
        "acf1": float(values_acf[1]) if lag >= 1 else np.nan,
        "acf24": float(values_acf[24]) if lag >= 24 else np.nan,
        "acf96": float(values_acf[96]) if lag >= 96 else np.nan,
        "significant_lags": int(np.sum(np.abs(values_acf[1:]) > threshold)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "acf": values_acf.tolist(),
    }


def plot_acf(dataset, signals, output_path, max_lag):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for axis, (name, values) in zip(axes.flat, signals.items()):
        summary = acf_summary(values, max_lag)
        acf_values = np.asarray(summary["acf"])
        if acf_values.size:
            axis.plot(np.arange(acf_values.size), acf_values, linewidth=0.8, color="#1f5a85")
            axis.axhline(0.0, color="#333333", linewidth=0.6)
            axis.axhline(1.96 / np.sqrt(summary["n"]), color="#b23a48", linewidth=0.6, linestyle="--")
            axis.axhline(-1.96 / np.sqrt(summary["n"]), color="#b23a48", linewidth=0.6, linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("lag")
        axis.set_ylabel("ACF")
        axis.set_xlim(0, max_lag)
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(signals):]:
        axis.axis("off")
    fig.suptitle(f"{dataset}: training-only autocorrelation", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_dataset(root, dataset, config, output_dir, max_lag):
    relative_path, target, kind = config
    path = root / relative_path
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    if target not in frame.columns:
        target = numeric.columns[-1]
    train_end, val_end, test_end = split_bounds(len(frame), kind, 96)
    train = frame.iloc[:train_end]
    target_values = pd.to_numeric(train[target], errors="coerce").to_numpy()
    signals = {name: safe_signal(target_values, name) for name in
               ("raw", "diff", "log_return", "abs_diff", "squared_diff")}

    rows = []
    for signal_name, values in signals.items():
        summary = acf_summary(values, max_lag)
        summary.pop("acf", None)
        rows.append({"dataset": dataset, "target": target, "signal": signal_name,
                     "source": relative_path, "split": "train", **summary})

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_acf(dataset, signals, output_dir / f"{dataset}_acf.png", max_lag)
    return {
        "dataset": dataset,
        "source": relative_path,
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "numeric_columns": int(len(numeric.columns)),
        "target": target,
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame["date"].duplicated().sum()) if "date" in frame else None,
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_end": int(test_end),
        "train_windows_seq96_pred96": int(max(0, train_end - 96 - 96 + 1)),
        "rows_summary": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/data_analysis"))
    parser.add_argument("--max_lag", type=int, default=720)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifests = []
    rows = []
    for dataset, config in DATASETS.items():
        result = analyze_dataset(root, dataset, config, output_dir, args.max_lag)
        if result is None:
            continue
        manifests.append({key: value for key, value in result.items() if key != "rows_summary"})
        rows.extend(result["rows_summary"])

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "acf_summary.csv", index=False)
    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"repo_root": str(root), "max_lag": args.max_lag,
                   "datasets": manifests}, handle, indent=2)
    print(f"Analyzed {len(manifests)} datasets; outputs written to {output_dir}")


if __name__ == "__main__":
    main()
