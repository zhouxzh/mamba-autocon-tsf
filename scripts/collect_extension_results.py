#!/usr/bin/env python
"""Collect exploratory extension results without changing their provenance.

The extension runner stores one result directory per model/seed, often below
an additional model-specific directory.  This utility recursively reads the
compact metric artifacts, joins optional naive/resource records, and writes a
seed-level summary.  It never selects a model or rewrites metric artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_identity import as_bool, canonical_model


def _ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))


def _model_name(config: dict) -> str:
    return canonical_model(config)


def _load_metric_row(config_path: Path) -> dict | None:
    result_dir = config_path.parent
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    values_path = result_dir / "metrics.npy"
    if values_path.is_file():
        try:
            values = np.asarray(np.load(values_path), dtype=float).reshape(-1)
        except (OSError, ValueError):
            return None
    else:
        stream_path = result_dir / "stream_metrics.json"
        if not stream_path.is_file():
            return None
        try:
            stream = json.loads(stream_path.read_text(encoding="utf-8"))
            values = np.asarray([
                stream["scaled"][key] for key in ("mae", "mse", "rmse", "mape", "mspe")
            ], dtype=float)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
    if values.size < 3 or not np.all(np.isfinite(values[:3])):
        return None
    is_autocon = as_bool(config.get("AutoCon", False))
    original = np.full(5, np.nan, dtype=float)
    original_path = result_dir / "original_metrics.npy"
    if original_path.is_file():
        try:
            loaded = np.asarray(np.load(original_path), dtype=float).reshape(-1)
            original[: min(loaded.size, 5)] = loaded[:5]
        except (OSError, ValueError):
            pass
    return {
        "dataset": config.get("data"),
        "model": _model_name(config),
        "features": config.get("features"),
        "seq_len": int(config.get("seq_len", 0)),
        "pred_len": int(config.get("pred_len", 0)),
        "seed": int(config.get("run_seed", config.get("seed", 0))),
        "autocon": is_autocon,
        "autocon_lambda": float(config.get("AutoCon_lambda", 0.0)) if is_autocon else 0.0,
        "enc_in": int(config.get("enc_in", 0)),
        "c_out": int(config.get("c_out", 0)),
        "mae": float(values[0]), "mse": float(values[1]), "rmse": float(values[2]),
        "mape": float(values[3]) if values.size > 3 else np.nan,
        "mspe": float(values[4]) if values.size > 4 else np.nan,
        "original_mae": float(original[0]), "original_mse": float(original[1]),
        "original_rmse": float(original[2]),
        "config_path": str(config_path.resolve()),
    }


def load_results(root: Path) -> pd.DataFrame:
    rows = []
    for config_path in sorted(root.rglob("config.json")):
        row = _load_metric_row(config_path)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def load_naive(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for _, item in frame.iterrows():
            rows.append({
                "dataset": item["dataset"], "model": "NaiveLast" if item["model"] == "last" else "NaiveMean",
                "features": item["features"], "seq_len": int(item["seq_len"]),
                "pred_len": int(item["pred_len"]), "seed": 0, "autocon": False,
                "autocon_lambda": 0.0, "enc_in": int(item.get("enc_in", 0)),
                "c_out": int(item.get("c_out", 0)), "mae": float(item["scaled_mae"]),
                "mse": float(item["scaled_mse"]), "rmse": float(item["scaled_rmse"]),
                "mape": np.nan, "mspe": np.nan,
                "original_mae": float(item.get("original_mae", np.nan)),
                "original_mse": float(item.get("original_mse", np.nan)),
                "original_rmse": float(item.get("original_rmse", np.nan)),
                "config_path": str(path.resolve()),
            })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = ["dataset", "model", "features", "seq_len", "pred_len", "autocon_lambda"]
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, values))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        for metric in ("mae", "mse", "rmse", "original_mse"):
            samples = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(samples))
            row[f"{metric}_std"] = float(np.nanstd(samples, ddof=1)) if np.sum(np.isfinite(samples)) > 1 else 0.0
            row[f"{metric}_ci95"] = _ci95(samples)
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["dataset", "pred_len", "mse_mean"])


def add_resource_columns(summary: pd.DataFrame, resource_paths: list[Path]) -> pd.DataFrame:
    if summary.empty or not resource_paths:
        return summary
    records = []
    for path in resource_paths:
        if path.is_file():
            try:
                records.append(pd.read_csv(path))
            except (OSError, pd.errors.ParserError):
                continue
    if not records:
        return summary
    resources = pd.concat(records, ignore_index=True)
    resources["model"] = resources["model"].astype(str)
    keep = [column for column in ("model", "dataset", "features", "seq_len", "pred_len",
                                  "parameter_total", "latency_ms_mean", "synthetic_input_sequences_per_sec",
                                  "steady_state_peak_allocated_mb") if column in resources]
    resources = resources[keep].drop_duplicates(["model", "dataset", "features", "seq_len", "pred_len"])
    return summary.merge(resources, on=["model", "dataset", "features", "seq_len", "pred_len"], how="left")


def write_markdown(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Extension benchmark summary",
        "",
        "Metrics are train-standardized. Trained rows are aggregated across seeds; naive rows have seed 0.",
        "This report is descriptive and does not perform model or hyperparameter selection.",
        "",
    ]
    if summary.empty:
        lines.append("No valid metric artifacts were found.")
    else:
        columns = ["dataset", "model", "pred_len", "n_seeds", "mse_mean", "mse_ci95",
                   "mae_mean", "mae_ci95", "parameter_total", "latency_ms_mean"]
        table = summary[[column for column in columns if column in summary]].copy()
        for column in ("mse_mean", "mse_ci95", "mae_mean", "mae_ci95", "latency_ms_mean"):
            if column in table:
                table[column] = table[column].map(lambda value: "" if not np.isfinite(value) else f"{value:.6f}")
        try:
            lines.append(table.to_markdown(index=False))
        except (ImportError, ModuleNotFoundError):
            lines.append(table.to_string(index=False))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--naive", type=Path, action="append", default=[])
    parser.add_argument("--resources", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trained = load_results(args.results_root.resolve())
    naive = load_naive([path.resolve() for path in args.naive])
    frame = pd.concat([trained, naive], ignore_index=True, sort=False)
    summary = add_resource_columns(summarize(frame), [path.resolve() for path in args.resources])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "raw_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    write_markdown(summary, output / "summary.md")
    print(f"Collected {len(frame)} rows and {len(summary)} groups into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
