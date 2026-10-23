#!/usr/bin/env python3
"""Strict validation-only reader for the six-model baseline expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = ("DLinear", "NLinear", "PatchTST", "TimesNet", "Mamba1", "Mamba2")
DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
HORIZONS = (96, 336, 720)
SEEDS = (2021, 2022, 2023)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _best_validation_mse(checkpoint_dir: Path) -> float:
    losses = _load(checkpoint_dir / "loss_logs.json").get("val_loss", [])
    values = [float(value) for value in losses]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid validation loss: {}".format(checkpoint_dir))
    return min(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.phase_root.resolve()
    validation = _load(root / "phase_validation.json")
    expected_count = len(DATASETS) * len(HORIZONS) * len(SEEDS) * len(MODELS)
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_count:
        raise ValueError("phase is incomplete or invalid")
    source_hash = _load(root / "provenance.json").get("source_tree_sha256")
    if not source_hash:
        raise ValueError("missing source snapshot hash")

    rows, seen = [], set()
    manifest = root / "logs" / "job_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError("source snapshot mismatch")
        result_dir = Path(record["expected_artifact"]).parent
        config = _load(result_dir / "config.json")
        model = str(config.get("model"))
        if model not in MODELS or config.get("skip_test") is not True:
            raise ValueError("model or test boundary mismatch")
        if any((result_dir / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact present")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result_dir.name
        rows.append({
            "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]),
            "model": model,
            "best_val_mse": _best_validation_mse(checkpoint),
            "elapsed_sec": float(record["elapsed_sec"]),
            "best_epoch": int(record.get("best_epoch", -1)),
        })

    expected = {(dataset, horizon, seed, model) for dataset in DATASETS for horizon in HORIZONS for seed in SEEDS for model in MODELS}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if len(rows) != expected_count or found != expected:
        raise ValueError("coordinate coverage mismatch")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["horizon"], row["model"])].append(row)
    summaries, pairwise = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            means = {model: statistics.fmean(row["best_val_mse"] for row in grouped[(dataset, horizon, model)]) for model in MODELS}
            winner = min(means, key=means.get)
            for rank, model in enumerate(sorted(MODELS, key=means.get), 1):
                arm = grouped[(dataset, horizon, model)]
                summaries.append({
                    "dataset": dataset, "horizon": horizon, "rank": rank, "model": model,
                    "mean_val_mse": means[model],
                    "std_val_mse": statistics.stdev(row["best_val_mse"] for row in arm),
                    "mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in arm),
                    "best_model": winner,
                })
            for mamba in ("Mamba1", "Mamba2"):
                candidate = {row["seed"]: row for row in grouped[(dataset, horizon, mamba)]}
                for baseline in MODELS:
                    reference = {row["seed"]: row for row in grouped[(dataset, horizon, baseline)]}
                    deltas = [100.0 * (candidate[seed]["best_val_mse"] - reference[seed]["best_val_mse"]) / reference[seed]["best_val_mse"] for seed in SEEDS]
                    pairwise.append({
                        "dataset": dataset, "horizon": horizon, "mamba": mamba,
                        "baseline": baseline, "mamba_mean_val_mse": means[mamba],
                        "baseline_mean_val_mse": means[baseline],
                        "mean_relative_change_pct": statistics.fmean(deltas),
                        "mamba_wins": sum(delta < 0.0 for delta in deltas), "n": len(SEEDS),
                    })

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("validation_rows.csv", rows), ("ranking_summary.csv", summaries), ("mamba_pairwise_summary.csv", pairwise)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = ["# Six-Model Plain Baseline Expansion", "", "Validation-only, three paired seeds; no test metrics were produced.", "", "| Dataset | Horizon | Rank | Model | MSE mean +/- SD | Mean wall time (s) |", "| --- | ---: | ---: | --- | ---: | ---: |"]
    for row in summaries:
        lines.append("| {dataset} | {horizon} | {rank} | {model} | {mean_val_mse:.6f} +/- {std_val_mse:.6f} | {mean_elapsed_sec:.2f} |".format(**row))
    lines.extend(("", "Pairwise Mamba deltas are in `mamba_pairwise_summary.csv`; negative values favor the indicated Mamba arm. These are selection-stage validation results, not a test release or a speed claim."))
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "ranking_cells": len(summaries), "pairwise_cells": len(pairwise), "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
