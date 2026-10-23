#!/usr/bin/env python3
"""Strict validation-only reader for the MambaTS v6 H96 screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = (
    "DLinear", "NLinear", "PatchTST", "iTransformer", "MambaTSv3AutoCon",
    "MambaTSv6AutoCon", "MambaSMAutoCon",
)
DATASETS = ("ETTh1", "weather", "exchange_rate", "national_illness")
SEEDS = (2021, 2022, 2023)
CANDIDATE = "MambaTSv6AutoCon"
AUTOCON = frozenset(("MambaTSv3AutoCon", "MambaTSv6AutoCon", "MambaSMAutoCon"))


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.phase_root.resolve()
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != 84:
        raise ValueError("invalid phase validation")
    source = load(root / "provenance.json").get("source_tree_sha256")
    if not source:
        raise ValueError("missing source snapshot hash")

    rows, seen = [], set()
    manifest = root / "logs" / "job_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source snapshot mismatch")
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        model = str(config.get("model"))
        if model not in MODELS or config.get("skip_test") is not True:
            raise ValueError("config boundary mismatch")
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact present")
        if model in AUTOCON:
            if config.get("AutoCon") is not True or float(config.get("AutoCon_lambda", -1.0)) != 0.001:
                raise ValueError("AutoCon contract mismatch")
        if model == CANDIDATE and config.get("mamba_ts_variant") != "v6":
            raise ValueError("MambaTS v6 runtime variant mismatch")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = load(checkpoint / "loss_logs.json").get("val_loss", [])
        values = [float(value) for value in losses]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("invalid validation loss")
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]), "model": model,
            "best_val_mse": min(values), "elapsed_sec": float(record["elapsed_sec"]),
        })

    expected = {(dataset, 96, seed, model) for dataset in DATASETS for seed in SEEDS for model in MODELS}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if len(rows) != 84 or found != expected:
        raise ValueError("coordinate coverage mismatch")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)

    summary = []
    for dataset in DATASETS:
        means = {
            model: statistics.fmean(row["best_val_mse"] for row in grouped[(dataset, model)])
            for model in MODELS
        }
        candidate = {row["seed"]: row for row in grouped[(dataset, CANDIDATE)]}
        winner = min(means, key=means.get)
        for baseline in MODELS:
            reference = {row["seed"]: row for row in grouped[(dataset, baseline)]}
            changes = [
                100.0 * (candidate[seed]["best_val_mse"] - reference[seed]["best_val_mse"])
                / reference[seed]["best_val_mse"] for seed in SEEDS
            ]
            summary.append({
                "dataset": dataset, "horizon": 96, "best_model": winner,
                "baseline": baseline, "candidate_mse_mean": means[CANDIDATE],
                "baseline_mse_mean": means[baseline],
                "mean_mse_change_pct": statistics.fmean(changes),
                "candidate_wins": sum(change < 0.0 for change in changes), "n": len(SEEDS),
                "candidate_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in candidate.values()),
                "baseline_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in reference.values()),
            })

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("validation_rows.csv", rows), ("paired_summary.csv", summary)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "# MambaTS v6 H96 Validation Screen", "",
        "Validation-only, three paired seeds; no test metrics were produced.", "",
        "| Dataset | Best model | Candidate MSE | Baseline | Baseline MSE | Change | Wins / 3 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            "| {dataset} | {best_model} | {candidate_mse_mean:.6f} | {baseline} | "
            "{baseline_mse_mean:.6f} | {mean_mse_change_pct:+.3f}% | "
            "{candidate_wins}/{n} |".format(**item)
        )
    lines.extend((
        "", "Negative change favors MambaTS v6. This is an exploratory architecture "
        "screen, not a test-performance or cross-dataset non-inferiority claim.",
    ))
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False})


if __name__ == "__main__":
    main()
