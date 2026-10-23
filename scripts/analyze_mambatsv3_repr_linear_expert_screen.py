#!/usr/bin/env python3
"""Strict validation-only readout for the MambaTS v3 linear-expert retry."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SEEDS = (2021, 2022, 2023)
COORDINATES = {
    "exchange_rate": (96, 336),
    "national_illness": (24, 60),
}
MODEL = "MambaTSv3AutoCon"


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase_rows(root: Path, expected_jobs: int, expert_enabled: bool):
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_jobs:
        raise ValueError("phase validation failed: {}".format(root))
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    rows = []
    seen = set()
    manifest = root / "logs" / "job_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source snapshot mismatch: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        if config.get("skip_test") is not True:
            raise ValueError("test boundary violated: {}".format(job_id))
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact present: {}".format(job_id))
        if config.get("model") != MODEL or config.get("AutoCon") is not True:
            raise ValueError("candidate contract mismatch: {}".format(job_id))
        if float(config.get("AutoCon_lambda", -1.0)) != 0.001:
            raise ValueError("lambda mismatch: {}".format(job_id))
        if bool(config.get("mamba_ts_repr_linear_expert", False)) is not expert_enabled:
            raise ValueError("linear expert arm mismatch: {}".format(job_id))
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        loss = load(checkpoint / "loss_logs.json")
        values = [float(value) for value in loss.get("val_loss", [])]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("missing/non-finite validation loss: {}".format(job_id))
        rows.append({
            "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]),
            "arm": "expert_on" if expert_enabled else "expert_off",
            "best_val_mse": min(values),
            "elapsed_sec": float(record["elapsed_sec"]),
            "best_epoch": int(record["best_epoch"]),
        })
    expected = {(dataset, horizon, seed) for dataset, horizons in COORDINATES.items() for horizon in horizons for seed in SEEDS}
    found = {(row["dataset"], row["horizon"], row["seed"]) for row in rows}
    if len(rows) != expected_jobs or found != expected:
        raise ValueError("coordinate coverage mismatch: {}".format(root))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = phase_rows(args.control_root, 12, False) + phase_rows(args.expert_root, 12, True)
    index = {(row["dataset"], row["horizon"], row["seed"], row["arm"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate paired coordinate")
    summary = []
    paired_rows = []
    for dataset, horizon in sorted((d, h) for d, hs in COORDINATES.items() for h in hs):
        deltas = []
        ratios = []
        elapsed_changes = []
        wins = 0
        for seed in SEEDS:
            control = index[(dataset, horizon, seed, "expert_off")]
            expert = index[(dataset, horizon, seed, "expert_on")]
            delta = expert["best_val_mse"] - control["best_val_mse"]
            ratio = expert["best_val_mse"] / control["best_val_mse"]
            deltas.append(delta)
            ratios.append(ratio)
            elapsed_changes.append(expert["elapsed_sec"] - control["elapsed_sec"])
            wins += int(delta < 0.0)
            paired_rows.append({
                "dataset": dataset, "horizon": horizon, "seed": seed,
                "control_best_val_mse": control["best_val_mse"],
                "expert_best_val_mse": expert["best_val_mse"],
                "mse_change_pct": 100.0 * (ratio - 1.0),
                "control_elapsed_sec": control["elapsed_sec"],
                "expert_elapsed_sec": expert["elapsed_sec"],
            })
        summary.append({
            "dataset": dataset, "horizon": horizon,
            "control_mse_mean": statistics.fmean(index[(dataset, horizon, seed, "expert_off")]["best_val_mse"] for seed in SEEDS),
            "expert_mse_mean": statistics.fmean(index[(dataset, horizon, seed, "expert_on")]["best_val_mse"] for seed in SEEDS),
            "mean_mse_change_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
            "mean_delta": statistics.fmean(deltas),
            "expert_wins": wins, "n": len(SEEDS),
            "control_elapsed_mean_sec": statistics.fmean(index[(dataset, horizon, seed, "expert_off")]["elapsed_sec"] for seed in SEEDS),
            "expert_elapsed_mean_sec": statistics.fmean(index[(dataset, horizon, seed, "expert_on")]["elapsed_sec"] for seed in SEEDS),
            "mean_elapsed_change_sec": statistics.fmean(elapsed_changes),
        })
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (("validation_rows.csv", rows), ("paired_rows.csv", paired_rows), ("paired_summary.csv", summary)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "# MambaTS v3 Representation-Gated Linear Expert",
        "",
        "Validation-only paired retry. The original expert-off artifacts are preserved; the corrected expert-on arm uses a new source snapshot and no test metrics were generated.",
        "",
        "| Dataset | H | Expert-off MSE | Expert-on MSE | Change | Expert wins | Off elapsed | On elapsed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append("| {dataset} | {horizon} | {control_mse_mean:.6f} | {expert_mse_mean:.6f} | {mean_mse_change_pct:+.3f}% | {expert_wins}/{n} | {control_elapsed_mean_sec:.2f}s | {expert_elapsed_mean_sec:.2f}s |".format(**item))
    lines.extend([
        "",
        "Negative MSE change favors the expert. This screen is architectural exploration and cannot support a test-performance claim or select a final model without an independent confirmation protocol.",
    ])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False})


if __name__ == "__main__":
    main()
