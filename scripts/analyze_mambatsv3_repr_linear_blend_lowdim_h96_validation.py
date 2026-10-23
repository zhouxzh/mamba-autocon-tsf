#!/usr/bin/env python3
"""Strict readout for the bounded MambaTSv3/linear blend validation screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SEEDS = (3011, 3012, 3013)
COORDINATES = {
    "ETTh1": 96,
    "exchange_rate": 96,
    "national_illness": 24,
}
BASELINES = ("DLinear", "NLinear", "PatchTST")


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_rows(root: Path, expected_jobs: int, *, arm: str):
    phase = load(root / "phase_validation.json")
    if phase.get("valid") is not True or phase.get("expected_jobs") != expected_jobs:
        raise ValueError(f"phase validation failed: {root}")
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError(f"missing source snapshot: {root}")
    manifest = root / "logs" / "job_manifest.jsonl"
    rows = []
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source snapshot mismatch: {job_id}")
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        if config.get("skip_test") is not True:
            raise ValueError(f"test execution enabled: {job_id}")
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError(f"test artifact present: {job_id}")
        if arm == "blend":
            if config.get("model") != "MambaTSv3AutoCon" or config.get("mamba_ts_repr_linear_expert") is not True:
                raise ValueError(f"blend config mismatch: {job_id}")
            if config.get("mamba_ts_repr_linear_expert_mode") != "blend":
                raise ValueError(f"blend mode mismatch: {job_id}")
            if float(config.get("mamba_ts_repr_linear_blend_bias")) != -4.0:
                raise ValueError(f"blend bias mismatch: {job_id}")
        elif arm == "control":
            if config.get("model") != "MambaTSv3AutoCon" or config.get("mamba_ts_repr_linear_expert") is not False:
                raise ValueError(f"control config mismatch: {job_id}")
        else:
            if config.get("model") not in BASELINES or config.get("mamba_ts_repr_linear_expert") is not False:
                raise ValueError(f"baseline config mismatch: {job_id}")
        command = record["command"]
        checkpoints = Path(command[command.index("--checkpoints") + 1])
        loss_path = checkpoints / result.name / "loss_logs.json"
        losses = load(loss_path).get("val_loss", [])
        values = [float(value) for value in losses]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"missing/non-finite validation losses: {job_id}")
        rows.append({
            "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]),
            "model": str(config["model"]),
            "arm": arm,
            "best_val_mse": min(values),
            "elapsed_sec": float(record["elapsed_sec"]),
        })
    expected = {
        (dataset, horizon, seed)
        for dataset, horizon in COORDINATES.items()
        for seed in SEEDS
    }
    found = {(row["dataset"], row["horizon"], row["seed"]) for row in rows}
    if found != expected or len(rows) != expected_jobs:
        raise ValueError(f"coordinate coverage mismatch for {root}: {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = (
        _artifact_rows(root / "candidate", 9, arm="blend")
        + _artifact_rows(root / "control", 9, arm="control")
        + _artifact_rows(root / "baselines", 27, arm="baseline")
    )
    index = {(r["dataset"], r["horizon"], r["seed"], r["model"], r["arm"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate paired coordinate")

    summaries = []
    paired = []
    for dataset, horizon in COORDINATES.items():
        for seed in SEEDS:
            blend = index[(dataset, horizon, seed, "MambaTSv3AutoCon", "blend")]
            control = index[(dataset, horizon, seed, "MambaTSv3AutoCon", "control")]
            paired.append({
                "dataset": dataset, "horizon": horizon, "seed": seed,
                "blend_mse": blend["best_val_mse"],
                "control_mse": control["best_val_mse"],
                "change_vs_v3_pct": 100.0 * (blend["best_val_mse"] / control["best_val_mse"] - 1.0),
                "blend_elapsed_sec": blend["elapsed_sec"],
                "control_elapsed_sec": control["elapsed_sec"],
            })
        blend_values = [index[(dataset, horizon, seed, "MambaTSv3AutoCon", "blend")]["best_val_mse"] for seed in SEEDS]
        control_values = [index[(dataset, horizon, seed, "MambaTSv3AutoCon", "control")]["best_val_mse"] for seed in SEEDS]
        baseline_means = {
            model: statistics.fmean(index[(dataset, horizon, seed, model, "baseline")]["best_val_mse"] for seed in SEEDS)
            for model in BASELINES
        }
        strongest_model = min(baseline_means, key=baseline_means.get)
        strongest = baseline_means[strongest_model]
        wins = sum(int(b < c) for b, c in zip(blend_values, control_values))
        summaries.append({
            "dataset": dataset, "horizon": horizon,
            "blend_mse_mean": statistics.fmean(blend_values),
            "control_mse_mean": statistics.fmean(control_values),
            "change_vs_v3_pct": 100.0 * (statistics.fmean(blend_values) / statistics.fmean(control_values) - 1.0),
            "blend_wins_vs_v3": wins, "n": len(SEEDS),
            "strongest_baseline": strongest_model,
            "strongest_baseline_mse": strongest,
            "change_vs_strongest_pct": 100.0 * (statistics.fmean(blend_values) / strongest - 1.0),
            "baseline_means": baseline_means,
        })

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)

    lines = [
        "# MambaTSv3 AutoCon bounded Mamba/linear blend validation",
        "",
        "Validation-only strict readout. All 45 jobs completed with one source snapshot and no test metrics or test artifacts.",
        "",
        "| Dataset | H | Blend MSE | Exact v3 MSE | Change vs v3 | Wins | Strongest baseline | Change vs strongest | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    pass_count = 0
    for item in summaries:
        paired_gate = item["change_vs_v3_pct"] < 0.0 and item["blend_wins_vs_v3"] >= 2
        external_gate = item["change_vs_strongest_pct"] <= 2.0
        passed = paired_gate and external_gate
        pass_count += int(passed)
        lines.append(
            "| {dataset} | {horizon} | {blend_mse_mean:.6f} | {control_mse_mean:.6f} | {change_vs_v3_pct:+.3f}% | {blend_wins_vs_v3}/{n} | {strongest_baseline} | {change_vs_strongest_pct:+.3f}% | {gate} |".format(
                **item, gate="PASS" if passed else "FAIL"
            )
        )
        lines.append(
            "|  |  |  |  | baseline means: "
            + ", ".join(f"{model}={value:.6f}" for model, value in item["baseline_means"].items())
            + " |  |  |  |  |"
        )
    lines.extend([
        "",
        f"Datasets passing both gates: {pass_count}/3. Required for continuation: at least 2/3.",
        "",
        "Negative change favors the blend. This result is validation-only and does not authorize a held-out release; any continuation must freeze this architecture and register a new test plan.",
    ])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_cells": len(summaries), "datasets_passing": pass_count, "test_metrics_produced": False})


if __name__ == "__main__":
    main()
