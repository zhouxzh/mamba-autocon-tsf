#!/usr/bin/env python3
"""Read the validated cross-domain MambaTSv3 fusion screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = {
    "fusion_candidate_h96_validation": ("candidate_h96", 9, True),
    "fusion_baselines_h96_validation": ("baselines_h96", 63, False),
    "fusion_candidate_ili_h48_validation": ("candidate_ili", 3, True),
    "fusion_baselines_ili_h48_validation": ("baselines_ili", 21, False),
}
CANDIDATE = "MambaTSv3AutoCon"
SEEDS = (2411, 2412, 2413)
COORDINATES = (("ETTh1", 96), ("weather", 96), ("exchange_rate", 96), ("national_illness", 48))


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, phase: str, expected: int, candidate_phase: bool) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {phase}")
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError(f"missing strict provenance: {phase}")
    rows, seen = [], set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        job = json.loads(line)
        job_id = job.get("job_id")
        if not job_id or job_id in seen or job.get("status") != "ok":
            raise ValueError(f"bad manifest job: {job_id}")
        seen.add(job_id)
        if job.get("source_tree_sha256_before") != source or job.get("source_tree_sha256_after") != source:
            raise ValueError(f"source mismatch: {job_id}")
        result = Path(job["expected_artifact"]).parent
        if (result / "stream_metrics.json").exists() or (result / "metrics.npy").exists():
            raise ValueError(f"test artifact in validation job: {job_id}")
        config, complete = load(result / "config.json"), load(result / "training_complete.json")
        if config.get("skip_test") is not True or complete.get("status") != "trained_without_test":
            raise ValueError(f"validation contract mismatch: {job_id}")
        if candidate_phase:
            if config.get("model") != CANDIDATE or config.get("mamba_ts_repr_forecast_fusion") is not True:
                raise ValueError(f"candidate fusion binding mismatch: {job_id}")
            if float(config.get("AutoCon_lambda", -1)) != 0.001:
                raise ValueError(f"candidate lambda mismatch: {job_id}")
        elif config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError(f"baseline inherited fusion flag: {job_id}")
        checkpoint = Path(job["command"][job["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"missing finite validation loss: {job_id}")
        rows.append({"phase": phase, "dataset": str(config["data"]), "horizon": int(config["pred_len"]),
                     "seed": int(config["seed"]), "model": str(config["model"]),
                     "best_validation_scaled_mse": min(losses),
                     "elapsed_sec": float(job["elapsed_sec"]), "epochs_ran": int(job["epochs_ran"])})
    if len(rows) != expected:
        raise ValueError(f"coverage mismatch for {phase}")
    return source, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sources, rows = set(), []
    for phase, (directory, expected, candidate_phase) in PHASES.items():
        source, phase_rows = read_phase(args.artifact_root / directory, phase, expected, candidate_phase)
        sources.add(source)
        rows.extend(phase_rows)
    if len(sources) != 1:
        raise ValueError("phases do not share one source snapshot")
    index = {(r["dataset"], r["horizon"], r["seed"], r["model"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate coordinate")
    summary = []
    for dataset, horizon in COORDINATES:
        candidate = [index[(dataset, horizon, seed, CANDIDATE)] for seed in SEEDS]
        baselines = sorted({row["model"] for row in rows if row["dataset"] == dataset and row["horizon"] == horizon and row["model"] != CANDIDATE})
        for baseline in baselines:
            refs = [index[(dataset, horizon, seed, baseline)] for seed in SEEDS]
            ratios = [left["best_validation_scaled_mse"] / right["best_validation_scaled_mse"] for left, right in zip(candidate, refs)]
            summary.append({"dataset": dataset, "horizon": horizon, "baseline": baseline,
                            "candidate_mse_mean": statistics.fmean(r["best_validation_scaled_mse"] for r in candidate),
                            "baseline_mse_mean": statistics.fmean(r["best_validation_scaled_mse"] for r in refs),
                            "mean_relative_change_pct": 100 * (statistics.fmean(ratios) - 1),
                            "candidate_wins": sum(ratio < 1 for ratio in ratios), "n": len(ratios),
                            "candidate_elapsed_mean_sec": statistics.fmean(r["elapsed_sec"] for r in candidate),
                            "baseline_elapsed_mean_sec": statistics.fmean(r["elapsed_sec"] for r in refs)})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "validation_rows.csv", rows)
    write_csv(out / "paired_summary.csv", summary)
    lines = ["# Cross-domain MambaTSv3 AutoCon Fusion Validation", "", "Validation-only fresh-seed comparison. Every phase has passed strict validation and no test artifacts were produced.", "", "| Dataset | H | Baseline | Candidate MSE | Baseline MSE | Change | Wins |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {horizon} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {mean_relative_change_pct:+.3f}% | {candidate_wins}/{n} |".format(**row))
    lines.extend(["", "This screen compares the fusion candidate to external baselines. It has no exact fusion-off or lambda-zero control, so it does not identify a causal AutoCon or fusion effect."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "readout_metadata.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))


if __name__ == "__main__":
    main()
