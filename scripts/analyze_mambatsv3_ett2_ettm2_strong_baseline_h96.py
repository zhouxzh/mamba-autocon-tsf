#!/usr/bin/env python3
"""Strict reader for the current-source ETTh2/ETTm2 H96 validation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


CANDIDATE = "MambaTSv3AutoCon"
BASELINES = ("RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")
DATASETS = ("ETTh2", "ETTm2")
SEEDS = (3191, 3192, 3193)


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, expected_jobs: int, expected_models: set[str]) -> tuple[list[dict], str]:
    phase = load(root / "phase_validation.json")
    if phase.get("valid") is not True or phase.get("expected_jobs") != expected_jobs:
        raise ValueError("phase validation failed: {}".format(root))
    source = load(root / "provenance.json").get("source_tree_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError("missing source hash")
    rows = []
    seen = set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = record.get("job_id")
        if job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest record: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source snapshot mismatch: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        model = str(config.get("model"))
        if model not in expected_models or config.get("skip_test") is not True:
            raise ValueError("model/test contract mismatch: {}".format(job_id))
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact found: {}".format(job_id))
        if model == CANDIDATE:
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not True:
                raise ValueError("candidate contract mismatch: {}".format(job_id))
        elif config.get("AutoCon") is True or config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError("baseline received candidate-only flag: {}".format(job_id))
        command = record["command"]
        checkpoint_root = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint_root / "loss_logs.json").get("val_loss", [])]
        if not losses:
            raise ValueError("missing validation trajectory: {}".format(job_id))
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]),
        })
    expected = {(dataset, 96, seed, model) for dataset in DATASETS for seed in SEEDS for model in expected_models}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if len(rows) != expected_jobs or found != expected:
        raise ValueError("coordinate coverage mismatch: {}".format(root))
    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_source = read_phase(args.artifact_root / "candidate_h96", 6, {CANDIDATE})
    baselines, baseline_source = read_phase(args.artifact_root / "baselines_h96", 30, set(BASELINES))
    if candidate_source != baseline_source:
        raise ValueError("arms have different source snapshots")
    rows = candidate + baselines
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    summaries = []
    for dataset in DATASETS:
        for model in (CANDIDATE,) + BASELINES:
            group = grouped[(dataset, model)]
            summaries.append({
                "dataset": dataset, "model": model, "mean_val_mse": statistics.fmean(r["best_val_mse"] for r in group),
                "std_val_mse": statistics.stdev(r["best_val_mse"] for r in group), "mean_elapsed_sec": statistics.fmean(r["elapsed_sec"] for r in group), "n": len(group),
            })
    index = {(r["dataset"], r["seed"], r["model"]): r for r in rows}
    paired = []
    for dataset in DATASETS:
        for baseline in BASELINES:
            deltas = [index[(dataset, seed, CANDIDATE)]["best_val_mse"] - index[(dataset, seed, baseline)]["best_val_mse"] for seed in SEEDS]
            candidate_values = [index[(dataset, seed, CANDIDATE)]["best_val_mse"] for seed in SEEDS]
            baseline_values = [index[(dataset, seed, baseline)]["best_val_mse"] for seed in SEEDS]
            paired.append({
                "dataset": dataset, "baseline": baseline, "candidate_mean_val_mse": statistics.fmean(candidate_values),
                "baseline_mean_val_mse": statistics.fmean(baseline_values), "mean_delta_mse": statistics.fmean(deltas),
                "relative_delta_pct": 100.0 * statistics.fmean(deltas) / statistics.fmean(baseline_values),
                "candidate_wins": sum(delta < 0 for delta in deltas), "n": len(deltas),
            })
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    for name, values in (("validation_rows.csv", rows), ("summary.csv", summaries), ("paired_summary.csv", paired)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ["# Current-source MambaTSv3 AutoCon ETTh2/ETTm2 H96 Validation", "", "Validation-only strict readout. No test metric or prediction artifact was read or produced.", "", "| Dataset | Model | Mean validation MSE | SD | Mean job sec | Seeds |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append("| {dataset} | {model} | {mean_val_mse:.6f} | {std_val_mse:.6f} | {mean_elapsed_sec:.3f} | {n} |".format(**row))
    lines.extend(["", "## Paired Candidate Deltas", "", "Negative delta favors MambaTSv3AutoCon.", "", "| Dataset | Baseline | Candidate MSE | Baseline MSE | Delta | Relative delta | Candidate wins |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in paired:
        lines.append("| {dataset} | {baseline} | {candidate_mean_val_mse:.6f} | {baseline_mean_val_mse:.6f} | {mean_delta_mse:.6f} | {relative_delta_pct:.2f}% | {candidate_wins}/{n} |".format(**row))
    lines.extend(["", "All rows share source snapshot `{}`. These are validation results only and cannot support held-out or causal claims.".format(candidate_source)])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_rows": len(paired), "source_tree_sha256": candidate_source, "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
