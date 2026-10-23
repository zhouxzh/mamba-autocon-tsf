#!/usr/bin/env python3
"""Strict reader for the current-source MambaTSv3 AutoCon H96 validation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


CANDIDATE = "MambaTSv3AutoCon"
BASELINES = ("RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")
DATASETS = ("ETTh1", "weather", "electricity", "traffic")
SEEDS = (3141, 3142, 3143)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, expected_jobs: int, expected_models: set[str]) -> tuple[list[dict], str]:
    validation = load_json(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_jobs:
        raise ValueError("phase validation failed for {}".format(root))
    source_hash = load_json(root / "provenance.json").get("source_tree_sha256")
    if not source_hash:
        raise ValueError("missing provenance source hash")

    records = {}
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = record.get("job_id")
        if job_id in records or record.get("status") != "ok":
            raise ValueError("invalid manifest record for {}".format(root))
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError("source snapshot mismatch for {}".format(job_id))
        records[job_id] = record
    if len(records) != expected_jobs:
        raise ValueError("manifest coverage mismatch for {}".format(root))

    rows = []
    for record in records.values():
        result_dir = Path(record["expected_artifact"]).parent
        config = load_json(result_dir / "config.json")
        model = config.get("model")
        if model not in expected_models or config.get("skip_test") is not True:
            raise ValueError("model or test boundary mismatch for {}".format(record["job_id"]))
        if any((result_dir / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact found in validation-only result")
        if model == CANDIDATE:
            if not (config.get("AutoCon") is True and config.get("mamba_ts_variant") == "v3" and config.get("mamba_ts_repr_forecast_fusion") is True):
                raise ValueError("candidate contract mismatch")
        elif config.get("AutoCon") is True or config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError("baseline received candidate-only mechanism")

        command = record["command"]
        checkpoint_root = Path(command[command.index("--checkpoints") + 1]) / result_dir.name
        losses = [float(value) for value in load_json(checkpoint_root / "loss_logs.json").get("val_loss", [])]
        if not losses:
            raise ValueError("validation trajectory missing")
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]),
            "epochs_ran": int(record.get("epochs_ran", 0)), "source_tree_sha256": source_hash,
        })
    expected = {(dataset, 96, seed, model) for dataset in DATASETS for seed in SEEDS for model in expected_models}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if found != expected:
        raise ValueError("coordinate mismatch for {}".format(root))
    return rows, source_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_rows, candidate_hash = read_phase(args.artifact_root / "candidate_h96", 12, {CANDIDATE})
    baseline_rows, baseline_hash = read_phase(args.artifact_root / "baselines_h96", 60, set(BASELINES))
    if candidate_hash != baseline_hash:
        raise ValueError("candidate and baselines have different source snapshots")
    rows = candidate_rows + baseline_rows

    summaries = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    for dataset in DATASETS:
        for model in (CANDIDATE,) + BASELINES:
            group = grouped[(dataset, model)]
            summaries.append({
                "dataset": dataset, "model": model, "mean_val_mse": statistics.fmean(row["best_val_mse"] for row in group),
                "std_val_mse": statistics.stdev(row["best_val_mse"] for row in group),
                "mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in group), "n": len(group),
            })

    by_coordinate = {(row["dataset"], row["seed"], row["model"]): row for row in rows}
    paired = []
    for dataset in DATASETS:
        candidate_values = [by_coordinate[(dataset, seed, CANDIDATE)]["best_val_mse"] for seed in SEEDS]
        for baseline in BASELINES:
            deltas = [
                by_coordinate[(dataset, seed, CANDIDATE)]["best_val_mse"] - by_coordinate[(dataset, seed, baseline)]["best_val_mse"]
                for seed in SEEDS
            ]
            baseline_values = [by_coordinate[(dataset, seed, baseline)]["best_val_mse"] for seed in SEEDS]
            paired.append({
                "dataset": dataset, "baseline": baseline,
                "candidate_mean_val_mse": statistics.fmean(candidate_values), "baseline_mean_val_mse": statistics.fmean(baseline_values),
                "mean_delta_mse": statistics.fmean(deltas),
                "relative_delta_pct": 100.0 * statistics.fmean(deltas) / statistics.fmean(baseline_values),
                "candidate_wins": sum(delta < 0.0 for delta in deltas), "n": len(deltas),
            })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("validation_rows.csv", rows), ("summary.csv", summaries), ("paired_summary.csv", paired)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = [
        "# Current-Source MambaTSv3 AutoCon H96 Validation", "",
        "Validation-only paired screen. No held-out test metric or prediction artifact was read or produced.", "",
        "| Dataset | Model | Mean validation MSE | SD | Mean job sec | Seeds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append("| {dataset} | {model} | {mean_val_mse:.6f} | {std_val_mse:.6f} | {mean_elapsed_sec:.3f} | {n} |".format(**row))
    lines.extend(["", "## Paired Candidate Deltas", "", "Negative delta favors MambaTSv3AutoCon.", "", "| Dataset | Baseline | Candidate MSE | Baseline MSE | Delta | Relative delta | Candidate wins |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in paired:
        lines.append("| {dataset} | {baseline} | {candidate_mean_val_mse:.6f} | {baseline_mean_val_mse:.6f} | {mean_delta_mse:.6f} | {relative_delta_pct:.2f}% | {candidate_wins}/{n} |".format(**row))
    lines.extend(["", "All rows share source snapshot `{}`. These are validation results only and cannot support held-out or causal claims.".format(candidate_hash)])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_rows": len(paired), "source_tree_sha256": candidate_hash, "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
