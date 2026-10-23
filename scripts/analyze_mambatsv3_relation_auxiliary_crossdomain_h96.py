#!/usr/bin/env python3
"""Strict reader for the cross-domain relation-auxiliary-only H96 screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


CANDIDATE = "MambaTSv3AutoCon"
BASELINES = ("RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")
DATASETS = ("ETTh1", "weather")
SEEDS = (3331, 3332, 3333)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, expected_jobs: int, models: set[str], candidate: bool) -> tuple[list[dict], str]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"phase validation failed: {root}")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source:
        raise ValueError(f"missing source hash: {root}")
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid job: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        dataset, model, seed = str(config.get("data")), str(config.get("model")), int(config.get("seed", -1))
        if dataset not in DATASETS or model not in models or seed not in SEEDS or int(config.get("pred_len", -1)) != 96:
            raise ValueError(f"coordinate mismatch: {job_id}")
        if config.get("skip_test") is not True or any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")):
            raise ValueError(f"test boundary violation: {job_id}")
        if candidate:
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False:
                raise ValueError(f"candidate contract mismatch: {job_id}")
        elif config.get("AutoCon") is True or config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError(f"baseline isolation mismatch: {job_id}")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses:
            raise ValueError(f"missing validation trajectory: {job_id}")
        rows.append({"dataset": dataset, "horizon": 96, "seed": seed, "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected_jobs:
        raise ValueError(f"coverage mismatch: {root}")
    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_source = read_phase(args.artifact_root / "candidate", 6, {CANDIDATE}, True)
    baselines, baseline_source = read_phase(args.artifact_root / "baselines", 30, set(BASELINES), False)
    if candidate_source != baseline_source:
        raise ValueError("source snapshots differ")
    rows = candidate + baselines
    index = {(r["dataset"], r["model"], r["seed"]): r for r in rows}
    expected = {(dataset, model, seed) for dataset in DATASETS for model in (CANDIDATE,) + BASELINES for seed in SEEDS}
    if set(index) != expected:
        raise ValueError("incomplete coordinate matrix")
    summary, grouped = [], defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    for dataset in DATASETS:
        for baseline in BASELINES:
            left = [index[(dataset, CANDIDATE, seed)]["best_val_mse"] for seed in SEEDS]
            right = [index[(dataset, baseline, seed)]["best_val_mse"] for seed in SEEDS]
            deltas = [a - b for a, b in zip(left, right)]
            summary.append({"dataset": dataset, "baseline": baseline, "candidate_mse": statistics.fmean(left), "baseline_mse": statistics.fmean(right), "relative_delta_pct": 100 * statistics.fmean(deltas) / statistics.fmean(right), "candidate_wins": sum(delta < 0 for delta in deltas), "n": 3})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# Cross-Domain Relation-Auxiliary-Only H96 Validation", "", "Validation-only exact paired matrix. Negative relative delta favors the candidate.", "", "| Dataset | Baseline | Candidate MSE | Baseline MSE | Relative delta | Wins |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {baseline} | {candidate_mse:.6f} | {baseline_mse:.6f} | {relative_delta_pct:+.2f}% | {candidate_wins}/{n} |".format(**row) for row in summary)
    lines.extend(["", f"All rows share source snapshot `{candidate_source}`. Test artifacts were not produced."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
