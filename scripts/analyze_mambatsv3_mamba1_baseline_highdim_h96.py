#!/usr/bin/env python3
"""Strict reader for the plain Mamba-1 high-dimensional baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3351, 3352, 3353)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != "mamba1_h96_validation" or validation.get("expected_jobs") != 6:
        raise ValueError("invalid Mamba-1 phase validation")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError("invalid Mamba-1 provenance")
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")):
            raise ValueError(f"test artifact found: {job_id}")
        config = load(result / "config.json")
        if str(config.get("data")) not in DATASETS or config.get("model") != "Mamba1" or int(config.get("seed", -1)) not in SEEDS or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True:
            raise ValueError(f"coordinate or test contract mismatch: {job_id}")
        if config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError(f"Mamba-1 candidate feature leakage: {job_id}")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation loss: {job_id}")
        rows.append({"dataset": str(config["data"]), "horizon": 96, "seed": int(config["seed"]), "model": "Mamba1", "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != 6:
        raise ValueError("Mamba-1 coverage mismatch")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_source, baseline = read_phase(args.baseline_root)
    candidate_provenance = load(args.candidate_root / "provenance.json")
    candidate_source = str(candidate_provenance.get("source_tree_sha256", ""))
    if baseline_source != candidate_source:
        raise ValueError("candidate and Mamba-1 source snapshots differ")
    candidate_rows = []
    for raw in (args.candidate_root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); result = Path(record["expected_artifact"]).parent; config = load(result / "config.json")
        if str(config.get("data")) not in DATASETS or config.get("model") != "MambaTSv3AutoCon" or config.get("skip_test") is not True or config.get("mamba_ts_repr_forecast_fusion") is not False or config.get("AutoCon") is not True:
            raise ValueError("candidate reference contract mismatch")
        command = record["command"]; checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name; losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        candidate_rows.append({"dataset": str(config["data"]), "horizon": 96, "seed": int(config["seed"]), "model": "MambaTSv3AutoCon", "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    index = {(row["dataset"], row["model"], row["seed"]): row for row in candidate_rows + baseline}
    expected = {(dataset, model) for dataset in DATASETS for model in ("MambaTSv3AutoCon", "Mamba1")}
    if {(dataset, model) for dataset, model, _ in index} != expected:
        raise ValueError("candidate/Mamba-1 coordinate mismatch")
    summary = []
    for dataset in DATASETS:
        candidate_values = [row["best_val_mse"] for key, row in index.items() if key[0] == dataset and key[1] == "MambaTSv3AutoCon"]
        baseline_values = [row["best_val_mse"] for key, row in index.items() if key[0] == dataset and key[1] == "Mamba1"]
        summary.append({"dataset": dataset, "candidate_mse": statistics.fmean(candidate_values), "mamba1_mse": statistics.fmean(baseline_values), "candidate_vs_mamba1_pct": 100 * (statistics.fmean(candidate_values) / statistics.fmean(baseline_values) - 1), "candidate_wins": None, "n": len(candidate_values)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = candidate_rows + baseline
    for name, values in (("validation_rows.csv", rows), ("paired_summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ["# Relation-Auxiliary-Only MambaTSv3 vs Plain Mamba-1 H96", "", "Validation-only descriptive fresh-seed comparison. Candidate seeds are 3291--3293 and plain Mamba-1 seeds are 3351--3353; no paired wins are reported. Negative change favors the AutoCon candidate.", "", "| Dataset | Candidate MSE | Plain Mamba-1 MSE | Candidate vs Mamba-1 | Candidate n |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {candidate_mse:.6f} | {mamba1_mse:.6f} | {candidate_vs_mamba1_pct:+.3f}% | {n} |".format(**row) for row in summary)
    lines.extend(["", f"Both phases share source snapshot `{baseline_source}` and no test artifacts were produced. This isolates a plain Mamba-1 comparator; it is not a paired, held-out, or universal causal result."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
