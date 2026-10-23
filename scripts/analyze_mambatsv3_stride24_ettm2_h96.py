#!/usr/bin/env python3
"""Strict reader for the frozen stride-24 ETTm2 H96 validation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASET = "ETTm2"
SEEDS = (3401, 3402, 3403)
BASELINES = ("DLinear", "RLinear", "PatchTST", "iTransformer", "Mamba1", "TimesNet", "Mamba2")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")
PHASES = {
    "candidate": ("stride24_candidate_h96_validation", 3, {"MambaTSv3AutoCon"}),
    "baseline": ("stride24_baselines_h96_validation", 21, set(BASELINES)),
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, label: str) -> tuple[str, list[dict]]:
    phase, expected_jobs, models = PHASES[label]
    validation = load(root / "phase_validation.json")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected_jobs or not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError(f"invalid phase/provenance: {label}")
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
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"test-boundary violation: {job_id}")
        config = load(result / "config.json")
        dataset, model, seed = str(config.get("data")), str(config.get("model")), int(config.get("seed", -1))
        if dataset != DATASET or model not in models or seed not in SEEDS or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True:
            raise ValueError(f"coordinate mismatch: {job_id}")
        if int(config.get("patch_len", -1)) != 24 or int(config.get("patch_stride", -1)) != 24:
            raise ValueError(f"stride mismatch: {job_id}")
        if model == "MambaTSv3AutoCon":
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False or float(config.get("AutoCon_lambda", 0.0)) != 0.001:
                raise ValueError(f"candidate contract mismatch: {job_id}")
        elif config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError(f"baseline isolation mismatch: {job_id}")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation losses: {job_id}")
        rows.append({"dataset": dataset, "horizon": 96, "seed": seed, "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected_jobs:
        raise ValueError(f"coverage mismatch: {label}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate_source, candidate_rows = read_phase(args.candidate_root, "candidate")
    baseline_source, baseline_rows = read_phase(args.baseline_root, "baseline")
    if candidate_source != baseline_source:
        raise ValueError("phases do not share one immutable source")
    rows = candidate_rows + baseline_rows
    index = {(row["model"], row["seed"]): row for row in rows}
    expected = {(model, seed) for model in ("MambaTSv3AutoCon",) + BASELINES for seed in SEEDS}
    if set(index) != expected or len(index) != len(rows):
        raise ValueError("incomplete ETTm2 matrix")
    candidate = [index[("MambaTSv3AutoCon", seed)] for seed in SEEDS]
    summary = []
    for baseline in BASELINES:
        comparator = [index[(baseline, seed)] for seed in SEEDS]
        ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, comparator)]
        summary.append({"dataset": DATASET, "horizon": 96, "baseline": baseline, "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate), "baseline_mse_mean": statistics.fmean(row["best_val_mse"] for row in comparator), "candidate_vs_baseline_pct": 100 * (statistics.fmean(ratios) - 1), "candidate_wins": sum(value < 1 for value in ratios), "n": len(ratios)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# Stride-24 Relation-Auxiliary MambaTSv3 ETTm2 H96 Matrix", "", "Validation-only, three fresh paired seeds, 24/24 successful jobs, one immutable source snapshot. Negative change favors the candidate.", "", "| Dataset | H | Baseline | Candidate MSE | Baseline MSE | Candidate vs baseline | Wins |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {horizon} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {candidate_vs_baseline_pct:+.3f}% | {candidate_wins}/{n} |".format(**row) for row in summary)
    lines.extend(["", "This matrix is validation-only. It does not establish held-out accuracy, universal speed, independent AutoCon causality, or universal generalization."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": candidate_source, "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
