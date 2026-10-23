#!/usr/bin/env python3
"""Strictly read the relation-auxiliary-only candidate against existing H96 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3291, 3292, 3293)
BASELINES = ("DLinear", "NLinear", "RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_phase(root: Path, phase: str, expected_count: int, allowed_models: set[str], *, candidate: bool) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected_count:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError(f"invalid provenance: {root}")
    rows, seen = [], set()
    manifest = root / "logs" / "job_manifest.jsonl"
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json")
        dataset, model, seed = str(config.get("data")), str(config.get("model")), int(config.get("seed", -1))
        if dataset not in DATASETS or model not in allowed_models or seed not in SEEDS:
            raise ValueError(f"coordinate/model mismatch: {job_id}")
        if int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True:
            raise ValueError(f"validation mismatch: {job_id}")
        if candidate:
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False or float(config.get("AutoCon_lambda", 0.0)) != 0.001:
                raise ValueError(f"candidate contract mismatch: {job_id}")
        elif config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError(f"baseline isolation mismatch: {job_id}")
        command = record.get("command", [])
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid loss series: {job_id}")
        rows.append({"dataset": dataset, "horizon": 96, "seed": seed, "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected_count:
        raise ValueError(f"coverage mismatch: {root}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--dlinear-root", type=Path, required=True)
    parser.add_argument("--remaining-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources, rows = set(), []
    source, candidate = read_phase(args.candidate_root, "relation_auxiliary_only_candidate_h96_validation", 6, {"MambaTSv3AutoCon"}, candidate=True)
    sources.add(source)
    rows.extend(candidate)
    source, linear = read_phase(args.dlinear_root, "linear_baselines_h96_validation", 12, {"DLinear", "NLinear"}, candidate=False)
    sources.add(source)
    rows.extend(linear)
    source, remaining = read_phase(args.remaining_root, "remaining_baselines_h96_validation", 30, set(BASELINES[2:]), candidate=False)
    sources.add(source)
    rows.extend(remaining)
    if len(sources) != 1:
        raise ValueError("source snapshots differ")
    index = {(row["dataset"], row["model"], row["seed"]): row for row in rows}
    expected = {(dataset, model, seed) for dataset in DATASETS for model in ("MambaTSv3AutoCon",) + BASELINES for seed in SEEDS}
    if set(index) != expected:
        raise ValueError("incomplete auxiliary-only matrix")
    summary = []
    for dataset in DATASETS:
        candidate_rows = [index[(dataset, "MambaTSv3AutoCon", seed)] for seed in SEEDS]
        for baseline in BASELINES:
            baseline_rows = [index[(dataset, baseline, seed)] for seed in SEEDS]
            ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate_rows, baseline_rows)]
            summary.append({"dataset": dataset, "horizon": 96, "baseline": baseline, "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate_rows), "baseline_mse_mean": statistics.fmean(row["best_val_mse"] for row in baseline_rows), "candidate_vs_baseline_pct": 100 * (statistics.fmean(ratios) - 1.0), "candidate_wins": sum(value < 1.0 for value in ratios), "n": 3})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_rows.csv", rows)
    write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = ["# Electricity / Traffic H96 Relation-Auxiliary-Only Matrix", "", "Validation-only confirmation: MambaTSv3AutoCon uses AutoCon relation training with prediction-connected fusion disabled. Existing baselines are read-only same-source artifacts; test artifacts are rejected.", "", "| Dataset | H | Baseline | Candidate MSE | Baseline MSE | Candidate vs baseline | Wins |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {horizon} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {candidate_vs_baseline_pct:+.3f}% | {candidate_wins}/{n} |".format(**row) for row in summary)
    lines.extend(["", "Negative change favors the relation-auxiliary-only candidate. This is validation evidence only; it is not a held-out result and does not establish universal speed or causal relation-loss benefit."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
