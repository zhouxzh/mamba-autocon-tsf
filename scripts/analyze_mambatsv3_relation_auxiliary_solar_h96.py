#!/usr/bin/env python3
"""Strict reader for the fresh Solar relation-auxiliary-only H96 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


CANDIDATE = "MambaTSv3AutoCon"
BASELINES = ("DLinear", "NLinear", "RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")
SEEDS = (3341, 3342, 3343)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, phase: str, expected: int, models: set[str], candidate: bool) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError(f"invalid provenance: {root}")
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
        if any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json")
        if str(config.get("data")) != "solar" or str(config.get("model")) not in models or int(config.get("seed", -1)) not in SEEDS or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True:
            raise ValueError(f"coordinate mismatch: {job_id}")
        if candidate:
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False or abs(float(config.get("AutoCon_lambda", 0.0)) - .001) > 1e-12:
                raise ValueError(f"candidate contract mismatch: {job_id}")
        elif config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError(f"baseline isolation mismatch: {job_id}")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        values = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid validation losses: {job_id}")
        rows.append({"dataset": "solar", "horizon": 96, "seed": int(config["seed"]), "model": str(config["model"]), "best_val_mse": min(values), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected:
        raise ValueError(f"coverage mismatch: {root}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_a, candidate = read_phase(args.artifact_root / "candidate", "candidate_solar_h96_validation", 3, {CANDIDATE}, True)
    source_b, baseline = read_phase(args.artifact_root / "baselines", "baselines_solar_h96_validation", 21, set(BASELINES), False)
    if source_a != source_b:
        raise ValueError("candidate/baseline snapshots differ")
    rows = candidate + baseline
    index = {(row["model"], row["seed"]): row for row in rows}
    expected = {(model, seed) for model in (CANDIDATE,) + BASELINES for seed in SEEDS}
    if set(index) != expected:
        raise ValueError("incomplete Solar matrix")
    summary = []
    for baseline_name in BASELINES:
        candidate_rows = [index[(CANDIDATE, seed)] for seed in SEEDS]
        baseline_rows = [index[(baseline_name, seed)] for seed in SEEDS]
        ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate_rows, baseline_rows)]
        summary.append({"dataset": "solar", "horizon": 96, "baseline": baseline_name, "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate_rows), "baseline_mse_mean": statistics.fmean(row["best_val_mse"] for row in baseline_rows), "candidate_vs_baseline_pct": 100 * (statistics.fmean(ratios) - 1), "candidate_wins": sum(value < 1 for value in ratios), "n": 3})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("validation_rows.csv", rows), ("paired_summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ["# Solar H96 Relation-Auxiliary-Only Matrix", "", "Validation-only, three fresh paired seeds. The reader rejects test artifacts before aggregation.", "", "| Baseline | Candidate MSE | Baseline MSE | Candidate vs baseline | Wins |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {candidate_vs_baseline_pct:+.3f}% | {candidate_wins}/{n} |".format(**row) for row in summary)
    lines.extend(["", "Negative change favors the candidate. This is validation evidence only; it does not establish held-out performance, universal speed, or an independent AutoCon causal effect."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": len(rows), "paired_cells": len(summary), "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
