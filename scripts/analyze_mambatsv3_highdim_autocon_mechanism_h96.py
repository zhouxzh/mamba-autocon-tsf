#!/usr/bin/env python3
"""Strict reader for the high-dimensional MambaTSv3 AutoCon mechanism matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3301, 3302, 3303)
PHASES = {"active_fusion_h96": ("active", 6, True, 0.001), "fusion_off_h96": ("fusion_off", 6, False, 0.001), "lambda_zero_h96": ("lambda_zero", 6, True, 0.0)}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _phase(root: Path, name: str, directory: str, expected: int, fusion: bool, lam: float) -> tuple[str, list[dict]]:
    validation, provenance = _load(root / directory / "phase_validation.json"), _load(root / directory / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != name or validation.get("expected_jobs") != expected or not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError("invalid phase contract: {}".format(name))
    rows, seen = [], set()
    for line in (root / directory / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok": raise ValueError("invalid job: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source: raise ValueError("source drift: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / item).exists() for item in TEST_ARTIFACTS): raise ValueError("test boundary violation: {}".format(job_id))
        config = _load(result / "config.json")
        if (config.get("model") != "MambaTSv3AutoCon" or config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not fusion or float(config.get("AutoCon_lambda", -1)) != lam or config.get("skip_test") is not True): raise ValueError("model control mismatch: {}".format(job_id))
        dataset, seed = str(config.get("data")), int(config.get("seed", -1))
        if dataset not in DATASETS or seed not in SEEDS or int(config.get("pred_len", -1)) != 96: raise ValueError("coordinate mismatch: {}".format(job_id))
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in _load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses): raise ValueError("invalid loss: {}".format(job_id))
        rows.append({"phase": name, "dataset": dataset, "horizon": 96, "seed": seed, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected: raise ValueError("coverage mismatch: {}".format(name))
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    rows, sources = [], set()
    for name, (directory, expected, fusion, lam) in PHASES.items():
        source, current = _phase(args.artifact_root, name, directory, expected, fusion, lam); sources.add(source); rows.extend(current)
    if len(sources) != 1: raise ValueError("phases do not share one source snapshot")
    index = {(row["phase"], row["dataset"], row["seed"]): row for row in rows}
    if len(index) != len(rows): raise ValueError("duplicate coordinate")
    summary = []
    for dataset in DATASETS:
        active = [index[("active_fusion_h96", dataset, seed)] for seed in SEEDS]
        for label, control_phase in (("active_vs_fusion_off", "fusion_off_h96"), ("active_vs_lambda_zero", "lambda_zero_h96")):
            control = [index[(control_phase, dataset, seed)] for seed in SEEDS]
            ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(active, control)]
            summary.append({"dataset": dataset, "horizon": 96, "contrast": label, "active_mse_mean": statistics.fmean(row["best_val_mse"] for row in active), "control_mse_mean": statistics.fmean(row["best_val_mse"] for row in control), "active_vs_control_pct": 100 * (statistics.fmean(ratios) - 1), "active_wins": sum(value < 1 for value in ratios), "n": len(ratios), "max_active_degradation_pct": 100 * (max(ratios) - 1), "local_gate_pass": statistics.fmean(ratios) < 1 and sum(value < 1 for value in ratios) >= 2 and max(ratios) <= 1.02})
    args.output_dir.mkdir(parents=True, exist_ok=True); _write(args.output_dir / "validation_rows.csv", rows); _write(args.output_dir / "paired_summary.csv", summary)
    lines = ["# Electricity / Traffic H96 MambaTSv3 AutoCon Mechanism Matrix", "", "Validation-only exact paired mechanism controls. No held-out metric, prediction, or streaming artifact was produced.", "", "| Dataset | H | Contrast | Active MSE | Control MSE | Active vs control | Wins | Max degradation | Local gate |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary: lines.append("| {dataset} | {horizon} | {contrast} | {active_mse_mean:.6f} | {control_mse_mean:.6f} | {active_vs_control_pct:+.3f}% | {active_wins}/{n} | {max_active_degradation_pct:+.3f}% | {local_gate_pass} |".format(**row))
    lines.extend(["", "The fusion contrast isolates whether the AutoCon-trained representation connects to prediction. The lambda-zero contrast isolates the contrastive objective within the same fusion graph. Failed contrasts remain negative evidence and do not license retuning."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
