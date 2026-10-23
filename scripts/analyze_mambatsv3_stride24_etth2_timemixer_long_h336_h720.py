#!/usr/bin/env python3
"""Strict reader for the ETTh2 stride-24 candidate versus TimeMixer long matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASET, SEEDS, HORIZONS = "ETTh2", (3461, 3462, 3463), (336, 720)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")
PHASES = {"candidate": ("stride24_candidate_timemixer_long_validation", 6, {"MambaTSv3AutoCon"}), "timemixer": ("timemixer_baseline_long_validation", 6, {"TimeMixer"})}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, label: str) -> tuple[str, list[dict]]:
    phase, expected, models = PHASES[label]; validation, provenance = load(root / "phase_validation.json"), load(root / "provenance.json"); source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected or not source or provenance.get("test_selection_allowed") is not False: raise ValueError(f"invalid phase/provenance: {label}")
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok": raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source: raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_ARTIFACTS): raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json"); model, seed, horizon = str(config.get("model")), int(config.get("seed", -1)), int(config.get("pred_len", -1))
        if config.get("data") != DATASET or model not in models or seed not in SEEDS or horizon not in HORIZONS or config.get("skip_test") is not True or int(config.get("patch_len", -1)) != 24 or int(config.get("patch_stride", -1)) != 24: raise ValueError(f"coordinate/patch mismatch: {job_id}")
        if model == "MambaTSv3AutoCon":
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False or float(config.get("AutoCon_lambda", 0.0)) != 0.001: raise ValueError(f"candidate contract mismatch: {job_id}")
        elif config.get("AutoCon") is not False or config.get("timemixer_down_sampling_window", 0) != 2 or config.get("timemixer_down_sampling_layers", 0) != 1 or config.get("timemixer_channel_independence") is not False: raise ValueError(f"TimeMixer contract mismatch: {job_id}")
        command = record["command"]; checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name; losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses): raise ValueError(f"invalid validation losses: {job_id}")
        rows.append({"dataset": DATASET, "horizon": horizon, "seed": seed, "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected: raise ValueError(f"coverage mismatch: {label}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--candidate-root", type=Path, required=True); parser.add_argument("--timemixer-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    source_a, candidate_rows = read_phase(args.candidate_root, "candidate"); source_b, timemixer_rows = read_phase(args.timemixer_root, "timemixer")
    if source_a != source_b: raise ValueError("phases do not share one immutable source")
    rows = candidate_rows + timemixer_rows; index = {(row["model"], row["horizon"], row["seed"]): row for row in rows}; expected = {(model, horizon, seed) for model in ("MambaTSv3AutoCon", "TimeMixer") for horizon in HORIZONS for seed in SEEDS}
    if set(index) != expected or len(index) != len(rows): raise ValueError("incomplete TimeMixer long matrix")
    summary = []
    for horizon in HORIZONS:
        candidate = [index[("MambaTSv3AutoCon", horizon, seed)] for seed in SEEDS]; baseline = [index[("TimeMixer", horizon, seed)] for seed in SEEDS]; ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, baseline)]
        summary.append({"dataset": DATASET, "horizon": horizon, "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate), "timemixer_mse_mean": statistics.fmean(row["best_val_mse"] for row in baseline), "candidate_vs_timemixer_pct": 100 * (statistics.fmean(ratios) - 1), "candidate_wins": sum(value < 1 for value in ratios), "n": len(ratios)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# Stride-24 MambaTSv3AutoCon versus TimeMixer on ETTh2 H336/H720", "", "Validation-only, three fresh paired seeds at each horizon, 12/12 successful jobs, one immutable source snapshot. Negative change favors the candidate.", "", "| Dataset | H | Candidate MSE | TimeMixer MSE | Candidate vs TimeMixer | Wins |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {horizon} | {candidate_mse_mean:.6f} | {timemixer_mse_mean:.6f} | {candidate_vs_timemixer_pct:+.3f}% | {candidate_wins}/{n} |".format(**row) for row in summary)
    lines.extend(["", "This matrix is validation-only. It does not establish held-out accuracy, universal speed, independent AutoCon causality, or universal generalization."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": source_a, "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
