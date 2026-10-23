#!/usr/bin/env python3
"""Strict validation-only readout for the AGCRN traffic baseline expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SEEDS = (3501, 3502, 3503)
PHASES = {"candidate_h12_validation": ("candidate", 6, "MambaTSv3AutoCon"), "agcrn_h12_validation": ("agcrn", 6, "AGCRN")}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle: return json.load(handle)


def read_phase(root: Path, phase: str, expected: int, model: str) -> tuple[str, list[dict]]:
    validation, provenance = load(root / "phase_validation.json"), load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected or not source or provenance.get("test_selection_allowed") is not False: raise ValueError(f"invalid phase: {phase}")
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", "")); result = Path(record["expected_artifact"]).parent
        if not job_id or job_id in seen or record.get("status") != "ok" or record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source: raise ValueError(f"invalid manifest: {job_id}")
        seen.add(job_id)
        if not (result / "training_complete.json").is_file() or any((result / item).exists() for item in TEST_ARTIFACTS): raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json")
        if config.get("model") != model or config.get("skip_test") is not True or int(config.get("pred_len", -1)) != 12 or int(config.get("seed", -1)) not in SEEDS: raise ValueError(f"coordinate mismatch: {job_id}")
        if model == "MambaTSv3AutoCon" and (config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not True or float(config.get("AutoCon_lambda", 0.0)) != 0.001): raise ValueError(f"candidate contract mismatch: {job_id}")
        if model == "AGCRN" and config.get("AutoCon") is not False: raise ValueError(f"AGCRN AutoCon contract mismatch: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses): raise ValueError(f"invalid validation loss: {job_id}")
        rows.append({"phase": phase, "dataset": str(config["data"]), "horizon": 12, "seed": int(config["seed"]), "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected or {row["seed"] for row in rows} != set(SEEDS) or {row["dataset"] for row in rows} != {"metr_la", "pems_bay"}: raise ValueError(f"coverage mismatch: {phase}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    rows, sources = [], set()
    for phase, (directory, expected, model) in PHASES.items():
        source, current = read_phase(args.artifact_root / directory, phase, expected, model); sources.add(source); rows.extend(current)
    if len(sources) != 1: raise ValueError("phases do not share a source snapshot")
    index = {(row["phase"], row["dataset"], row["seed"]): row for row in rows}
    if len(index) != len(rows): raise ValueError("duplicate validation coordinate")
    summary = []
    for dataset in ("metr_la", "pems_bay"):
        candidate = [index[("candidate_h12_validation", dataset, seed)] for seed in SEEDS]; agcrn = [index[("agcrn_h12_validation", dataset, seed)] for seed in SEEDS]; ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, agcrn)]
        summary.append({"dataset": dataset, "horizon": 12, "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate), "agcrn_mse_mean": statistics.fmean(row["best_val_mse"] for row in agcrn), "candidate_vs_agcrn_pct": 100 * (statistics.fmean(ratios) - 1), "candidate_wins": sum(value < 1 for value in ratios), "n": len(ratios), "candidate_elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in candidate), "agcrn_elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in agcrn)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# METR-LA / PEMS-BAY AGCRN Baseline Expansion", "", "Validation-only fresh-seed baseline expansion. No held-out test metrics or artifacts are included.", "", "| Dataset | H | Candidate MSE | AGCRN MSE | Candidate vs AGCRN | Wins | Candidate job sec | AGCRN job sec |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {horizon} | {candidate_mse_mean:.6f} | {agcrn_mse_mean:.6f} | {candidate_vs_agcrn_pct:+.3f}% | {candidate_wins}/{n} | {candidate_elapsed_sec_mean:.1f} | {agcrn_elapsed_sec_mean:.1f} |".format(**row) for row in summary)
    lines.extend(["", "Negative candidate-vs-AGCRN change favors the frozen MambaTSv3AutoCon candidate. Elapsed seconds are full validation job times and are not an inference-latency claim.", "", "AGCRN is a local canonical-style implementation, not a bitwise external reproduction. This comparison closes a baseline gap but does not authorize held-out testing."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
