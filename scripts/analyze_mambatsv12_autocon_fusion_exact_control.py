#!/usr/bin/env python3
"""Strict paired readout for the MambaTSv12 AutoCon fusion control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PAIRS = (
    ("ETTh1", 96, "fusion_off_h96", "fusion_on_h96"),
    ("weather", 96, "fusion_off_h96", "fusion_on_h96"),
    ("pems07", 12, "fusion_off_pems07_h12", "fusion_on_pems07_h12"),
)
SEEDS = (2461, 2462, 2463)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase_rows(root: Path, name: str, expected_fusion: bool, expected_jobs: int) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"invalid phase validator: {name}")
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError(f"missing strict provenance: {name}")
    rows, seen = [], set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"bad job: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if any((result / artifact).exists() for artifact in ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")):
            raise ValueError(f"test artifact present: {job_id}")
        config = load(result / "config.json")
        complete = load(result / "training_complete.json")
        if config.get("skip_test") is not True or complete.get("status") != "trained_without_test":
            raise ValueError(f"validation contract mismatch: {job_id}")
        if config.get("model") != "MambaTSv12AutoCon" or config.get("mamba_ts_variant") != "v12":
            raise ValueError(f"v12 config mismatch: {job_id}")
        actual_fusion = bool(config.get("mamba_ts_repr_forecast_fusion"))
        if actual_fusion is not expected_fusion:
            raise ValueError(f"fusion config mismatch: {job_id}")
        if config.get("mamba_ts_delta_multipliers") not in ([1.0, 2.0], (1.0, 2.0)):
            raise ValueError(f"Delta config mismatch: {job_id}")
        if float(config.get("AutoCon_lambda", -1)) != 0.001:
            raise ValueError(f"lambda config mismatch: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation telemetry: {job_id}")
        rows.append({"phase": name, "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected_jobs:
        raise ValueError(f"incomplete phase: {name}")
    return source, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    all_rows, sources, summary = [], set(), []
    for dataset, horizon, off_dir, on_dir in PAIRS:
        expected = 3 if dataset == "pems07" else 6
        off_source, off_rows = phase_rows(args.artifact_root / off_dir, off_dir, False, expected)
        on_source, on_rows = phase_rows(args.artifact_root / on_dir, on_dir, True, expected)
        sources.update((off_source, on_source)); all_rows.extend(off_rows + on_rows)
        off = {row["seed"]: row for row in off_rows if row["dataset"] == dataset}; on = {row["seed"]: row for row in on_rows if row["dataset"] == dataset}
        if set(off) != set(SEEDS) or set(on) != set(SEEDS):
            raise ValueError(f"seed mismatch: {dataset}")
        ratios = [on[seed]["best_val_mse"] / off[seed]["best_val_mse"] for seed in SEEDS]
        summary.append({"dataset": dataset, "horizon": horizon, "fusion_off_mse_mean": statistics.fmean(row["best_val_mse"] for row in off.values()), "fusion_on_mse_mean": statistics.fmean(row["best_val_mse"] for row in on.values()), "fusion_on_vs_off_pct": 100 * (statistics.fmean(ratios) - 1), "fusion_on_wins": sum(ratio < 1 for ratio in ratios), "n": len(ratios), "max_fusion_on_degradation_pct": 100 * (max(ratios) - 1), "fusion_gate_pass": statistics.fmean(ratios) < 1 and sum(ratio < 1 for ratio in ratios) >= 2 and max(ratios) <= 1.02})
    if len(sources) != 1:
        raise ValueError("source snapshots differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_rows.csv", all_rows); write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = ["# Exact MambaTSv12 AutoCon Fusion Control", "", "Validation-only fresh three-seed one-factor control. The sole model difference is prediction-connected representation fusion.", "", "| Dataset | H | Fusion off | Fusion on | On vs off | Wins | Max degradation | Gate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        lines.append("| {dataset} | {horizon} | {fusion_off_mse_mean:.6f} | {fusion_on_mse_mean:.6f} | {fusion_on_vs_off_pct:+.3f}% | {fusion_on_wins}/{n} | {max_fusion_on_degradation_pct:+.3f}% | {fusion_gate_pass} |".format(**row))
    lines.extend(["", "A passing cell supports local fusion attribution only; it does not establish universal forecasting or speed superiority."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(all_rows), "summary": summary, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))


if __name__ == "__main__":
    main()
