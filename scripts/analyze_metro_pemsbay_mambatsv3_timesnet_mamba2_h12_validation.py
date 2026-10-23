#!/usr/bin/env python3
"""Strict validation-only reader for the METR-LA/PEMS-BAY baseline expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = {
    "candidate_h12_validation": ("candidate", 6, "candidate"),
    "timesnet_mamba2_baselines_h12_validation": ("baselines", 12, "baseline"),
}
SEEDS = (2971, 2972, 2973)
BASELINES = ("TimesNet", "Mamba2")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, phase: str, expected: int, kind: str) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {phase}")
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError(f"missing strict provenance: {phase}")
    rows, seen = [], set()
    manifest = root / "logs" / "job_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid job: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        prohibited = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")
        if any((result / artifact).exists() for artifact in prohibited):
            raise ValueError(f"test artifact present: {job_id}")
        config = load(result / "config.json")
        if config.get("skip_test") is not True or int(config.get("pred_len", -1)) != 12:
            raise ValueError(f"validation config mismatch: {job_id}")
        model = str(config.get("model"))
        if kind == "candidate":
            if model != "MambaTSv3AutoCon" or config.get("mamba_ts_variant") != "v3":
                raise ValueError(f"candidate binding mismatch: {job_id}")
            if config.get("mamba_ts_repr_forecast_fusion") is not True or float(config.get("AutoCon_lambda", 0)) != 0.001:
                raise ValueError(f"candidate AutoCon binding mismatch: {job_id}")
        else:
            if model not in BASELINES or config.get("mamba_ts_repr_forecast_fusion") is True:
                raise ValueError(f"baseline binding mismatch: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation loss: {job_id}")
        rows.append({
            "phase": phase,
            "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]),
            "model": model,
            "best_val_mse": min(losses),
            "elapsed_sec": float(record["elapsed_sec"]),
        })
    if len(rows) != expected:
        raise ValueError(f"coverage mismatch: {phase}")
    return source, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, sources = [], set()
    for phase, (directory, expected, kind) in PHASES.items():
        source, current = read_phase(args.artifact_root / directory, phase, expected, kind)
        sources.add(source)
        rows.extend(current)
    if len(sources) != 1:
        raise ValueError("phases do not share one source snapshot")
    index = {(r["phase"], r["dataset"], r["seed"], r["model"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate coordinate")
    summary = []
    for dataset in sorted({r["dataset"] for r in rows}):
        candidate = [index[("candidate_h12_validation", dataset, seed, "MambaTSv3AutoCon")] for seed in SEEDS]
        for model in BASELINES:
            comparator = [index[("timesnet_mamba2_baselines_h12_validation", dataset, seed, model)] for seed in SEEDS]
            ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, comparator)]
            summary.append({
                "dataset": dataset,
                "horizon": 12,
                "contrast": "MambaTSv3AutoCon vs " + model,
                "left_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate),
                "right_mse_mean": statistics.fmean(row["best_val_mse"] for row in comparator),
                "relative_change_pct": 100 * (statistics.fmean(ratios) - 1),
                "left_wins": sum(ratio < 1 for ratio in ratios),
                "n": len(ratios),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_rows.csv", rows)
    write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = ["# METR-LA / PEMS-BAY TimesNet and Mamba2 Expansion", "", "Validation-only fresh-seed baseline expansion. This readout contains no held-out test metrics.", "", "| Dataset | H | Contrast | Candidate MSE | Comparator MSE | Change | Wins |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {horizon} | {contrast} | {left_mse_mean:.6f} | {right_mse_mean:.6f} | {relative_change_pct:+.3f}% | {left_wins}/{n} |".format(**row))
    lines.extend(["", "The candidate configuration was frozen before this expansion. Any favorable validation result remains descriptive until a separately registered held-out comparison."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
