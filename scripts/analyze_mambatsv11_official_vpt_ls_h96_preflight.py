#!/usr/bin/env python3
"""Strict readout for the MambaTS v11 python-tsp VPT preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = {
    "v11_vpt_greedy_control_h96": ("v11_greedy", 6, "v11", "greedy"),
    "v11_vpt_local_search_h96": ("v11_local_search", 6, "v11", "ls"),
    "v3_fusion_reference_h96": ("v3_fusion_reference", 6, "v3", None),
    "external_baselines_h96": ("baselines", 24, "baseline", None),
}
SEEDS = (2431, 2432, 2433)
DATASETS = ("ETTh1", "weather")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, phase: str, expected: int, kind: str, solver: str | None) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {phase}")
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError(f"missing strict provenance: {phase}")
    rows, seen = [], set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid job: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if any((result / artifact).exists() for artifact in ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")):
            raise ValueError(f"test artifact present: {job_id}")
        config = load(result / "config.json")
        if config.get("skip_test") is not True or config.get("model") is None:
            raise ValueError(f"validation config mismatch: {job_id}")
        if kind == "v11":
            if config.get("model") != "MambaTSv11AutoCon" or config.get("mamba_ts_variant") != "v11":
                raise ValueError(f"v11 binding mismatch: {job_id}")
            if config.get("mamba_ts_vpt_solver") != solver or config.get("mamba_ts_repr_forecast_fusion") is not True:
                raise ValueError(f"v11 solver/fusion mismatch: {job_id}")
        elif kind == "v3":
            if config.get("model") != "MambaTSv3AutoCon" or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not True:
                raise ValueError(f"v3 reference mismatch: {job_id}")
        elif config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError(f"baseline inherited fusion: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        logs = load(checkpoint / "loss_logs.json")
        losses = [float(value) for value in logs.get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation loss: {job_id}")
        row = {"phase": phase, "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "model": str(config["model"]), "solver": config.get("mamba_ts_vpt_solver", "none"), "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])}
        if kind == "v11":
            coverage = [float(value) for value in logs.get("vpt_transition_coverage", [])]
            counts = [float(value) for value in logs.get("vpt_transition_count", [])]
            if not coverage or min(coverage) <= 0 or not counts or max(counts) <= 0:
                raise ValueError(f"inactive VPT telemetry: {job_id}")
            row.update({"vpt_transition_coverage_final": coverage[-1], "vpt_transition_count_final": counts[-1]})
        rows.append(row)
    if len(rows) != expected:
        raise ValueError(f"coverage mismatch: {phase}")
    return source, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows, sources = [], set()
    for phase, (directory, expected, kind, solver) in PHASES.items():
        source, current = read_phase(args.artifact_root / directory, phase, expected, kind, solver)
        sources.add(source); rows.extend(current)
    if len(sources) != 1:
        raise ValueError("phases do not share one source snapshot")
    index = {(row["phase"], row["dataset"], row["seed"], row["model"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate coordinate")
    summary = []
    for dataset in DATASETS:
        greedy = [index[("v11_vpt_greedy_control_h96", dataset, seed, "MambaTSv11AutoCon")] for seed in SEEDS]
        local = [index[("v11_vpt_local_search_h96", dataset, seed, "MambaTSv11AutoCon")] for seed in SEEDS]
        v3 = [index[("v3_fusion_reference_h96", dataset, seed, "MambaTSv3AutoCon")] for seed in SEEDS]
        baselines = sorted({row["model"] for row in rows if row["phase"] == "external_baselines_h96" and row["dataset"] == dataset})
        for label, left, right, right_label in (("v11_local_search", local, greedy, "greedy"), ("v11_local_search", local, v3, "v3_fusion_reference")):
            ratios = [a["best_val_mse"] / b["best_val_mse"] for a, b in zip(left, right)]
            summary.append({"dataset": dataset, "contrast": label + " vs " + right_label, "left_mse_mean": statistics.fmean(r["best_val_mse"] for r in left), "right_mse_mean": statistics.fmean(r["best_val_mse"] for r in right), "relative_change_pct": 100 * (statistics.fmean(ratios) - 1), "left_wins": sum(ratio < 1 for ratio in ratios), "n": len(ratios)})
        for baseline in baselines:
            refs = [index[("external_baselines_h96", dataset, seed, baseline)] for seed in SEEDS]
            left = local
            ratios = [a["best_val_mse"] / b["best_val_mse"] for a, b in zip(left, refs)]
            summary.append({"dataset": dataset, "contrast": "v11_local_search vs " + baseline, "left_mse_mean": statistics.fmean(r["best_val_mse"] for r in left), "right_mse_mean": statistics.fmean(r["best_val_mse"] for r in refs), "relative_change_pct": 100 * (statistics.fmean(ratios) - 1), "left_wins": sum(ratio < 1 for ratio in ratios), "n": len(ratios)})
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "validation_rows.csv", rows); write_csv(out / "paired_summary.csv", summary)
    lines = ["# MambaTS v11 Official-style VPT Local Search H96 Preflight", "", "Validation-only fresh-seed readout. v11 transition coverage is required to be nonzero for every job.", "", "| Dataset | Contrast | Left MSE | Right MSE | Change | Wins |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {contrast} | {left_mse_mean:.6f} | {right_mse_mean:.6f} | {relative_change_pct:+.3f}% | {left_wins}/{n} |".format(**row))
    lines.extend(["", "The local-search route is a python-tsp adaptation of the public MambaTS VPT solver family; it is not a bitwise official reproduction."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))


if __name__ == "__main__":
    main()
