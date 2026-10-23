#!/usr/bin/env python3
"""Strict validation-only readout for the MambaTS v12 Delta screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = {
    "v12_candidate_h96_validation": ("v12_candidate_h96", 6, "v12"),
    "v12_candidate_pems07_h12_validation": ("v12_candidate_pems07", 3, "v12"),
    "v3_reference_h96_h12_validation": ("v3_reference", 6, "v3"),
    "v3_reference_pems07_h12_validation": ("v3_reference_pems07", 3, "v3"),
    "strong_baselines_h96_h12_validation": ("strong_baselines", 24, "baseline"),
    "strong_baselines_pems07_h12_validation": ("strong_baselines_pems07", 12, "baseline"),
}
SEEDS = (2451, 2452, 2453)


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
        model = str(config["model"])
        if kind == "v12":
            if model != "MambaTSv12AutoCon" or config.get("mamba_ts_variant") != "v12":
                raise ValueError(f"v12 binding mismatch: {job_id}")
            if config.get("mamba_ts_delta_multipliers") not in ([1.0, 2.0], (1.0, 2.0)):
                raise ValueError(f"v12 Delta binding mismatch: {job_id}")
            if config.get("mamba_ts_repr_forecast_fusion") is not True:
                raise ValueError(f"v12 AutoCon fusion mismatch: {job_id}")
        elif kind == "v3":
            if model != "MambaTSv3AutoCon" or config.get("mamba_ts_variant") != "v3":
                raise ValueError(f"v3 binding mismatch: {job_id}")
        elif config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError(f"baseline inherited fusion: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        logs = load(checkpoint / "loss_logs.json")
        losses = [float(value) for value in logs.get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation loss: {job_id}")
        rows.append({
            "phase": phase, "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "model": model, "best_val_mse": min(losses),
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
    candidate_phases = ["v12_candidate_h96_validation", "v12_candidate_pems07_h12_validation"]
    reference_phases = ["v3_reference_h96_h12_validation", "v3_reference_pems07_h12_validation"]
    baseline_phases = ["strong_baselines_h96_h12_validation", "strong_baselines_pems07_h12_validation"]
    for candidate_phase, reference_phase, baseline_phase in zip(candidate_phases, reference_phases, baseline_phases):
        datasets = sorted({r["dataset"] for r in rows if r["phase"] == candidate_phase})
        for dataset in datasets:
            candidate = [index[(candidate_phase, dataset, seed, "MambaTSv12AutoCon")] for seed in SEEDS]
            reference = [index[(reference_phase, dataset, seed, "MambaTSv3AutoCon")] for seed in SEEDS]
            ratios = [a["best_val_mse"] / b["best_val_mse"] for a, b in zip(candidate, reference)]
            summary.append({"dataset": dataset, "horizon": candidate[0]["horizon"], "contrast": "v12 vs v3", "left_mse_mean": statistics.fmean(r["best_val_mse"] for r in candidate), "right_mse_mean": statistics.fmean(r["best_val_mse"] for r in reference), "relative_change_pct": 100 * (statistics.fmean(ratios) - 1), "left_wins": sum(r < 1 for r in ratios), "n": len(ratios)})
            baseline_names = sorted({r["model"] for r in rows if r["phase"] == baseline_phase and r["dataset"] == dataset})
            for model in baseline_names:
                baseline = [index[(baseline_phase, dataset, seed, model)] for seed in SEEDS]
                ratios = [a["best_val_mse"] / b["best_val_mse"] for a, b in zip(candidate, baseline)]
                summary.append({"dataset": dataset, "horizon": candidate[0]["horizon"], "contrast": "v12 vs " + model, "left_mse_mean": statistics.fmean(r["best_val_mse"] for r in candidate), "right_mse_mean": statistics.fmean(r["best_val_mse"] for r in baseline), "relative_change_pct": 100 * (statistics.fmean(ratios) - 1), "left_wins": sum(r < 1 for r in ratios), "n": len(ratios)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_rows.csv", rows)
    write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = ["# MambaTS v12 Two-Delta AutoCon Validation", "", "Validation-only fresh-seed readout; all test artifacts are prohibited.", "", "| Dataset | H | Contrast | Candidate MSE | Comparator MSE | Change | Wins |", "| --- | ---: | --- | ---: | ---: | ---: | ---:|"]
    for row in summary:
        lines.append("| {dataset} | {horizon} | {contrast} | {left_mse_mean:.6f} | {right_mse_mean:.6f} | {relative_change_pct:+.3f}% | {left_wins}/{n} |".format(**row))
    lines.extend(["", "v12 is an independent implementation of the high-level multi-Delta idea; it is not a bitwise reproduction of external code.", "The separate resource benchmark found v12 slower than the one-scale v3 under the PEMS07 high-dimensional shape, so no speed claim is made."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))


if __name__ == "__main__":
    main()
