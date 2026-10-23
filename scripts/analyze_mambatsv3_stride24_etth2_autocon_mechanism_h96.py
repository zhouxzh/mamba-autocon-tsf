#!/usr/bin/env python3
"""Strict reader for the ETTh2 stride-24 AutoCon active/null control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SEEDS = (3451, 3452, 3453)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")
PHASES = {"active": ("autocon_active_h96", "active", 3, 0.001), "null": ("autocon_lambda_zero_h96", "lambda_zero", 3, 0.0)}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def series(logs: dict, key: str, job_id: str) -> list[float]:
    values = logs.get(key)
    if not isinstance(values, list) or not values or not all(math.isfinite(float(value)) for value in values): raise ValueError(f"invalid {key}: {job_id}")
    return [float(value) for value in values]


def read_phase(root: Path, label: str) -> tuple[str, list[dict]]:
    phase, directory, expected, expected_lambda = PHASES[label]
    validation, provenance = load(root / directory / "phase_validation.json"), load(root / directory / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected or not source or provenance.get("test_selection_allowed") is not False: raise ValueError(f"invalid phase: {label}")
    rows, seen = [], set()
    for raw in (root / directory / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok": raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source: raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_ARTIFACTS): raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json")
        if config.get("data") != "ETTh2" or config.get("model") != "MambaTSv3AutoCon" or int(config.get("seed", -1)) not in SEEDS or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True or config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False or int(config.get("patch_len", -1)) != 24 or int(config.get("patch_stride", -1)) != 24 or float(config.get("AutoCon_lambda", -1)) != expected_lambda: raise ValueError(f"model control mismatch: {job_id}")
        command = record["command"]; checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name; logs = load(checkpoint / "loss_logs.json")
        auto, mse, total, effective = (series(logs, key, job_id) for key in ("autocon_loss", "mse_loss", "train_loss", "effective_lambda"))
        if not len(auto) == len(mse) == len(total) == len(effective) or any(value != expected_lambda for value in effective) or not any(value > 0 for value in auto): raise ValueError(f"loss contract mismatch: {job_id}")
        residual = max(abs(total_i - (mse_i + expected_lambda * auto_i)) for total_i, mse_i, auto_i in zip(total, mse, auto))
        if residual > 1e-6: raise ValueError(f"loss identity mismatch: {job_id}")
        val_loss = series(logs, "val_loss", job_id)
        rows.append({"phase": label, "dataset": "ETTh2", "horizon": 96, "seed": int(config["seed"]), "best_val_mse": min(val_loss), "autocon_loss_first": auto[0], "autocon_loss_last": auto[-1], "max_identity_residual": residual, "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected: raise ValueError(f"coverage mismatch: {label}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    sources, rows = [], []
    for label in PHASES:
        source, current = read_phase(args.artifact_root, label); sources.append(source); rows.extend(current)
    if len(set(sources)) != 1: raise ValueError("phases do not share one source snapshot")
    index = {(row["phase"], row["seed"]): row for row in rows}
    if set(index) != {(phase, seed) for phase in PHASES for seed in SEEDS}: raise ValueError("incomplete paired matrix")
    active = [index[("active", seed)] for seed in SEEDS]; null = [index[("null", seed)] for seed in SEEDS]; ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(active, null)]
    summary = {"dataset": "ETTh2", "horizon": 96, "active_mse_mean": statistics.fmean(row["best_val_mse"] for row in active), "null_mse_mean": statistics.fmean(row["best_val_mse"] for row in null), "active_vs_null_pct": 100 * (statistics.fmean(ratios) - 1), "active_wins": sum(value < 1 for value in ratios), "n": 3, "max_active_degradation_pct": 100 * (max(ratios) - 1), "local_gate_pass": statistics.fmean(ratios) < 1 and sum(value < 1 for value in ratios) >= 2 and max(ratios) <= 1.02}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    lines = ["# ETTh2 H96 Stride-24 AutoCon Active versus Lambda-Zero", "", "Validation-only exact paired control, three fresh seeds. Negative active-vs-null change favors the AutoCon objective.", "", "| Dataset | Active MSE | Lambda-zero MSE | Active vs null | Wins | Max degradation | Local gate |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |", "| {dataset} | {active_mse_mean:.6f} | {null_mse_mean:.6f} | {active_vs_null_pct:+.3f}% | {active_wins}/{n} | {max_active_degradation_pct:+.3f}% | {local_gate_pass} |".format(**summary), "", "The active arm has finite nonzero AutoCon loss with effective lambda=.001; the null arm still logs the raw loss but has effective lambda=0, so its contrastive objective has no weight in train_loss. This is a local mechanism result, not held-out evidence or proof of universal causality."]
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": len(rows), "paired_cells": 1, "source_tree_sha256": sources[0], "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": 1, "test_metrics_produced": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
