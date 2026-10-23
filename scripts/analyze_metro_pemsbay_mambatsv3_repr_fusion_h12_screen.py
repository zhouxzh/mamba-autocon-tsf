#!/usr/bin/env python3
"""Strict paired readout for the METR-LA/PEMS-BAY MambaTSv3 fusion screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_training_plan import _job_specs, _load_json, _validate_plan

DATASETS = ("metr_la", "pems_bay")
SEEDS = (2811, 2812, 2813)
PHASES = ("fusion_off_h12", "fusion_on_h12")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return _load_json(path)


def _read(root: Path, plan: dict, phase_name: str) -> tuple[list[dict], str]:
    phase = next(p for p in plan["phases"] if p["name"] == phase_name)
    phase_root = root / phase_name
    validation = _json(phase_root / "phase_validation.json")
    expected = _job_specs(plan, phase, {}, allow_unresolved=True)
    if validation.get("valid") is not True or validation.get("expected_jobs") != len(expected):
        raise ValueError(f"invalid phase validation: {phase_name}")
    expected_map = {(j["dataset"], j["seed"]): j for j in expected}
    records = [json.loads(line) for line in (phase_root / "logs/job_manifest.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != len(expected_map):
        raise ValueError(f"manifest coverage mismatch: {phase_name}")
    source = str(_json(phase_root / "provenance.json").get("source_tree_sha256")); seen = set(); rows = []
    for record in records:
        key = (str(record["dataset"]), int(record["seed"]))
        if key not in expected_map or key in seen or record.get("status") != "ok":
            raise ValueError(f"invalid coordinate: {key}")
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {record.get('job_id')}")
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file():
            raise ValueError(f"missing completion: {result}")
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")):
            raise ValueError(f"test artifact present: {result}")
        config = _json(result / "config.json")
        job = expected_map[key]
        if config.get("plan_job_id") != record.get("job_id") or config.get("model") != "MambaTSv3AutoCon" or config.get("data") != key[0] or int(config.get("pred_len")) != 12 or int(config.get("seed")) != key[1] or config.get("AutoCon") is not True:
            raise ValueError(f"config mismatch: {result}")
        expected_fusion = phase_name == "fusion_on_h12"
        if config.get("mamba_ts_repr_forecast_fusion") is not expected_fusion or config.get("mamba_use_acf_memory") is not False:
            raise ValueError(f"fusion contract mismatch: {result}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        logs = _json(checkpoint / "loss_logs.json")
        losses = [float(v) for v in logs.get("val_loss", [])]
        if not losses or not all(math.isfinite(v) for v in losses):
            raise ValueError(f"invalid losses: {checkpoint}")
        if not logs.get("autocon_loss"):
            raise ValueError(f"missing AutoCon telemetry: {checkpoint}")
        rows.append({"phase": phase_name, "dataset": key[0], "seed": key[1], "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]), "autocon_loss_mean": statistics.fmean(float(v) for v in logs["autocon_loss"])})
        seen.add(key)
    if seen != set(expected_map):
        raise ValueError(f"incomplete phase: {phase_name}")
    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); plan = _json(args.plan); errors = _validate_plan(plan)
    if errors: raise ValueError("invalid plan: " + "; ".join(errors))
    rows = []; sources = set()
    for phase in PHASES:
        current, source = _read(args.root, plan, phase); rows.extend(current); sources.add(source)
    if len(sources) != 1: raise ValueError("source hashes disagree")
    index = {(r["phase"], r["dataset"], r["seed"]): r for r in rows}; summaries = []
    for dataset in DATASETS:
        off = [index[(PHASES[0], dataset, s)]["best_val_mse"] for s in SEEDS]; on = [index[(PHASES[1], dataset, s)]["best_val_mse"] for s in SEEDS]
        summaries.append({"dataset": dataset, "fusion_off_mean_val_mse": statistics.fmean(off), "fusion_on_mean_val_mse": statistics.fmean(on), "fusion_vs_off_percent": 100 * (statistics.fmean(on) / statistics.fmean(off) - 1), "wins": sum(a < b for a, b in zip(on, off)), "max_paired_degradation_percent": max((a / b - 1) * 100 for a, b in zip(on, off)), "passes_gate": statistics.fmean(on) < statistics.fmean(off) and sum(a < b for a, b in zip(on, off)) >= 2 and max((a / b - 1) * 100 for a, b in zip(on, off)) <= 2})
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_rows.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "summary.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=sorted(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    payload = {"plan_sha256": _sha(args.plan), "source_tree_sha256": next(iter(sources)), "rows": len(rows), "summaries": summaries, "test_metrics_produced": False}
    (output / "readout.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# MambaTSv3 AutoCon representation fusion H12 screen", "", "Validation-only; no test metrics were generated or read.", "", "| Dataset | fusion off | fusion on | on vs off | wins / 3 | max degradation | gate |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for s in summaries: lines.append(f"| {s['dataset']} | {s['fusion_off_mean_val_mse']:.6f} | {s['fusion_on_mean_val_mse']:.6f} | {s['fusion_vs_off_percent']:+.3f}% | {s['wins']}/3 | {s['max_paired_degradation_percent']:+.3f}% | {s['passes_gate']} |")
    (output / "readout.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
