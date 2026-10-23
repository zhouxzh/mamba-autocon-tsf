#!/usr/bin/env python3
"""Bind the frozen fusion candidate to its same-snapshot H12 baselines."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from scripts.analyze_metro_pemsbay_mambatsv3_repr_fusion_h12_screen import _json, _read
from scripts.run_training_plan import _job_specs, _validate_plan

DATASETS = ("metr_la", "pems_bay")
SEEDS = (2811, 2812, 2813)
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3")


def _baseline_rows(root: Path, plan: dict) -> tuple[list[dict], str]:
    phase = plan["phases"][0]
    expected = {(j["dataset"], j["model"], j["seed"]): j for j in _job_specs(plan, phase, {}, allow_unresolved=True)}
    phase_root = root / phase["name"]
    validation = _json(phase_root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != len(expected):
        raise ValueError("baseline phase validation failed")
    source = str(_json(phase_root / "provenance.json").get("source_tree_sha256"))
    records = [json.loads(line) for line in (phase_root / "logs/job_manifest.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != len(expected): raise ValueError("baseline manifest coverage failed")
    seen = set(); rows = []
    for record in records:
        key = (str(record["dataset"]), str(record["model"]), int(record["seed"]))
        if key not in expected or key in seen or record.get("status") != "ok": raise ValueError(f"invalid baseline coordinate: {key}")
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source: raise ValueError("baseline source drift")
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file(): raise ValueError(f"missing baseline completion: {result}")
        if any((result / n).exists() for n in ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")): raise ValueError(f"test artifact: {result}")
        config = _json(result / "config.json")
        if config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False or config.get("mamba_acf_relation_loss") is not False: raise ValueError(f"baseline objective mismatch: {result}")
        ck = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(v) for v in _json(ck / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(v) for v in losses): raise ValueError(f"invalid baseline loss: {ck}")
        rows.append({"dataset": key[0], "model": key[1], "seed": key[2], "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
        seen.add(key)
    if seen != set(expected): raise ValueError("incomplete baseline coordinates")
    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate-plan", type=Path, required=True); parser.add_argument("--candidate-root", type=Path, required=True); parser.add_argument("--baseline-plan", type=Path, required=True); parser.add_argument("--baseline-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    candidate_plan = _json(args.candidate_plan); baseline_plan = _json(args.baseline_plan)
    if _validate_plan(candidate_plan) or _validate_plan(baseline_plan): raise ValueError("invalid plan")
    candidate_rows, candidate_source = _read(args.candidate_root, candidate_plan, "fusion_on_h12")
    baseline_rows, baseline_source = _baseline_rows(args.baseline_root, baseline_plan)
    if candidate_source != baseline_source: raise ValueError(f"source hashes differ: {candidate_source} vs {baseline_source}")
    cindex = {(r["dataset"], r["seed"]): r for r in candidate_rows}; summaries = []; all_rows = []
    for dataset in DATASETS:
        cvals = [cindex[(dataset, seed)]["best_val_mse"] for seed in SEEDS]
        means = {model: statistics.fmean(r["best_val_mse"] for r in baseline_rows if r["dataset"] == dataset and r["model"] == model) for model in BASELINES}
        strongest = min(means, key=means.get); strongest_mean = means[strongest]; cmean = statistics.fmean(cvals)
        summaries.append({"dataset": dataset, "candidate_mean_val_mse": cmean, "strongest_baseline": strongest, "strongest_baseline_mean_val_mse": strongest_mean, "candidate_vs_strongest_percent": 100 * (cmean / strongest_mean - 1), "passes_external_gate": cmean <= strongest_mean * 1.02, **{f"{m}_mean_val_mse": v for m, v in means.items()}})
        for seed in SEEDS:
            all_rows.append({"dataset": dataset, "seed": seed, "model": "MambaTSv3AutoCon_repr_fusion", "best_val_mse": cindex[(dataset, seed)]["best_val_mse"]})
    for r in baseline_rows: all_rows.append(r)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_rows.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=sorted({k for r in all_rows for k in r})); writer.writeheader(); writer.writerows(all_rows)
    with (output / "summary.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=sorted(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    payload = {"candidate_source_tree_sha256": candidate_source, "baseline_source_tree_sha256": baseline_source, "candidate_plan_sha256": __import__('hashlib').sha256(args.candidate_plan.read_bytes()).hexdigest(), "baseline_plan_sha256": __import__('hashlib').sha256(args.baseline_plan.read_bytes()).hexdigest(), "summaries": summaries, "test_metrics_produced": False}
    (output / "readout.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# MambaTSv3 AutoCon representation-fusion H12 baseline confirmation", "", "Validation-only; candidate and baselines share one source snapshot; no test metrics were generated or read.", "", "| Dataset | candidate | strongest baseline | candidate vs strongest | gate |", "| --- | ---: | --- | ---: | --- |"]
    for s in summaries: lines.append(f"| {s['dataset']} | {s['candidate_mean_val_mse']:.6f} | {s['strongest_baseline']} ({s['strongest_baseline_mean_val_mse']:.6f}) | {s['candidate_vs_strongest_percent']:+.3f}% | {s['passes_external_gate']} |")
    (output / "readout.md").write_text("\n".join(lines) + "\n"); print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
