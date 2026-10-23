#!/usr/bin/env python3
"""Strict reader for the MambaTS v3 variable-permutation screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mamba_ts_v3_multidataset_autocon_screen as base

PHASES = ("global_repr_active_h96", "permuted_repr_null_h96", "permuted_repr_active_h96")
DATASETS, SEEDS, HORIZON = base.DATASETS, base.SEEDS, 96


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = base._json(plan_path); errors = base._validate_plan(plan)
    if errors: raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(p["name"]): p for p in plan["phases"]}
    if tuple(phases) != PHASES: raise ValueError(f"unexpected phases: {tuple(phases)}")
    expected = {name: base._job_specs(plan, phase, {}, allow_unresolved=False) for name, phase in phases.items()}
    hashes, rows = set(), []
    for phase_name in PHASES:
        root = artifact_root / phase_name; validation = base._json(root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 12: raise ValueError(f"phase validation failed: {phase_name}")
        source_hash = str(validation.get("source_tree_sha256", "")); hashes.add(source_hash)
        records = base._manifest(root / "logs" / "job_manifest.jsonl")
        if len(records) != 12: raise ValueError(f"manifest coverage failed: {phase_name}")
        coordinates = {(j["phase"], j["dataset"], j["model"], int(j["pred_len"]), int(j["seed"])): j for j in expected[phase_name]}; seen = set()
        for record in records.values():
            if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash or not base._artifact_exists(record): raise ValueError(f"invalid provenance/artifact: {record.get('job_id')}")
            coordinate = (str(record.get("phase")), str(record.get("dataset")), str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")))
            if coordinate not in coordinates or coordinate in seen: raise ValueError(f"invalid coordinate: {coordinate}")
            seen.add(coordinate); job = coordinates[coordinate]; result = Path(str(record["expected_artifact"])).parent; config = base._json(result / "config.json")
            expected_permutation = phase_name != PHASES[0]
            if config.get("plan_job_id") != record.get("job_id") or config.get("AutoCon") is not True or config.get("mamba_ts_repr_forecast_fusion") is not True or bool(config.get("mamba_ts_variable_permutation")) != expected_permutation: raise ValueError(f"architecture contract mismatch: {result}")
            if str(config.get("data")) != str(job["dataset"]) or int(config.get("pred_len")) != HORIZON or int(config.get("run_seed", config.get("seed"))) != int(job["seed"]) or float(config.get("AutoCon_lambda")) != float(job["autocon_lambda"]): raise ValueError(f"coordinate/lambda mismatch: {result}")
            if str(config.get("AutoCon_wnorm")) != "MeanStd" or config.get("mamba_use_acf_memory") is not False: raise ValueError(f"normalization/memory mismatch: {result}")
            base._no_test_artifacts(result, Path(str(record["expected_artifact"])))
            checkpoint = Path(base._command_value(record.get("command", []), "--checkpoints")) / result.name; best, epoch, epochs, logs = base._trajectory(checkpoint)
            rows.append({"phase": phase_name, "dataset": job["dataset"], "horizon": HORIZON, "seed": int(job["seed"]), "best_val_mse": best, "best_epoch": epoch, "epochs_ran": epochs, "elapsed_sec": base._finite(record["elapsed_sec"], "elapsed_sec"), "autocon_loss_mean": base._mean_optional(logs, "autocon_loss")})
        if seen != set(coordinates): raise ValueError(f"incomplete coordinates: {phase_name}")
    if len(hashes) != 1 or not next(iter(hashes)): raise ValueError(f"source hashes disagree: {hashes}")
    return sorted(rows, key=lambda r: (r["phase"], r["dataset"], r["seed"]))


def pairs(rows: list[dict], treatment: str, reference: str) -> list[dict]:
    index = {(r["phase"], r["dataset"], r["seed"]): r for r in rows}; output = []
    for dataset in DATASETS:
        for seed in SEEDS:
            left, right = index[(treatment, dataset, seed)], index[(reference, dataset, seed)]; delta = left["best_val_mse"] - right["best_val_mse"]
            output.append({"treatment": treatment, "reference": reference, "dataset": dataset, "horizon": HORIZON, "seed": seed, "delta_mse": delta, "relative_change_pct": 100.0 * delta / right["best_val_mse"]})
    return output


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["treatment"], row["reference"], row["dataset"])].append(row)
    return [{"treatment": key[0], "reference": key[1], "dataset": key[2], "n_pairs": len(values), "mean_relative_change_pct": statistics.mean(v["relative_change_pct"] for v in values), "wins": sum(v["delta_mse"] < 0 for v in values)} for key, values in sorted(groups.items())]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve()); null_pairs = pairs(rows, PHASES[2], PHASES[1]); architecture_pairs = pairs(rows, PHASES[2], PHASES[0]); null_summary = summarize(null_pairs); architecture_summary = summarize(architecture_pairs); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", rows); _write_csv(out / "autocon_null_paired_deltas.csv", null_pairs); _write_csv(out / "autocon_null_summary.csv", null_summary); _write_csv(out / "permutation_vs_global_summary.csv", architecture_summary)
    lines = ["# MambaTS v3 variable permutation screen", "", "Status: exploratory, validation-only; test metrics are not produced.", "", "| Dataset | permutation active vs null | wins/3 | permutation active vs global active | wins/3 |", "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        null = next(v for v in null_summary if v["dataset"] == dataset); arch = next(v for v in architecture_summary if v["dataset"] == dataset)
        lines.append(f"| {dataset} | {null['mean_relative_change_pct']:.3f}% | {null['wins']}/3 | {arch['mean_relative_change_pct']:.3f}% | {arch['wins']}/3 |")
    lines.extend(["", "The first contrast isolates AutoCon within the permutation architecture; the second isolates training-time variable permutation at fixed active AutoCon.", f"- Plan SHA256: `{hashlib.sha256(args.plan.read_bytes()).hexdigest()}`", f"- Rows: `{len(rows)}`; test metrics produced: `false`."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print({"rows": len(rows), "null_pairs": len(null_pairs), "architecture_pairs": len(architecture_pairs), "test_metrics_produced": False}); return 0


if __name__ == "__main__": raise SystemExit(main())
