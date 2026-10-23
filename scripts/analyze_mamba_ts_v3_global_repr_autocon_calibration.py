#!/usr/bin/env python3
"""Strict reader for global prediction-connected MambaTS v3 AutoCon calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mamba_ts_v3_multidataset_autocon_screen as base

PHASES = ("global_repr_null_h96", "global_repr_lambda0003_h96", "global_repr_lambda001_h96", "global_repr_lambda003_h96", "global_repr_lambda003_warm4_h96")
DATASETS, SEEDS, HORIZON = base.DATASETS, base.SEEDS, 96


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _check_schedule(logs: dict, phase: str) -> None:
    values = logs.get("effective_lambda")
    if not isinstance(values, list) or not values:
        raise ValueError("missing effective_lambda telemetry")
    numeric = [base._finite(value, "effective_lambda") for value in values]
    if phase == "global_repr_null_h96":
        expected = 0.0
        if any(value != expected for value in numeric):
            raise ValueError("null effective lambda mismatch")
    elif phase == "global_repr_lambda003_warm4_h96":
        expected_prefix = (0.00075, 0.0015, 0.00225, 0.003)
        if len(numeric) < 4 or any(abs(actual - expected) > 1e-10 for actual, expected in zip(numeric[:4], expected_prefix)) or any(abs(actual - 0.003) > 1e-10 for actual in numeric[4:]):
            raise ValueError("warm-up effective lambda schedule mismatch")
    else:
        expected = {"global_repr_lambda0003_h96": 0.0003, "global_repr_lambda001_h96": 0.001, "global_repr_lambda003_h96": 0.003}[phase]
        if any(abs(value - expected) > 1e-10 for value in numeric):
            raise ValueError("active effective lambda mismatch")


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = base._json(plan_path); errors = base._validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(p["name"]): p for p in plan["phases"]}
    if tuple(phases) != PHASES:
        raise ValueError(f"unexpected phases: {tuple(phases)}")
    expected = {name: base._job_specs(plan, phase, {}, allow_unresolved=False) for name, phase in phases.items()}
    source_hashes, rows = set(), []
    for phase_name in PHASES:
        root = artifact_root / phase_name; validation = base._json(root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 12:
            raise ValueError(f"phase validation failed: {phase_name}")
        source_hash = str(validation.get("source_tree_sha256", "")); source_hashes.add(source_hash)
        records = base._manifest(root / "logs" / "job_manifest.jsonl")
        if len(records) != 12:
            raise ValueError(f"manifest coverage failed: {phase_name}")
        coordinates = {(j["phase"], j["dataset"], j["model"], int(j["pred_len"]), int(j["seed"])): j for j in expected[phase_name]}
        seen = set()
        for record in records.values():
            if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash or not base._artifact_exists(record):
                raise ValueError(f"invalid provenance/artifact: {record.get('job_id')}")
            coordinate = (str(record.get("phase")), str(record.get("dataset")), str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")))
            if coordinate not in coordinates or coordinate in seen:
                raise ValueError(f"invalid coordinate: {coordinate}")
            seen.add(coordinate); job = coordinates[coordinate]; result = Path(str(record["expected_artifact"])).parent
            config = base._json(result / "config.json")
            if config.get("plan_job_id") != record.get("job_id") or config.get("AutoCon") is not True or config.get("mamba_ts_repr_forecast_fusion") is not True or config.get("mamba_ts_repr_forecast_channelwise") is not False:
                raise ValueError(f"fusion architecture mismatch: {result}")
            if str(config.get("data")) != str(job["dataset"]) or int(config.get("pred_len")) != HORIZON or int(config.get("run_seed", config.get("seed"))) != int(job["seed"]) or float(config.get("AutoCon_lambda")) != float(job["autocon_lambda"]):
                raise ValueError(f"coordinate/lambda mismatch: {result}")
            if str(config.get("AutoCon_wnorm")) != "MeanStd" or config.get("mamba_use_acf_memory") is not False:
                raise ValueError(f"normalization/memory mismatch: {result}")
            base._no_test_artifacts(result, Path(str(record["expected_artifact"])))
            checkpoint = Path(base._command_value(record.get("command", []), "--checkpoints")) / result.name
            best, epoch, epochs, logs = base._trajectory(checkpoint); _check_schedule(logs, phase_name)
            autocon = base._mean_optional(logs, "autocon_loss")
            if autocon is None:
                raise ValueError(f"missing AutoCon telemetry: {checkpoint}")
            rows.append({"phase": phase_name, "dataset": job["dataset"], "horizon": HORIZON, "seed": int(job["seed"]), "autocon_lambda": float(config["AutoCon_lambda"]), "autocon_warmup": int(config["AutoCon_warmup_epochs"]), "best_val_mse": best, "best_epoch": epoch, "epochs_ran": epochs, "elapsed_sec": base._finite(record["elapsed_sec"], "elapsed_sec"), "autocon_loss_mean": autocon})
        if seen != set(coordinates):
            raise ValueError(f"incomplete coordinates: {phase_name}")
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError(f"source hashes disagree: {source_hashes}")
    return sorted(rows, key=lambda r: (r["phase"], r["dataset"], r["seed"]))


def paired(rows: list[dict], treatment: str, reference: str) -> list[dict]:
    index = {(r["phase"], r["dataset"], r["seed"]): r for r in rows}; output = []
    for dataset in DATASETS:
        for seed in SEEDS:
            left, right = index[(treatment, dataset, seed)], index[(reference, dataset, seed)]
            delta = left["best_val_mse"] - right["best_val_mse"]
            output.append({"treatment": treatment, "reference": reference, "dataset": dataset, "horizon": HORIZON, "seed": seed, "delta_mse": delta, "relative_change_pct": 100.0 * delta / right["best_val_mse"]})
    return output


def summary(pairs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in pairs:
        groups[(row["treatment"], row["reference"], row["dataset"])].append(row)
    return [{"treatment": k[0], "reference": k[1], "dataset": k[2], "n_pairs": len(v), "mean_relative_change_pct": statistics.mean(x["relative_change_pct"] for x in v), "wins": sum(x["delta_mse"] < 0 for x in v)} for k, v in sorted(groups.items())]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve()); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    all_pairs, all_summary = [], []
    for phase in PHASES[1:]:
        pairs = paired(rows, phase, PHASES[0]); all_pairs.extend(pairs); all_summary.extend(summary(pairs))
    warm_pairs = paired(rows, PHASES[4], PHASES[3]); warm_summary = summary(warm_pairs)
    _write_csv(out / "validation_rows.csv", rows); _write_csv(out / "null_paired_deltas.csv", all_pairs); _write_csv(out / "null_paired_summary.csv", all_summary); _write_csv(out / "warmup_paired_summary.csv", warm_summary)
    eligible = []
    for phase in PHASES[1:]:
        items = [row for row in all_summary if row["treatment"] == phase]
        if all(row["mean_relative_change_pct"] <= 0 for row in items) and sum(row["wins"] >= 2 for row in items) >= 3:
            eligible.append(phase)
    lines = ["# MambaTS v3 global representation AutoCon calibration", "", "Status: exploratory, validation-only; strict reader verified five phase manifests, source identity, exact coordinates, fusion flags, effective-lambda schedules, and zero test artifacts.", "", "| Treatment vs null | ETTh1 | ETTm1 | Electricity | Traffic | eligible by predeclared rule |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for phase in PHASES[1:]:
        values = {row["dataset"]: row for row in all_summary if row["treatment"] == phase}
        lines.append("| {} | {} | {} | {} | {} | {} |".format(phase, *[f"{values[d]['mean_relative_change_pct']:.3f}% ({values[d]['wins']}/3)" for d in DATASETS], "yes" if phase in eligible else "no"))
    lines.extend(["", "Warm-up contrast: `lambda=.003 warm4` versus `lambda=.003 warm0`.", "", "| Dataset | relative change | wins/3 |", "| --- | ---: | ---: |"])
    for item in warm_summary:
        lines.append(f"| {item['dataset']} | {item['mean_relative_change_pct']:.3f}% | {item['wins']}/3 |")
    lines.extend(["", f"- Eligible arms: `{eligible}`", f"- Plan SHA256: `{hashlib.sha256(args.plan.read_bytes()).hexdigest()}`", f"- Rows: `{len(rows)}`; test metrics produced: `false`."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pair_rows": len(all_pairs), "eligible": eligible, "test_metrics_produced": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
