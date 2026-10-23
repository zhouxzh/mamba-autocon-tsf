#!/usr/bin/env python3
"""Strict reader for the long-horizon MambaTS v4 ACF-memory selector screen."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mamba_ts_v3_multidataset_autocon_screen as base


PHASES = ("v4_memory_control_h336_h720", "v4_memory_selector_h336_h720")
DATASETS = base.DATASETS
HORIZONS = (336, 720)


def paired(rows: list[dict], treatment: str, reference: str) -> list[dict]:
    index = {(row["phase"], row["dataset"], row["horizon"], row["seed"]): row for row in rows}
    output = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in (2021, 2022, 2023):
                candidate = index[(treatment, dataset, horizon, seed)]
                control = index[(reference, dataset, horizon, seed)]
                delta = candidate["best_val_mse"] - control["best_val_mse"]
                output.append({"treatment": treatment, "reference": reference, "dataset": dataset, "horizon": horizon, "seed": seed, "treatment_mse": candidate["best_val_mse"], "reference_mse": control["best_val_mse"], "delta_mse": delta, "relative_change_pct": 100.0 * delta / control["best_val_mse"]})
    return output


def summarize(pairs: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            values = [row for row in pairs if row["dataset"] == dataset and row["horizon"] == horizon]
            output.append({"treatment": values[0]["treatment"], "reference": values[0]["reference"], "dataset": dataset, "horizon": horizon, "n_pairs": len(values), "mean_delta_mse": statistics.mean(row["delta_mse"] for row in values), "mean_relative_change_pct": statistics.mean(row["relative_change_pct"] for row in values), "wins": sum(row["delta_mse"] < 0 for row in values)})
    return output


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = base._json(plan_path)
    errors = base._validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(item["name"]): item for item in plan["phases"]}
    if tuple(phases) != PHASES:
        raise ValueError(f"unexpected phases: {tuple(phases)}")
    expected = {name: base._job_specs(plan, phase, {}, allow_unresolved=False) for name, phase in phases.items()}
    rows, source_hashes = [], set()
    for phase_name in PHASES:
        phase_root = artifact_root / phase_name
        validation = base._json(phase_root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 24:
            raise ValueError(f"phase validation failed: {phase_name}")
        source_hash = str(base._json(phase_root / "provenance.json").get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        records = base._manifest(phase_root / "logs" / "job_manifest.jsonl")
        if len(records) != 24:
            raise ValueError(f"manifest coverage failed: {phase_name}")
        coordinates = {(job["phase"], job["dataset"], job["model"], int(job["pred_len"]), int(job["seed"])): job for job in expected[phase_name]}
        seen = set()
        for record in records.values():
            if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash or not base._artifact_exists(record):
                raise ValueError(f"invalid provenance: {record.get('job_id')}")
            coordinate = (str(record.get("phase")), str(record.get("dataset")), str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")))
            if coordinate not in coordinates or coordinate in seen:
                raise ValueError(f"invalid coordinate: {coordinate}")
            seen.add(coordinate); job = coordinates[coordinate]
            result_dir = Path(str(record["expected_artifact"])).parent
            config = base._json(result_dir / "config.json")
            selected = phase_name == PHASES[1]
            if (config.get("plan_job_id") != record.get("job_id") or config.get("AutoCon") is not True or bool(config.get("mamba_use_acf_memory", False)) is not True or bool(config.get("mamba_ts_acf_memory_selector", False)) != selected or str(config.get("AutoCon_wnorm")) != "MeanStd" or str(config.get("AutoCon_acf_source")) != "raw" or list(config.get("mamba_acf_lags", [])) != [24, 48] or float(config.get("AutoCon_lambda")) != float(job["autocon_lambda"])):
                raise ValueError(f"selector contract mismatch: {result_dir}")
            if str(config.get("data")) != str(job["dataset"]) or int(config.get("pred_len")) != int(job["pred_len"]):
                raise ValueError(f"coordinate mismatch: {result_dir}")
            base._no_test_artifacts(result_dir, Path(str(record["expected_artifact"])))
            checkpoint_dir = Path(base._command_value(record.get("command", []), "--checkpoints")) / result_dir.name
            best, epoch, epochs, logs = base._trajectory(checkpoint_dir)
            autocon = base._mean_optional(logs, "autocon_loss")
            if autocon is None:
                raise ValueError(f"missing AutoCon telemetry: {checkpoint_dir}")
            rows.append({"phase": phase_name, "dataset": job["dataset"], "horizon": int(job["pred_len"]), "seed": int(job["seed"]), "best_val_mse": best, "best_epoch": epoch, "epochs_ran": epochs, "elapsed_sec": base._finite(record["elapsed_sec"], "elapsed_sec"), "autocon_loss_mean": autocon})
        if seen != set(coordinates):
            raise ValueError(f"incomplete coordinates: {phase_name}")
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError(f"source hashes disagree: {source_hashes}")
    return sorted(rows, key=lambda row: (row["phase"], row["dataset"], row["horizon"], row["seed"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve())
    pairs = paired(rows, PHASES[1], PHASES[0]); summary = summarize(pairs)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    base._write_csv(output / "validation_rows.csv", rows); base._write_csv(output / "paired_deltas.csv", pairs); base._write_csv(output / "paired_summary.csv", summary)
    lines = ["# MambaTS v4 Long-Horizon ACF Memory Selector Screen", "", "Status: exploratory, validation-only; test metrics are not produced.", "", "| Dataset | Horizon | Static memory MSE | Selector MSE | Relative change | Wins / 3 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            control = [r["best_val_mse"] for r in rows if r["phase"] == PHASES[0] and r["dataset"] == dataset and r["horizon"] == horizon]
            selected = [r["best_val_mse"] for r in rows if r["phase"] == PHASES[1] and r["dataset"] == dataset and r["horizon"] == horizon]
            item = next(r for r in summary if r["dataset"] == dataset and r["horizon"] == horizon)
            lines.append(f"| {dataset} | {horizon} | {sum(control)/len(control):.6f} | {sum(selected)/len(selected):.6f} | {item['mean_relative_change_pct']:.3f}% | {item['wins']}/{item['n_pairs']} |")
    lines.extend(["", "The selector is a zero-gated, representation-driven correction to the same immutable train-only ACF memory used in the control. Test metrics remain withheld.", f"- Plan SHA256: `{hashlib.sha256(args.plan.read_bytes()).hexdigest()}`", f"- Rows: `{len(rows)}`; test metrics produced: `false`."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_rows": len(pairs), "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
