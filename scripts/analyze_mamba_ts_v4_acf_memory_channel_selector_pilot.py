#!/usr/bin/env python3
"""Read the validation-only channel-conditioned selector pilot."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mamba_ts_v4_acf_memory_selector_multihorizon as base


PHASES = ("v4_global_selector_problem_coordinates", "v4_channel_selector_problem_coordinates")
DATASETS = ("ETTh1", "traffic")
HORIZONS = (336, 720)


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = base.base._json(plan_path)
    errors = base.base._validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(item["name"]): item for item in plan["phases"]}
    rows = []
    hashes = set()
    for phase_name in PHASES:
        phase_root = artifact_root / phase_name
        validation = base.base._json(phase_root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 4:
            raise ValueError(f"phase validation failed: {phase_name}")
        provenance = base.base._json(phase_root / "provenance.json")
        source_hash = str(provenance.get("source_tree_sha256", ""))
        hashes.add(source_hash)
        records = base.base._manifest(phase_root / "logs" / "job_manifest.jsonl")
        if len(records) != 4:
            raise ValueError(f"manifest coverage failed: {phase_name}")
        expected = {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
        seen = set()
        for record in records.values():
            if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash or not base.base._artifact_exists(record):
                raise ValueError(f"invalid provenance: {record.get('job_id')}")
            result_dir = Path(str(record["expected_artifact"])).parent
            config = base.base._json(result_dir / "config.json")
            dataset = str(config.get("data")); horizon = int(config.get("pred_len"))
            selected = phase_name == PHASES[1]
            if (dataset, horizon) not in expected or (dataset, horizon) in seen:
                raise ValueError(f"invalid coordinate: {(dataset, horizon)}")
            if bool(config.get("mamba_ts_acf_memory_selector", False)) is not True or bool(config.get("mamba_ts_acf_memory_channel_selector", False)) != selected or config.get("AutoCon") is not True or bool(config.get("mamba_use_acf_memory", False)) is not True:
                raise ValueError(f"selector contract mismatch: {result_dir}")
            base.base._no_test_artifacts(result_dir, Path(str(record["expected_artifact"])))
            checkpoint_dir = Path(base.base._command_value(record.get("command", []), "--checkpoints")) / result_dir.name
            best, epoch, epochs, logs = base.base._trajectory(checkpoint_dir)
            if base.base._mean_optional(logs, "autocon_loss") is None:
                raise ValueError(f"missing AutoCon telemetry: {checkpoint_dir}")
            rows.append({"phase": phase_name, "dataset": dataset, "horizon": horizon, "seed": 2021, "best_val_mse": best, "best_epoch": epoch, "epochs_ran": epochs, "elapsed_sec": base.base._finite(record["elapsed_sec"], "elapsed_sec")})
            seen.add((dataset, horizon))
        if seen != expected:
            raise ValueError(f"incomplete coordinates: {phase_name}")
    if len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError(f"source hashes disagree: {hashes}")
    return sorted(rows, key=lambda row: (row["dataset"], row["horizon"], row["phase"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve())
    index = {(r["phase"], r["dataset"], r["horizon"]): r for r in rows}
    deltas = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            control = index[(PHASES[0], dataset, horizon)]
            treatment = index[(PHASES[1], dataset, horizon)]
            delta = treatment["best_val_mse"] - control["best_val_mse"]
            relative = 100.0 * delta / control["best_val_mse"]
            deltas.append(relative)
            print(f"{dataset} H{horizon}: global={control['best_val_mse']:.6f} channel={treatment['best_val_mse']:.6f} change={relative:+.3f}%")
    print(f"mean_change={statistics.mean(deltas):+.3f}% rows={len(rows)} test_metrics_produced=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
