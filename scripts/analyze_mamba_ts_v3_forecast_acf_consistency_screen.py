#!/usr/bin/env python3
"""Strict validation-only reader for MambaTS forecast-ACF consistency."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_mamba_ts_v3_multidataset_autocon_screen as base


PHASES = ("forecast_acf_null_h96", "forecast_acf_active_h96")
DATASETS = base.DATASETS
SEEDS = base.SEEDS
HORIZON = 96


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = base._json(plan_path)
    errors = base._validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(item["name"]): item for item in plan["phases"]}
    if tuple(phases) != PHASES:
        raise ValueError(f"unexpected phases: {tuple(phases)}")
    expected = {
        name: base._job_specs(plan, phase, {}, allow_unresolved=False)
        for name, phase in phases.items()
    }
    source_hashes, rows = set(), []
    for phase_name in PHASES:
        phase_root = artifact_root / phase_name
        validation = base._json(phase_root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 12:
            raise ValueError(f"phase validation failed: {phase_name}")
        provenance = base._json(phase_root / "provenance.json")
        source_hash = str(provenance.get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        records = base._manifest(phase_root / "logs" / "job_manifest.jsonl")
        if len(records) != 12:
            raise ValueError(f"manifest coverage failed: {phase_name}")
        coordinates = {
            (job["phase"], job["dataset"], job["model"], int(job["pred_len"]), int(job["seed"])): job
            for job in expected[phase_name]
        }
        seen = set()
        for record in records.values():
            if (
                record.get("status") != "ok"
                or record.get("source_tree_sha256_before") != source_hash
                or record.get("source_tree_sha256_after") != source_hash
                or not base._artifact_exists(record)
            ):
                raise ValueError(f"invalid record provenance: {record.get('job_id')}")
            coordinate = (
                str(record.get("phase")), str(record.get("dataset")),
                str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")),
            )
            if coordinate not in coordinates or coordinate in seen:
                raise ValueError(f"invalid coordinate: {coordinate}")
            seen.add(coordinate)
            job = coordinates[coordinate]
            result_dir = Path(str(record["expected_artifact"])).parent
            config = base._json(result_dir / "config.json")
            expected_enabled = phase_name == PHASES[1]
            if (
                config.get("plan_job_id") != record.get("job_id")
                or config.get("AutoCon") is not True
                or bool(config.get("mamba_ts_forecast_acf_consistency", False)) != expected_enabled
                or float(config.get("mamba_ts_forecast_acf_weight", 0.0)) != (0.01 if expected_enabled else 0.0)
            ):
                raise ValueError(f"forecast-ACF contract mismatch: {result_dir}")
            if (
                str(config.get("data")) != str(job["dataset"])
                or int(config.get("pred_len")) != HORIZON
                or int(config.get("run_seed", config.get("seed"))) != int(job["seed"])
                or str(config.get("AutoCon_wnorm")) != "MeanStd"
                or float(config.get("AutoCon_lambda")) != float(job["autocon_lambda"])
            ):
                raise ValueError(f"coordinate/loss mismatch: {result_dir}")
            base._no_test_artifacts(result_dir, Path(str(record["expected_artifact"])))
            checkpoint_dir = Path(base._command_value(record.get("command", []), "--checkpoints")) / result_dir.name
            best, epoch, epochs, logs = base._trajectory(checkpoint_dir)
            autocon = base._mean_optional(logs, "autocon_loss")
            forecast_acf = base._mean_optional(logs, "forecast_acf_loss")
            if autocon is None or forecast_acf is None:
                raise ValueError(f"missing loss telemetry: {checkpoint_dir}")
            rows.append({
                "phase": phase_name, "dataset": job["dataset"], "horizon": HORIZON,
                "seed": int(job["seed"]), "best_val_mse": best, "best_epoch": epoch,
                "epochs_ran": epochs, "elapsed_sec": base._finite(record["elapsed_sec"], "elapsed_sec"),
                "autocon_loss_mean": autocon, "forecast_acf_loss_mean": forecast_acf,
            })
        if seen != set(coordinates):
            raise ValueError(f"incomplete coordinates: {phase_name}")
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError(f"source hashes disagree: {source_hashes}")
    return sorted(rows, key=lambda row: (row["phase"], row["dataset"], row["seed"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve())
    pairs = base.paired(rows, PHASES[1], PHASES[0])
    summary = base.summarize(pairs)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base._write_csv(output / "validation_rows.csv", rows)
    base._write_csv(output / "paired_deltas.csv", pairs)
    base._write_csv(output / "paired_summary.csv", summary)
    lines = [
        "# MambaTS v3 Forecast ACF Consistency Screen", "",
        "Status: exploratory, validation-only; test metrics are not produced.", "",
        "| Dataset | Null mean MSE | Active mean MSE | Relative change | Wins / 3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        null = [row["best_val_mse"] for row in rows if row["phase"] == PHASES[0] and row["dataset"] == dataset]
        active = [row["best_val_mse"] for row in rows if row["phase"] == PHASES[1] and row["dataset"] == dataset]
        item = next(row for row in summary if row["dataset"] == dataset)
        lines.append(
            f"| {dataset} | {sum(null) / len(null):.6f} | {sum(active) / len(active):.6f} | "
            f"{item['mean_relative_change_pct']:.3f}% | {item['wins']}/{item['n_pairs']} |"
        )
    lines.extend([
        "",
        "The active arm adds a parameter-free training-only loss over the normalized temporal ACF of the forecast. Lag targets are selected from the training split before training; the inference path is unchanged.",
        f"- Plan SHA256: `{hashlib.sha256(args.plan.read_bytes()).hexdigest()}`",
        f"- Rows: `{len(rows)}`; test metrics produced: `false`.",
    ])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_rows": len(pairs), "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
