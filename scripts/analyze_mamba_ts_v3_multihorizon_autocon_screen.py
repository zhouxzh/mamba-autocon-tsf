#!/usr/bin/env python3
"""Strict paired reader for the H336/H720 MambaTS v3 exploratory screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import _artifact_exists, _job_specs, _load_json, _validate_plan

DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
HORIZONS = (336, 720)
SEEDS = (2021, 2022, 2023)
PHASES = ("mambatsv3_plain_h336_h720", "mambatsv3_autocon_null_h336_h720", "mambatsv3_autocon_lambda001_h336_h720")


def _json(path: Path) -> dict:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _manifest(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in records:
            raise ValueError(f"invalid or duplicate job id: {job_id!r}")
        records[job_id] = record
    return records


def _flag(command: list, flag: str) -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"missing command flag {flag}") from exc


def _trajectory(checkpoint_dir: Path) -> tuple[float, int, int, float | None]:
    logs = _json(checkpoint_dir / "loss_logs.json")
    values = logs.get("val_loss")
    if not isinstance(values, list) or not values:
        raise ValueError(f"missing validation trajectory: {checkpoint_dir}")
    losses = [_finite(value, "val_loss") for value in values]
    acf = logs.get("autocon_loss")
    acf_mean = None if acf is None else statistics.mean(_finite(value, "autocon_loss") for value in (acf if isinstance(acf, list) else [acf]))
    return min(losses), int(logs.get("best_epoch", losses.index(min(losses)) + 1)), int(logs.get("epochs_ran", len(losses))), acf_mean


def _no_test_artifacts(result_dir: Path, marker: Path) -> None:
    if _json(marker).get("status") != "trained_without_test":
        raise ValueError(f"test-withheld marker missing: {marker}")
    forbidden = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "stream_samples.npz")
    if any((result_dir / name).exists() for name in forbidden):
        raise ValueError(f"test artifact found: {result_dir}")


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = _json(plan_path)
    errors = _validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(item["name"]): item for item in plan["phases"]}
    if tuple(phases) != PHASES:
        raise ValueError(f"unexpected phases: {tuple(phases)}")
    expected = {name: _job_specs(plan, phase, {}, allow_unresolved=False) for name, phase in phases.items()}
    source_hashes = set()
    rows = []
    for phase_name in PHASES:
        phase_root = artifact_root / phase_name
        validation = _json(phase_root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 24:
            raise ValueError(f"phase validation failed: {phase_name}")
        source_hash = str(validation.get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        records = _manifest(phase_root / "logs" / "job_manifest.jsonl")
        if len(records) != 24:
            raise ValueError(f"manifest coverage failed: {phase_name}")
        coordinate_map = {(j["phase"], j["dataset"], j["model"], int(j["pred_len"]), int(j["seed"])): j for j in expected[phase_name]}
        seen = set()
        for record in records.values():
            if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash or not _artifact_exists(record):
                raise ValueError(f"invalid provenance or artifact: {record.get('job_id')}")
            coordinate = (str(record.get("phase")), str(record.get("dataset")), str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")))
            if coordinate not in coordinate_map or coordinate in seen:
                raise ValueError(f"invalid coordinate: {coordinate}")
            seen.add(coordinate)
            job = coordinate_map[coordinate]
            result_dir = Path(str(record["expected_artifact"])).parent
            config = _json(result_dir / "config.json")
            if config.get("plan_job_id") != record.get("job_id") or str(config.get("data")) != str(job["dataset"]):
                raise ValueError(f"config/job mismatch: {result_dir}")
            if int(config.get("pred_len")) not in HORIZONS or int(config.get("run_seed", config.get("seed"))) != int(job["seed"]):
                raise ValueError(f"horizon/seed mismatch: {result_dir}")
            if str(config.get("AutoCon_wnorm")) != "MeanStd" or config.get("mamba_use_acf_memory") is not False:
                raise ValueError(f"architecture/normalization mismatch: {result_dir}")
            if phase_name == PHASES[0] and config.get("AutoCon") is True:
                raise ValueError(f"plain phase has AutoCon config: {result_dir}")
            if phase_name != PHASES[0] and config.get("AutoCon") is not True:
                raise ValueError(f"AutoCon phase missing AutoCon config: {result_dir}")
            _no_test_artifacts(result_dir, Path(str(record["expected_artifact"])))
            checkpoint_dir = Path(_flag(record.get("command", []), "--checkpoints")) / result_dir.name
            best, epoch, epochs, autocon = _trajectory(checkpoint_dir)
            rows.append({"phase": phase_name, "dataset": job["dataset"], "horizon": int(job["pred_len"]), "seed": int(job["seed"]), "best_val_mse": best, "best_epoch": epoch, "epochs_ran": epochs, "autocon_loss_mean": autocon, "elapsed_sec": _finite(record.get("elapsed_sec"), "elapsed_sec")})
        if seen != set(coordinate_map):
            raise ValueError(f"incomplete coordinates: {phase_name}")
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError(f"phase source hashes disagree: {source_hashes}")
    return sorted(rows, key=lambda row: (row["dataset"], row["horizon"], row["seed"], row["phase"]))


def _pairs(rows: list[dict], treatment: str, reference: str) -> list[dict]:
    index = {(r["phase"], r["dataset"], r["horizon"], r["seed"]): r for r in rows}
    output = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                left = index[(treatment, dataset, horizon, seed)]
                right = index[(reference, dataset, horizon, seed)]
                delta = left["best_val_mse"] - right["best_val_mse"]
                output.append({"dataset": dataset, "horizon": horizon, "seed": seed, "treatment": treatment, "reference": reference, "treatment_mse": left["best_val_mse"], "reference_mse": right["best_val_mse"], "delta_mse": delta, "relative_change_pct": 100.0 * delta / right["best_val_mse"]})
    return output


def _summary(pairs: list[dict]) -> list[dict]:
    groups = {}
    for row in pairs:
        groups.setdefault((row["treatment"], row["reference"], row["dataset"], row["horizon"]), []).append(row)
    return [{"treatment": key[0], "reference": key[1], "dataset": key[2], "horizon": key[3], "n_pairs": len(values), "mean_relative_change_pct": statistics.mean(v["relative_change_pct"] for v in values), "wins": sum(v["delta_mse"] < 0 for v in values)} for key, values in sorted(groups.items())]


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve())
    pairs = _pairs(rows, PHASES[2], PHASES[1])
    plain_pairs = _pairs(rows, PHASES[1], PHASES[0])
    summary = _summary(pairs); plain_summary = _summary(plain_pairs)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    _csv(out / "validation_rows.csv", rows); _csv(out / "autocon_paired_deltas.csv", pairs); _csv(out / "autocon_paired_summary.csv", summary); _csv(out / "null_vs_plain_paired_summary.csv", plain_summary)
    lines = ["# MambaTS v3 H336/H720 AutoCon screen", "", "Status: exploratory, validation-only, exact-coordinate paired screen; test metrics are not produced.", "", "| Dataset | H | lambda=.001 vs lambda=0 | wins/3 | lambda=0 vs plain | wins/3 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            active = next(v for v in summary if v["dataset"] == dataset and v["horizon"] == horizon)
            null = next(v for v in plain_summary if v["dataset"] == dataset and v["horizon"] == horizon)
            lines.append(f"| {dataset} | {horizon} | {active['mean_relative_change_pct']:.3f}% | {active['wins']}/{active['n_pairs']} | {null['mean_relative_change_pct']:.3f}% | {null['wins']}/{null['n_pairs']} |")
    lines.extend(["", "Only the lambda=.001 versus lambda=0 column tests the incremental AutoCon objective. The null-versus-plain column measures the architecture/head cost and is not a causal objective comparison.", "", f"- Plan SHA256: `{hashlib.sha256(args.plan.read_bytes()).hexdigest()}`", f"- Rows: `{len(rows)}`; paired AutoCon rows: `{len(pairs)}`; test metrics produced: `false`.\n"])
    (out / "readout.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_rows": len(pairs), "test_metrics_produced": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
