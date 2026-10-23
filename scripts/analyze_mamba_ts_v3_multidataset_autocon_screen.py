#!/usr/bin/env python3
"""Strict reader for the MambaTS v3 multidataset AutoCon null screen.

Only the pure forecasting validation trajectory is used as the primary
metric. The reader rejects missing coordinates, source/provenance drift,
normalization or lambda mismatches, and any test artifacts.
"""

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

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import _artifact_exists, _job_specs, _load_json, _validate_plan


PHASES = ("mambatsv3_autocon_null_h96", "mambatsv3_autocon_lambda001_h96")
DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
SEEDS = (2021, 2022, 2023)
HORIZON = 96


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _manifest(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed manifest line {number}: {path}") from exc
        job_id = record.get("job_id") if isinstance(record, dict) else None
        if not job_id or job_id in records:
            raise ValueError(f"invalid or duplicate job ID in {path}: {job_id!r}")
        records[str(job_id)] = record
    return records


def _command_value(command: list, flag: str) -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"missing command flag {flag}") from exc


def _trajectory(path: Path) -> tuple[float, int, int, dict]:
    logs = _json(path / "loss_logs.json")
    values = logs.get("val_loss")
    if not isinstance(values, list) or not values:
        raise ValueError(f"missing val_loss: {path}")
    numeric = [_finite(value, f"{path}:val_loss") for value in values]
    best = min(numeric)
    return best, int(logs.get("best_epoch", numeric.index(best) + 1)), int(logs.get("epochs_ran", len(numeric))), logs


def _mean_optional(logs: dict, key: str) -> float | None:
    values = logs.get(key)
    if values is None:
        return None
    if not isinstance(values, list):
        values = [values]
    numeric = [_finite(value, key) for value in values if value is not None]
    return float(statistics.mean(numeric)) if numeric else None


def _no_test_artifacts(result_dir: Path, marker: Path) -> None:
    if _json(marker).get("status") != "trained_without_test":
        raise ValueError(f"test-withheld marker missing: {marker}")
    forbidden = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "stream_samples.npz")
    if any((result_dir / name).exists() for name in forbidden):
        raise ValueError(f"test artifact found in {result_dir}")


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
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 12:
            raise ValueError(f"phase validation failed: {phase_name}")
        source_hash = str(validation.get("source_tree_sha256", ""))
        source_hashes.add(source_hash)
        records = _manifest(phase_root / "logs" / "job_manifest.jsonl")
        if len(records) != 12:
            raise ValueError(f"manifest coverage failed: {phase_name}")
        coordinate_map = {(j["phase"], j["dataset"], j["model"], int(j["pred_len"]), int(j["seed"])): j for j in expected[phase_name]}
        seen = set()
        for record in records.values():
            if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
                raise ValueError(f"invalid record provenance: {record.get('job_id')}")
            coordinate = (str(record.get("phase")), str(record.get("dataset")), str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")))
            if coordinate not in coordinate_map or coordinate in seen or not _artifact_exists(record):
                raise ValueError(f"invalid artifact coordinate: {coordinate}")
            seen.add(coordinate)
            job = coordinate_map[coordinate]
            result_dir = Path(str(record["expected_artifact"])).parent
            config = _json(result_dir / "config.json")
            if config.get("plan_job_id") != record.get("job_id") or config.get("AutoCon") is not True:
                raise ValueError(f"model/job contract mismatch: {result_dir}")
            if str(config.get("data")) != str(job["dataset"]) or int(config.get("pred_len")) != HORIZON or int(config.get("run_seed", config.get("seed"))) != int(job["seed"]):
                raise ValueError(f"coordinate mismatch: {result_dir}")
            if str(config.get("AutoCon_wnorm")) != "MeanStd" or config.get("mamba_use_acf_memory") is not False:
                raise ValueError(f"normalization or architecture mismatch: {result_dir}")
            actual_lambda = float(config.get("AutoCon_lambda"))
            if actual_lambda != float(job["autocon_lambda"]):
                raise ValueError(f"lambda mismatch: {result_dir}")
            _no_test_artifacts(result_dir, Path(str(record["expected_artifact"])))
            checkpoint_dir = Path(_command_value(record.get("command", []), "--checkpoints")) / result_dir.name
            best, best_epoch, epochs_ran, logs = _trajectory(checkpoint_dir)
            if _mean_optional(logs, "autocon_loss") is None:
                raise ValueError(f"missing AutoCon telemetry: {checkpoint_dir}")
            rows.append({
                "phase": phase_name, "dataset": job["dataset"], "horizon": HORIZON, "seed": int(job["seed"]),
                "model": job["model"], "autocon_lambda": actual_lambda, "best_val_mse": best,
                "best_epoch": best_epoch, "epochs_ran": epochs_ran, "elapsed_sec": _finite(record.get("elapsed_sec"), "elapsed_sec"),
                "autocon_loss_mean": _mean_optional(logs, "autocon_loss"),
                "config_path": str((result_dir / "config.json").resolve()), "checkpoint_path": str((checkpoint_dir / "loss_logs.json").resolve()),
            })
        if seen != set(coordinate_map):
            raise ValueError(f"incomplete exact coordinates: {phase_name}")
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError(f"phase source hashes disagree: {source_hashes}")
    return sorted(rows, key=lambda row: (row["phase"], row["dataset"], row["seed"]))


def paired(rows: list[dict], treatment: str, reference: str) -> list[dict]:
    index = {(r["phase"], r["dataset"], r["seed"]): r for r in rows}
    output = []
    for dataset in DATASETS:
        for seed in SEEDS:
            left, right = index[(treatment, dataset, seed)], index[(reference, dataset, seed)]
            delta = float(left["best_val_mse"]) - float(right["best_val_mse"])
            output.append({"treatment": treatment, "reference": reference, "dataset": dataset, "horizon": HORIZON, "seed": seed, "treatment_mse": left["best_val_mse"], "reference_mse": right["best_val_mse"], "delta_mse": delta, "relative_change_pct": 100.0 * delta / float(right["best_val_mse"])})
    return output


def summarize(pairs: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in pairs:
        grouped[(row["treatment"], row["reference"], row["dataset"])].append(row)
    output = []
    for (treatment, reference, dataset), values in sorted(grouped.items()):
        deltas = [float(value["delta_mse"]) for value in values]
        ci = float(stats.t.ppf(0.975, len(deltas) - 1) * stats.sem(deltas)) if len(deltas) > 1 else 0.0
        output.append({"treatment": treatment, "reference": reference, "dataset": dataset, "horizon": HORIZON, "n_pairs": len(deltas), "mean_delta_mse": statistics.mean(deltas), "mean_relative_change_pct": statistics.mean(float(value["relative_change_pct"]) for value in values), "wins": sum(value < 0 for value in deltas), "ci95_delta_mse": ci})
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readout(path: Path, plan: Path, root: Path, rows: list[dict], summary: list[dict]) -> None:
    lines = ["# MambaTS v3 multidataset AutoCon screen", "", "Status: exploratory, validation-only, exact-coordinate paired screen; test metrics are not produced.", "", "| Dataset | lambda=0 mean MSE | lambda=0.001 mean MSE | relation delta % | wins/3 |", "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        null = [r["best_val_mse"] for r in rows if r["phase"] == PHASES[0] and r["dataset"] == dataset]
        candidate = [r["best_val_mse"] for r in rows if r["phase"] == PHASES[1] and r["dataset"] == dataset]
        item = next(value for value in summary if value["dataset"] == dataset)
        lines.append(f"| {dataset} | {statistics.mean(null):.6f} | {statistics.mean(candidate):.6f} | {item['mean_relative_change_pct']:.3f}% | {item['wins']}/{item['n_pairs']} |")
    lines.extend(["", "The only causal AutoCon comparison is lambda=0.001 versus lambda=0. Both arms retain the same MambaTSv3 AutoCon architecture, M features, MeanStd, seeds, budget, and test-withheld contract.", "", f"- Plan SHA256: `{_sha256(plan)}`", f"- Artifact root: `{root.resolve()}`", f"- Rows: `{len(rows)}`; test metrics produced: `false`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.plan.resolve(), args.artifact_root.resolve())
    pairs = paired(rows, PHASES[1], PHASES[0])
    summary = summarize(pairs)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "paired_deltas.csv", pairs)
    _write_csv(output / "paired_summary.csv", summary)
    _write_readout(output / "readout.md", args.plan.resolve(), args.artifact_root.resolve(), rows, summary)
    print(json.dumps({"rows": len(rows), "paired_rows": len(pairs), "summary_rows": len(summary), "test_metrics_produced": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
