#!/usr/bin/env python3
"""Strict validation-only readout for the MambaTS v7 ACF/linear-base screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
SEEDS = (2021, 2022, 2023)
ARMS = {"v6": "MambaTSv6AutoCon", "v7": "MambaTSv7AutoCon"}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values):
    return statistics.fmean(values) if values else None


def _series(logs, key: str, result_dir: Path, allow_none: bool = False):
    values = logs.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError("missing {}: {}".format(key, result_dir))
    parsed = []
    for value in values:
        if value is None and allow_none:
            continue
        if value is None:
            raise ValueError("null {}: {}".format(key, result_dir))
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite {}: {}".format(key, result_dir))
        parsed.append(value)
    return parsed


def _check_config(config, record, arm: str, result_dir: Path):
    variant = "v6" if arm == "MambaTSv6AutoCon" else "v7"
    expected = {
        "model": arm, "data": record["dataset"], "seed": record["seed"],
        "features": "M", "pred_len": 96, "AutoCon": True, "AutoCon_lambda": 0.001,
        "AutoCon_wnorm": "MeanStd", "skip_test": True, "mamba_ts_variant": variant,
        "mamba_acf_relation_loss": False, "mamba_causal_embedding": True,
    }
    if variant == "v6":
        expected.update({"mamba_use_acf_memory": False, "mamba_ts_acf_memory_selector": False})
    else:
        expected.update({"mamba_use_acf_memory": True, "mamba_ts_acf_memory_selector": True})
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError("{} mismatch for {}: {!r} != {!r}".format(key, result_dir, config.get(key), wanted))
    if config.get("mamba_acf_lags") != [24, 48]:
        raise ValueError("unexpected ACF lags: {}".format(result_dir))


def _read(root: Path):
    expected = {(dataset, seed, arm) for dataset in DATASETS for seed in SEEDS for arm in ARMS.values()}
    rows, found, source_hashes = [], set(), set()
    for phase, arm in (("v6_linear_control_h96", ARMS["v6"]), ("v7_acf_selector_h96", ARMS["v7"])):
        manifest = root / phase / "logs" / "job_manifest.jsonl"
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for record in records:
            if record.get("status") != "ok" or record.get("returncode") != 0:
                raise ValueError("failed record: {}".format(record.get("job_id")))
            dataset, seed, model = str(record.get("dataset")), int(record.get("seed")), str(record.get("model"))
            coordinate = (dataset, seed, model)
            if coordinate not in expected or coordinate in found or model != arm:
                raise ValueError("invalid coordinate: {}".format(coordinate))
            found.add(coordinate)
            if int(record.get("pred_len")) != 96 or not math.isclose(float(record.get("autocon_lambda")), .001):
                raise ValueError("horizon/lambda mismatch: {}".format(record.get("job_id")))
            before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
            if not before or before != after:
                raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
            source_hashes.add(before)
            artifact = Path(record["expected_artifact"])
            if artifact.name != "training_complete.json" or not artifact.is_file():
                raise ValueError("missing training completion artifact: {}".format(record.get("job_id")))
            result_dir = artifact.parent
            if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
                raise ValueError("test artifact present: {}".format(result_dir))
            config = _load(result_dir / "config.json")
            _check_config(config, record, arm, result_dir)
            sidecar = result_dir / "acf_prior.npz"
            if (arm == ARMS["v7"]) != sidecar.is_file():
                raise ValueError("unexpected train-only ACF sidecar state: {}".format(result_dir))
            command = record.get("command", [])
            checkpoint_root = Path(command[command.index("--checkpoints") + 1])
            logs = _load(checkpoint_root / result_dir.name / "loss_logs.json")
            val_loss = _series(logs, "val_loss", result_dir)
            telemetry = {key: _mean(_series(logs, key, result_dir, allow_none=True)) for key in (
                "acf_runtime_gate_mean", "acf_effective_memory_rms", "acf_effective_memory_to_forecast_rms")}
            rows.append({
                "dataset": dataset, "horizon": 96, "seed": seed, "arm": arm,
                "best_val_mse": min(val_loss), "elapsed_sec": float(record["elapsed_sec"]),
                "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"]),
                "acf_runtime_gate_mean": telemetry["acf_runtime_gate_mean"],
                "acf_effective_memory_rms": telemetry["acf_effective_memory_rms"],
                "acf_effective_memory_to_forecast_rms": telemetry["acf_effective_memory_to_forecast_rms"],
                "acf_prior_sha256": _sha256(sidecar) if sidecar.is_file() else "",
                "result_dir": str(result_dir),
            })
    if found != expected or len(rows) != len(expected) or len(source_hashes) != 1:
        raise ValueError("coverage/source mismatch")
    return rows, next(iter(source_hashes))


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, source_hash = _read(args.root.resolve())
    keyed = {(r["dataset"], r["seed"], r["arm"]): r for r in rows}
    summary, paired = [], []
    for dataset in DATASETS:
        control = [keyed[(dataset, seed, ARMS["v6"])] for seed in SEEDS]
        candidate = [keyed[(dataset, seed, ARMS["v7"])] for seed in SEEDS]
        deltas = []
        for baseline, treatment in zip(control, candidate):
            delta = treatment["best_val_mse"] - baseline["best_val_mse"]
            relative = 100.0 * delta / baseline["best_val_mse"]
            deltas.append(relative)
            paired.append({"dataset": dataset, "horizon": 96, "seed": baseline["seed"], "v6_val_mse": baseline["best_val_mse"], "v7_val_mse": treatment["best_val_mse"], "v7_minus_v6_mse": delta, "relative_percent": relative})
        v6_time, v7_time = _mean([r["elapsed_sec"] for r in control]), _mean([r["elapsed_sec"] for r in candidate])
        summary.append({
            "dataset": dataset, "horizon": 96, "v6_mean_val_mse": _mean([r["best_val_mse"] for r in control]), "v7_mean_val_mse": _mean([r["best_val_mse"] for r in candidate]), "v7_vs_v6_relative_percent": _mean(deltas), "v7_seed_improvements": sum(delta < 0 for delta in deltas), "v6_mean_elapsed_sec": v6_time, "v7_mean_elapsed_sec": v7_time, "v7_vs_v6_elapsed_percent": 100.0 * (v7_time / v6_time - 1.0), "v7_acf_runtime_gate_mean": _mean([r["acf_runtime_gate_mean"] for r in candidate if r["acf_runtime_gate_mean"] is not None]), "v7_acf_effective_memory_rms": _mean([r["acf_effective_memory_rms"] for r in candidate if r["acf_effective_memory_rms"] is not None]),
        })
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows); _write_csv(output / "paired_summary.csv", paired); _write_csv(output / "summary.csv", summary)
    lines = ["# MambaTS v7 ACF/linear-base screen", "", "Validation-only paired readout. No test metric was read or produced.", "", "| Dataset | v6 MSE | v7 MSE | v7 vs v6 | v7 wins | v7 wall time change | ACF telemetry |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        telemetry = "present" if row["v7_acf_runtime_gate_mean"] is not None else "missing (None)"
        lines.append("| {dataset} | {v6_mean_val_mse:.6f} | {v7_mean_val_mse:.6f} | {v7_vs_v6_relative_percent:+.3f}% | {v7_seed_improvements}/3 | {v7_vs_v6_elapsed_percent:+.3f}% | {telemetry} |".format(telemetry=telemetry, **row))
    lines.extend(["", "Positive v7-vs-v6 percentage means worse validation MSE. Source snapshot: `{}`.".format(source_hash), "", "ACF telemetry is reported only when finite non-null values are logged; a missing value is not evidence that the branch was effective."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": source_hash, "test_metrics_produced": False})


if __name__ == "__main__":
    main()
