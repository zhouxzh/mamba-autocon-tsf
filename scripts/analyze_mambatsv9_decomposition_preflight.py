#!/usr/bin/env python3
"""Strict reader for the frozen MambaTS v9 ETTh1/H336 preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = {
    "null": ("v9_autocon_null", "MambaTSv9AutoCon", 0.0),
    "active": ("v9_autocon_active", "MambaTSv9AutoCon", 0.001),
    "patchtst": ("patchtst_baseline", "PatchTST", None),
}
SEEDS = (2191, 2192, 2193)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    return statistics.fmean(values)


def _read_phase(root, key):
    phase, model, lambda_value = PHASES[key]
    manifest = root / key / "logs" / "job_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(seed, model) for seed in SEEDS}
    found, rows, source_hashes = set(), [], set()
    for record in records:
        coordinate = (int(record.get("seed")), str(record.get("model")))
        if coordinate not in expected or coordinate in found:
            raise ValueError("invalid or duplicate coordinate: {}".format(coordinate))
        if record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("failed job: {}".format(record.get("job_id")))
        found.add(coordinate)
        before = record.get("source_tree_sha256_before")
        after = record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        source_hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file():
            raise ValueError("missing training completion: {}".format(result_dir))
        if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("test artifact present: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        if config.get("model") != model or config.get("data") != "ETTh1" or int(config.get("pred_len")) != 336:
            raise ValueError("configuration mismatch: {}".format(result_dir))
        if config.get("skip_test") is not True or int(config.get("seed")) != coordinate[0]:
            raise ValueError("test/seed contract mismatch: {}".format(result_dir))
        if key != "patchtst":
            if config.get("mamba_ts_variant") != "v9" or bool(config.get("AutoCon")) is not True:
                raise ValueError("v9 contract mismatch: {}".format(result_dir))
            if not math.isclose(float(config.get("AutoCon_lambda")), lambda_value, abs_tol=1e-12):
                raise ValueError("AutoCon lambda mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        logs = _load(checkpoint_root / result_dir.name / "loss_logs.json")
        values = [float(value) for value in logs.get("val_loss", [])]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("missing/non-finite validation losses: {}".format(result_dir))
        rows.append({
            "phase": phase,
            "model": model,
            "seed": coordinate[0],
            "best_val_mse": min(values),
            "epochs_ran": int(record["epochs_ran"]),
            "best_epoch": int(record["best_epoch"]),
            "elapsed_sec": float(record["elapsed_sec"]),
        })
    if found != expected or len(source_hashes) != 1:
        raise ValueError("coverage/source mismatch for phase {}".format(key))
    return rows, next(iter(source_hashes))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    phase_rows, hashes = {}, set()
    for key in PHASES:
        rows, source_hash = _read_phase(root, key)
        phase_rows[key] = rows
        hashes.add(source_hash)
    if len(hashes) != 1:
        raise ValueError("cross-phase source snapshot mismatch")
    keyed = {key: {row["seed"]: row for row in rows} for key, rows in phase_rows.items()}
    null_values = [keyed["null"][seed]["best_val_mse"] for seed in SEEDS]
    active_values = [keyed["active"][seed]["best_val_mse"] for seed in SEEDS]
    patch_values = [keyed["patchtst"][seed]["best_val_mse"] for seed in SEEDS]
    active_mean, null_mean, patch_mean = map(_mean, (active_values, null_values, patch_values))
    active_wins_null = sum(active < null for active, null in zip(active_values, null_values))
    active_wins_patch = sum(active < patch for active, patch in zip(active_values, patch_values))
    summary = [{
        "dataset": "ETTh1", "horizon": 336,
        "v9_null_mean_val_mse": null_mean,
        "v9_active_mean_val_mse": active_mean,
        "patchtst_mean_val_mse": patch_mean,
        "active_vs_null_percent": 100 * (active_mean / null_mean - 1),
        "active_vs_patchtst_percent": 100 * (active_mean / patch_mean - 1),
        "active_wins_vs_null": active_wins_null,
        "active_wins_vs_patchtst": active_wins_patch,
        "passes_autocon_gate": active_wins_null >= 2 and active_mean < null_mean,
        "passes_accuracy_gate": active_mean <= patch_mean,
        "source_snapshot": next(iter(hashes)),
        "test_metrics_produced": False,
    }]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", sum(phase_rows.values(), []))
    _write_csv(output / "summary.csv", summary)
    row = summary[0]
    lines = [
        "# MambaTS v9 Decomposition ETTh1 H336 Preflight",
        "",
        "Validation-only results. No test metric was generated or read.",
        "",
        "| v9 null MSE | v9 active MSE | PatchTST MSE | active vs null | active wins/null | active vs PatchTST | active wins/PatchTST | AutoCon gate | accuracy gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        "| {v9_null_mean_val_mse:.6f} | {v9_active_mean_val_mse:.6f} | {patchtst_mean_val_mse:.6f} | {active_vs_null_percent:+.3f}% | {active_wins_vs_null}/3 | {active_vs_patchtst_percent:+.3f}% | {active_wins_vs_patchtst}/3 | {passes_autocon_gate} | {passes_accuracy_gate} |".format(**row),
        "",
        "The pre-registered continuation rule requires both gates. Source snapshot: `{}`.".format(row["source_snapshot"]),
    ]
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary[0], sort_keys=True))


if __name__ == "__main__":
    main()
