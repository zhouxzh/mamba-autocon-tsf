#!/usr/bin/env python3
"""Strict validation-only reader for the frozen MambaTS-v7 long-horizon run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
HORIZONS = (336, 720)
SEEDS = (2024, 2025, 2026)
BASELINES = ("DLinear", "PatchTST", "TimesNet", "iTransformer")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    return statistics.fmean(values)


def _read(root, phase, expected_models):
    manifest = root / phase / "logs" / "job_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(dataset, horizon, seed, model) for dataset in DATASETS for horizon in HORIZONS for seed in SEEDS for model in expected_models}
    found, rows, source_hashes = set(), [], set()
    for record in records:
        coordinate = (str(record.get("dataset")), int(record.get("pred_len")), int(record.get("seed")), str(record.get("model")))
        if coordinate not in expected or coordinate in found:
            raise ValueError("invalid or duplicate coordinate: {}".format(coordinate))
        if record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("failed task: {}".format(record.get("job_id")))
        found.add(coordinate)
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        source_hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file() or any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("result/test boundary failure: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        if (config.get("model"), config.get("data"), int(config.get("pred_len")), int(config.get("seed")), config.get("skip_test")) != (coordinate[3], coordinate[0], coordinate[1], coordinate[2], True):
            raise ValueError("configuration mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        logs = _load(checkpoint_root / result_dir.name / "loss_logs.json")
        val_loss = [float(value) for value in logs.get("val_loss", [])]
        if not val_loss or not all(math.isfinite(value) for value in val_loss):
            raise ValueError("missing validation metrics: {}".format(result_dir))
        row = {"dataset": coordinate[0], "horizon": coordinate[1], "seed": coordinate[2], "model": coordinate[3], "best_val_mse": min(val_loss), "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"])}
        if coordinate[3] == "MambaTSv7AutoCon":
            if not (result_dir / "acf_prior.npz").is_file():
                raise ValueError("v7 ACF sidecar missing: {}".format(result_dir))
            for key in ("acf_runtime_gate_mean", "acf_effective_memory_rms", "acf_effective_memory_to_forecast_rms"):
                values = [float(value) for value in logs.get(key, [])]
                if not values or not all(math.isfinite(value) for value in values):
                    raise ValueError("v7 telemetry missing/non-finite {}: {}".format(key, result_dir))
                row[key] = _mean(values)
        rows.append(row)
    if found != expected or len(source_hashes) != 1:
        raise ValueError("coverage/source mismatch in {}".format(phase))
    return rows, next(iter(source_hashes))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    baseline_rows, baseline_hash = _read(root, "strong_baselines_fresh_long", BASELINES)
    v7_rows, v7_hash = _read(root, "v7_fresh_long", ("MambaTSv7AutoCon",))
    if baseline_hash != v7_hash:
        raise ValueError("cross-phase source snapshot mismatch")
    rows = baseline_rows + v7_rows
    keyed = {(row["dataset"], row["horizon"], row["seed"], row["model"]): row for row in rows}
    summary = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            means = {model: _mean([keyed[(dataset, horizon, seed, model)]["best_val_mse"] for seed in SEEDS]) for model in BASELINES + ("MambaTSv7AutoCon",)}
            best = min(BASELINES, key=means.get)
            v7 = [keyed[(dataset, horizon, seed, "MambaTSv7AutoCon")] for seed in SEEDS]
            reference = [keyed[(dataset, horizon, seed, best)] for seed in SEEDS]
            summary.append({"dataset": dataset, "horizon": horizon, "best_baseline": best, "best_baseline_mean_val_mse": means[best], "v7_mean_val_mse": means["MambaTSv7AutoCon"], "v7_vs_best_baseline_percent": 100 * (means["MambaTSv7AutoCon"] / means[best] - 1), "v7_seed_wins_vs_best_baseline": sum(x["best_val_mse"] < y["best_val_mse"] for x, y in zip(v7, reference)), "v7_mean_elapsed_sec": _mean([x["elapsed_sec"] for x in v7]), "best_baseline_mean_elapsed_sec": _mean([x["elapsed_sec"] for x in reference]), "v7_vs_best_baseline_elapsed_percent": 100 * (_mean([x["elapsed_sec"] for x in v7]) / _mean([x["elapsed_sec"] for x in reference]) - 1), "v7_acf_gate_mean": _mean([x["acf_runtime_gate_mean"] for x in v7]), "v7_acf_memory_to_forecast_rms": _mean([x["acf_effective_memory_to_forecast_rms"] for x in v7])})
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows); _write_csv(output / "summary.csv", summary)
    lines = ["# Frozen MambaTS v7 fresh-seed long-horizon confirmation", "", "Validation-only results. No test metric was read or produced.", "", "| Dataset | Horizon | Strongest baseline | Baseline MSE | v7 MSE | v7 vs baseline | v7 wins | v7 wall time change | ACF relative RMS |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {horizon} | {best_baseline} | {best_baseline_mean_val_mse:.6f} | {v7_mean_val_mse:.6f} | {v7_vs_best_baseline_percent:+.3f}% | {v7_seed_wins_vs_best_baseline}/3 | {v7_vs_best_baseline_elapsed_percent:+.3f}% | {v7_acf_memory_to_forecast_rms:.4f} |".format(**row))
    lines.extend(["", "Positive v7-vs-baseline percentages mean worse validation MSE. Source snapshot: `{}`.".format(baseline_hash), "", "This fixed-architecture fresh-seed screen is not a test release and does not itself prove an inference-speed advantage."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": baseline_hash, "test_metrics_produced": False})


if __name__ == "__main__":
    main()
