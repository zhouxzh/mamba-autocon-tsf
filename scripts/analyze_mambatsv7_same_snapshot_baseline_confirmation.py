#!/usr/bin/env python3
"""Strict validation-only readout for the same-snapshot MambaTS-v7 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
SEEDS = (2021, 2022, 2023)
BASELINES = ("DLinear", "PatchTST", "TimesNet", "iTransformer")
ARMS = BASELINES + ("MambaTSv6AutoCon", "MambaTSv7AutoCon")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    return statistics.fmean(values)


def _read_phase(root, phase, expected_models):
    records = [_load_line(line) for line in (root / phase / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, hashes = [], set()
    expected = {(dataset, seed, model) for dataset in DATASETS for seed in SEEDS for model in expected_models}
    found = set()
    for record in records:
        coordinate = (record.get("dataset"), int(record.get("seed")), record.get("model"))
        if coordinate not in expected or coordinate in found or record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("invalid manifest record: {}".format(record.get("job_id")))
        found.add(coordinate)
        if int(record.get("pred_len")) != 96:
            raise ValueError("horizon mismatch: {}".format(record.get("job_id")))
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file() or any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("invalid result boundary: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        if config.get("model") != coordinate[2] or config.get("data") != coordinate[0] or config.get("seed") != coordinate[1] or config.get("skip_test") is not True:
            raise ValueError("config mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        logs = _load(checkpoint_root / result_dir.name / "loss_logs.json")
        losses = [float(value) for value in logs.get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("missing/non-finite validation losses: {}".format(result_dir))
        row = {"dataset": coordinate[0], "horizon": 96, "seed": coordinate[1], "model": coordinate[2], "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"])}
        if coordinate[2] == "MambaTSv7AutoCon":
            if not (result_dir / "acf_prior.npz").is_file():
                raise ValueError("v7 missing ACF sidecar: {}".format(result_dir))
            for key in ("acf_runtime_gate_mean", "acf_effective_memory_rms", "acf_effective_memory_to_forecast_rms"):
                values = [float(value) for value in logs.get(key, [])]
                if not values or not all(math.isfinite(value) for value in values):
                    raise ValueError("v7 missing/non-finite {}: {}".format(key, result_dir))
                row[key] = _mean(values)
        rows.append(row)
    if found != expected or len(hashes) != 1:
        raise ValueError("coverage/source mismatch in {}".format(phase))
    return rows, next(iter(hashes))


def _load_line(line):
    return json.loads(line)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows, hashes = [], []
    for phase, models in (("strong_baselines_h96", BASELINES), ("v6_control_h96", ("MambaTSv6AutoCon",)), ("v7_treatment_h96", ("MambaTSv7AutoCon",))):
        phase_rows, source_hash = _read_phase(root, phase, models); rows.extend(phase_rows); hashes.append(source_hash)
    if len(set(hashes)) != 1:
        raise ValueError("cross-phase source snapshot mismatch")
    keyed = {(r["dataset"], r["seed"], r["model"]): r for r in rows}
    summary = []
    for dataset in DATASETS:
        means = {model: _mean([keyed[(dataset, seed, model)]["best_val_mse"] for seed in SEEDS]) for model in ARMS}
        best_baseline = min(BASELINES, key=means.get)
        v7 = [keyed[(dataset, seed, "MambaTSv7AutoCon")] for seed in SEEDS]
        v6 = [keyed[(dataset, seed, "MambaTSv6AutoCon")] for seed in SEEDS]
        baseline = [keyed[(dataset, seed, best_baseline)] for seed in SEEDS]
        summary.append({"dataset": dataset, "horizon": 96, "best_baseline": best_baseline, "best_baseline_mean_val_mse": means[best_baseline], "v6_mean_val_mse": means["MambaTSv6AutoCon"], "v7_mean_val_mse": means["MambaTSv7AutoCon"], "v7_vs_best_baseline_percent": 100 * (means["MambaTSv7AutoCon"] / means[best_baseline] - 1), "v7_vs_v6_percent": 100 * (means["MambaTSv7AutoCon"] / means["MambaTSv6AutoCon"] - 1), "v7_seed_wins_vs_v6": sum(x["best_val_mse"] < y["best_val_mse"] for x, y in zip(v7, v6)), "v7_mean_elapsed_sec": _mean([x["elapsed_sec"] for x in v7]), "best_baseline_mean_elapsed_sec": _mean([x["elapsed_sec"] for x in baseline]), "v7_vs_best_baseline_elapsed_percent": 100 * (_mean([x["elapsed_sec"] for x in v7]) / _mean([x["elapsed_sec"] for x in baseline]) - 1), "v7_acf_gate_mean": _mean([x["acf_runtime_gate_mean"] for x in v7]), "v7_acf_memory_to_forecast_rms": _mean([x["acf_effective_memory_to_forecast_rms"] for x in v7])})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", rows); _write_csv(out / "summary.csv", summary)
    lines = ["# MambaTS v7 same-snapshot baseline confirmation", "", "Validation-only results. No test metric was read or produced.", "", "| Dataset | Strongest baseline | Baseline MSE | v7 MSE | v7 vs baseline | v7 vs v6 | v7 wall time vs baseline | ACF relative RMS |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {best_baseline} | {best_baseline_mean_val_mse:.6f} | {v7_mean_val_mse:.6f} | {v7_vs_best_baseline_percent:+.3f}% | {v7_vs_v6_percent:+.3f}% | {v7_vs_best_baseline_elapsed_percent:+.3f}% | {v7_acf_memory_to_forecast_rms:.4f} |".format(**row))
    lines.extend(["", "Positive v7-vs-baseline or v7-vs-v6 MSE percentages mean worse error. Source snapshot: `{}`.".format(hashes[0]), "", "Wall-clock job time is a protocol-specific secondary measure; separate device-resident resource measurements are required for an inference-speed claim."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": hashes[0], "test_metrics_produced": False})


if __name__ == "__main__":
    main()
