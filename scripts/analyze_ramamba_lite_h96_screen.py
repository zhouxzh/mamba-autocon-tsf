#!/usr/bin/env python3
"""Strict validation-only readout for the RaMambaLite H96 exploratory matrix."""

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
ARMS = BASELINES + ("RaMambaLite", "RaMambaLiteAutoCon_null", "RaMambaLiteAutoCon_active")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")
PHASES = (
    ("strong_baselines_h96", BASELINES, None),
    ("ramamba_plain_h96", ("RaMambaLite",), None),
    ("ramamba_autocon_null_h96", ("RaMambaLiteAutoCon",), 0.0),
    ("ramamba_autocon_active_h96", ("RaMambaLiteAutoCon",), 0.001),
)


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    return statistics.fmean(values)


def _read_phase(root, phase, expected_models, expected_lambda):
    manifest = root / phase / "logs" / "job_manifest.jsonl"
    records = [_load_line(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(dataset, seed, model) for dataset in DATASETS for seed in SEEDS for model in expected_models}
    found, rows, hashes = set(), [], set()
    for record in records:
        coordinate = (record.get("dataset"), int(record.get("seed")), record.get("model"))
        if coordinate not in expected or coordinate in found or record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("invalid manifest record: {}".format(record.get("job_id")))
        if int(record.get("pred_len")) != 96:
            raise ValueError("horizon mismatch: {}".format(record.get("job_id")))
        actual_lambda = record.get("autocon_lambda")
        if expected_lambda is None:
            if actual_lambda is not None:
                raise ValueError("unexpected AutoCon lambda: {}".format(record.get("job_id")))
        elif actual_lambda is None or not math.isclose(float(actual_lambda), expected_lambda, abs_tol=1e-12):
            raise ValueError("AutoCon lambda mismatch: {}".format(record.get("job_id")))
        found.add(coordinate)
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file() or any((result_dir / item).exists() for item in TEST_ARTIFACTS):
            raise ValueError("invalid result boundary: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        if config.get("model") != coordinate[2] or config.get("data") != coordinate[0] or config.get("seed") != coordinate[1] or config.get("skip_test") is not True:
            raise ValueError("config mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        logs = _load(checkpoint_root / result_dir.name / "loss_logs.json")
        losses = [float(value) for value in logs.get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("missing/non-finite validation losses: {}".format(result_dir))
        label = coordinate[2]
        if expected_lambda == 0.0:
            label += "_null"
        elif expected_lambda is not None:
            label += "_active"
        rows.append({
            "dataset": coordinate[0], "horizon": 96, "seed": coordinate[1], "model": label,
            "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]),
            "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"]),
        })
    if found != expected or len(hashes) != 1:
        raise ValueError("coverage/source mismatch in {}".format(phase))
    return rows, next(iter(hashes))


def _load_line(line):
    return json.loads(line)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, hashes = [], []
    root = args.root.resolve()
    for phase, models, weight in PHASES:
        phase_rows, source_hash = _read_phase(root, phase, models, weight)
        rows.extend(phase_rows)
        hashes.append(source_hash)
    if len(set(hashes)) != 1:
        raise ValueError("cross-phase source snapshot mismatch")
    keyed = {(row["dataset"], row["seed"], row["model"]): row for row in rows}
    summary = []
    for dataset in DATASETS:
        means = {model: _mean([keyed[(dataset, seed, model)]["best_val_mse"] for seed in SEEDS]) for model in ARMS}
        best_baseline = min(BASELINES, key=means.get)
        baseline_rows = [keyed[(dataset, seed, best_baseline)] for seed in SEEDS]
        active_rows = [keyed[(dataset, seed, "RaMambaLiteAutoCon_active")] for seed in SEEDS]
        null_rows = [keyed[(dataset, seed, "RaMambaLiteAutoCon_null")] for seed in SEEDS]
        summary.append({
            "dataset": dataset, "horizon": 96, "best_baseline": best_baseline,
            "best_baseline_mean_val_mse": means[best_baseline],
            "ramamba_plain_mean_val_mse": means["RaMambaLite"],
            "ramamba_null_mean_val_mse": means["RaMambaLiteAutoCon_null"],
            "ramamba_active_mean_val_mse": means["RaMambaLiteAutoCon_active"],
            "active_vs_best_baseline_percent": 100 * (means["RaMambaLiteAutoCon_active"] / means[best_baseline] - 1),
            "active_vs_plain_percent": 100 * (means["RaMambaLiteAutoCon_active"] / means["RaMambaLite"] - 1),
            "active_vs_null_percent": 100 * (means["RaMambaLiteAutoCon_active"] / means["RaMambaLiteAutoCon_null"] - 1),
            "active_seed_wins_vs_null": sum(active["best_val_mse"] < null["best_val_mse"] for active, null in zip(active_rows, null_rows)),
            "active_mean_elapsed_sec": _mean([row["elapsed_sec"] for row in active_rows]),
            "best_baseline_mean_elapsed_sec": _mean([row["elapsed_sec"] for row in baseline_rows]),
            "active_vs_best_baseline_elapsed_percent": 100 * (_mean([row["elapsed_sec"] for row in active_rows]) / _mean([row["elapsed_sec"] for row in baseline_rows]) - 1),
        })
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", rows)
    _write_csv(out / "summary.csv", summary)
    lines = [
        "# RaMambaLite H96 exploratory screen", "",
        "Validation-only results. No test metric was read or produced.", "",
        "| Dataset | Strongest baseline | Baseline MSE | Plain MSE | Null MSE | Active MSE | Active vs baseline | Active vs null | Active seed wins vs null |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append("| {dataset} | {best_baseline} | {best_baseline_mean_val_mse:.6f} | {ramamba_plain_mean_val_mse:.6f} | {ramamba_null_mean_val_mse:.6f} | {ramamba_active_mean_val_mse:.6f} | {active_vs_best_baseline_percent:+.3f}% | {active_vs_null_percent:+.3f}% | {active_seed_wins_vs_null}/3 |".format(**row))
    lines.extend([
        "",
        "Positive MSE percentages mean worse error. The null arm is the same AutoCon-capable architecture with zero auxiliary-loss weight; active-vs-null is the registered AutoCon effect contrast.",
        "",
        "Source snapshot: `{}`. Wall-clock job time is protocol-specific; separate device-resident measurements are required for an inference-speed claim.".format(hashes[0]),
    ])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": hashes[0], "test_metrics_produced": False})


if __name__ == "__main__":
    main()
