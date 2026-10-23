#!/usr/bin/env python3
"""Strict validation-only readout for the current MambaACF comparison."""

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
PHASES = (
    ("strong_baselines_h96", BASELINES, None),
    ("acf_memory_h96", ("MambaACF",), None),
    ("autocon_relation_null_h96", ("MambaACFAutoCon",), 0.0),
    ("autocon_relation_active_h96", ("MambaACFAutoCon",), 0.001),
)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


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
        if expected_lambda is None and actual_lambda is not None:
            raise ValueError("unexpected AutoCon lambda: {}".format(record.get("job_id")))
        if expected_lambda is not None and (actual_lambda is None or not math.isclose(float(actual_lambda), expected_lambda, abs_tol=1e-12)):
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
        row = {"dataset": coordinate[0], "horizon": 96, "seed": coordinate[1], "model": label, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"])}
        if coordinate[2].startswith("MambaACF"):
            prior = result_dir / "acf_prior.npz"
            if not prior.is_file():
                raise ValueError("MambaACF missing train-only sidecar: {}".format(result_dir))
            if expected_lambda is not None:
                for key in ("acf_effective_memory_rms", "acf_effective_memory_to_forecast_rms"):
                    values = [float(value) for value in logs.get(key, [])]
                    if not values or not all(math.isfinite(value) for value in values):
                        raise ValueError("missing/non-finite {}: {}".format(key, result_dir))
                    row[key] = _mean(values)
                values = [float(value) for value in logs.get("acf_relation_global_magnitude_ce", [])]
                if not values or not all(math.isfinite(value) for value in values):
                    raise ValueError("missing relation telemetry: {}".format(result_dir))
                row["relation_ce"] = _mean(values)
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
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows, hashes = [], []
    for phase, models, weight in PHASES:
        phase_rows, source_hash = _read_phase(root, phase, models, weight)
        rows.extend(phase_rows)
        hashes.append(source_hash)
    if len(set(hashes)) != 1:
        raise ValueError("cross-phase source snapshot mismatch")
    keyed = {(row["dataset"], row["seed"], row["model"]): row for row in rows}
    summary = []
    for dataset in DATASETS:
        means = {model: _mean([keyed[(dataset, seed, model)]["best_val_mse"] for seed in SEEDS]) for model in BASELINES + ("MambaACF", "MambaACFAutoCon_null", "MambaACFAutoCon_active")}
        best_baseline = min(BASELINES, key=means.get)
        memory = [keyed[(dataset, seed, "MambaACF")] for seed in SEEDS]
        null = [keyed[(dataset, seed, "MambaACFAutoCon_null")] for seed in SEEDS]
        active = [keyed[(dataset, seed, "MambaACFAutoCon_active")] for seed in SEEDS]
        baseline = [keyed[(dataset, seed, best_baseline)] for seed in SEEDS]
        summary.append({"dataset": dataset, "horizon": 96, "best_baseline": best_baseline, "best_baseline_mean_val_mse": means[best_baseline], "acf_memory_mean_val_mse": means["MambaACF"], "null_mean_val_mse": means["MambaACFAutoCon_null"], "active_mean_val_mse": means["MambaACFAutoCon_active"], "active_vs_baseline_percent": 100 * (means["MambaACFAutoCon_active"] / means[best_baseline] - 1), "active_vs_null_percent": 100 * (means["MambaACFAutoCon_active"] / means["MambaACFAutoCon_null"] - 1), "memory_vs_acf_off_unavailable": True, "active_seed_wins_vs_null": sum(a["best_val_mse"] < n["best_val_mse"] for a, n in zip(active, null)), "active_mean_elapsed_sec": _mean([r["elapsed_sec"] for r in active]), "baseline_mean_elapsed_sec": _mean([r["elapsed_sec"] for r in baseline]), "active_vs_baseline_elapsed_percent": 100 * (_mean([r["elapsed_sec"] for r in active]) / _mean([r["elapsed_sec"] for r in baseline]) - 1), "active_memory_rms": _mean([r["acf_effective_memory_to_forecast_rms"] for r in active]), "active_relation_ce": _mean([r["relation_ce"] for r in active])})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", rows); _write_csv(out / "summary.csv", summary)
    lines = ["# Current MambaACF H96 strong-baseline screen", "", "Validation-only results. No test metric was read or produced.", "", "| Dataset | Strongest baseline | Baseline MSE | ACF memory MSE | AutoCon null MSE | AutoCon active MSE | Active vs baseline | Active vs null | Active wins vs null |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {best_baseline} | {best_baseline_mean_val_mse:.6f} | {acf_memory_mean_val_mse:.6f} | {null_mean_val_mse:.6f} | {active_mean_val_mse:.6f} | {active_vs_baseline_percent:+.3f}% | {active_vs_null_percent:+.3f}% | {active_seed_wins_vs_null}/3 |".format(**row))
    lines.extend(["", "Positive MSE percentages mean worse error. Active-vs-null isolates the AutoCon relation effect; ACF memory is a separate architecture path.", "", "Source snapshot: `{}`. Resource speed claims require the separate device-resident benchmark.".format(hashes[0])])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": hashes[0], "test_metrics_produced": False})


if __name__ == "__main__":
    main()
