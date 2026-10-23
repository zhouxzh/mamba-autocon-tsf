#!/usr/bin/env python3
"""Strict reader for the frozen d24 MambaTSv3 high-dimensional H12 transfer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (2301, 2302, 2303)
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimesNet", "TimeMixer", "iTransformer", "Mamba2")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest(root: Path, directory: str) -> list[dict]:
    path = root / directory / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise ValueError("missing manifest: {}".format(path))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _checkpoint_root(record: dict, result_dir: Path) -> Path:
    command = record.get("command", [])
    if "--checkpoints" not in command:
        raise ValueError("checkpoint option missing: {}".format(record.get("job_id")))
    return Path(command[command.index("--checkpoints") + 1]) / result_dir.name


def _validate_record(record: dict) -> Path:
    if record.get("status") != "ok" or record.get("returncode") != 0:
        raise ValueError("failed job: {}".format(record.get("job_id")))
    before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
    if not before or before != after:
        raise ValueError("source drift: {}".format(record.get("job_id")))
    result_dir = Path(record.get("expected_artifact", "")).parent
    if not (result_dir / "training_complete.json").is_file():
        raise ValueError("missing completion marker: {}".format(result_dir))
    if any((result_dir / artifact).exists() for artifact in TEST_ARTIFACTS):
        raise ValueError("test artifact present: {}".format(result_dir))
    return result_dir


def _loss(record: dict, result_dir: Path) -> float:
    logs = _json(_checkpoint_root(record, result_dir) / "loss_logs.json")
    values = [float(value) for value in logs.get("val_loss", [])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid validation loss: {}".format(result_dir))
    return min(values)


def _common(config: dict, model: str, dataset: str, seed: int) -> None:
    channels = 321 if dataset == "electricity" else 862
    required = {
        "model": model, "data": dataset, "features": "M", "seq_len": 96,
        "pred_len": 12, "enc_in": channels, "dec_in": channels,
        "c_out": channels, "seed": seed, "skip_test": True,
        "d_model": 24, "d_ff": 48, "patch_len": 24, "patch_stride": 24,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("common config mismatch: {}".format(config.get("plan_job_id")))


def _read_baselines(root: Path) -> tuple[list[dict], set[str]]:
    expected = {(dataset, model, seed) for dataset in DATASETS for model in BASELINES for seed in SEEDS}
    seen, rows, hashes = set(), [], set()
    for record in _manifest(root, "baselines"):
        coordinate = (str(record.get("dataset")), str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError("invalid baseline coordinate: {}".format(coordinate))
        result_dir = _validate_record(record)
        config = _json(result_dir / "config.json")
        _common(config, coordinate[1], coordinate[0], coordinate[2])
        forbidden = {"AutoCon": False, "mamba_use_acf_memory": False, "mamba_acf_relation_loss": False}
        if any(config.get(key) != value for key, value in forbidden.items()):
            raise ValueError("candidate option leaked into baseline: {}".format(result_dir))
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        rows.append({"phase": "strong_baselines_h12", "dataset": coordinate[0], "model": coordinate[1], "seed": coordinate[2], "best_val_mse": _loss(record, result_dir), "best_epoch": record.get("best_epoch"), "epochs_ran": record.get("epochs_ran"), "elapsed_sec": record.get("elapsed_sec")})
    if seen != expected:
        raise ValueError("incomplete baseline coverage")
    return rows, hashes


def _read_candidate(root: Path, directory: str, phase: str, expected_lambda: float) -> tuple[list[dict], set[str]]:
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    seen, rows, hashes = set(), [], set()
    for record in _manifest(root, directory):
        coordinate = (str(record.get("dataset")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen or record.get("model") != "MambaTSv3AutoCon":
            raise ValueError("invalid candidate coordinate: {}".format(coordinate))
        result_dir = _validate_record(record)
        config = _json(result_dir / "config.json")
        _common(config, "MambaTSv3AutoCon", coordinate[0], coordinate[1])
        required = {"AutoCon": True, "AutoCon_lambda": expected_lambda, "AutoCon_wnorm": "MeanStd", "mamba_ts_variant": "v3", "mamba_causal_embedding": True}
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("candidate config mismatch: {}".format(result_dir))
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        rows.append({"phase": phase, "dataset": coordinate[0], "model": "MambaTSv3AutoCon", "seed": coordinate[1], "best_val_mse": _loss(record, result_dir), "best_epoch": record.get("best_epoch"), "epochs_ran": record.get("epochs_ran"), "elapsed_sec": record.get("elapsed_sec")})
    if seen != expected:
        raise ValueError("incomplete candidate coverage for {}".format(phase))
    return rows, hashes


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    baseline_rows, baseline_hashes = _read_baselines(root)
    null_rows, null_hashes = _read_candidate(root, "null", "d24_autocon_null_h12", 0.0)
    active_rows, active_hashes = _read_candidate(root, "active", "d24_autocon_active_h12", 0.001)
    all_hashes = baseline_hashes | null_hashes | active_hashes
    if len(all_hashes) != 1:
        raise ValueError("source snapshot differs across phases")
    rows = baseline_rows + null_rows + active_rows
    summary = []
    for dataset in DATASETS:
        active = [row["best_val_mse"] for row in active_rows if row["dataset"] == dataset]
        null = [row["best_val_mse"] for row in null_rows if row["dataset"] == dataset]
        baseline_means = {model: statistics.fmean(row["best_val_mse"] for row in baseline_rows if row["dataset"] == dataset and row["model"] == model) for model in BASELINES}
        strongest = min(baseline_means, key=baseline_means.get)
        active_mean, null_mean = statistics.fmean(active), statistics.fmean(null)
        summary.append({
            "dataset": dataset, "horizon": 12, "active_mean_val_mse": active_mean,
            "null_mean_val_mse": null_mean,
            "active_vs_null_percent": 100.0 * (active_mean / null_mean - 1.0),
            "active_wins_vs_null": sum(a < b for a, b in zip(active, null)),
            "strongest_baseline": strongest, "strongest_baseline_mean_val_mse": baseline_means[strongest],
            "active_vs_strongest_percent": 100.0 * (active_mean / baseline_means[strongest] - 1.0),
            "passes_external_gate": active_mean <= baseline_means[strongest] * 1.02,
            "passes_autocon_noninferiority": active_mean <= null_mean * 1.01,
            "passes_autocon_improvement": active_mean < null_mean and sum(a < b for a, b in zip(active, null)) >= 2,
            "source_snapshot": next(iter(all_hashes)), "test_metrics_produced": False,
        })
    transfer_pass = all(row["passes_external_gate"] and row["passes_autocon_noninferiority"] for row in summary)
    for row in summary:
        row["passes_highdim_transfer"] = transfer_pass
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "summary.csv", summary)
    lines = ["# D24 MambaTSv3 High-Dimensional H12 Transfer Readout", "", "Validation-only. No held-out test metric was generated or read.", "", "| Dataset | Active MSE | Null MSE | Active vs null | wins/null | Strongest baseline | Active vs baseline | External gate | Transfer gate |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |"]
    for row in summary:
        lines.append("| {dataset} | {active_mean_val_mse:.6f} | {null_mean_val_mse:.6f} | {active_vs_null_percent:+.3f}% | {active_wins_vs_null}/3 | {strongest_baseline} ({strongest_baseline_mean_val_mse:.6f}) | {active_vs_strongest_percent:+.3f}% | {passes_external_gate} | {passes_highdim_transfer} |".format(**row))
    lines.extend(["", "Source snapshot: `{}`.".format(summary[0]["source_snapshot"])])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passes_highdim_transfer": transfer_pass, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
