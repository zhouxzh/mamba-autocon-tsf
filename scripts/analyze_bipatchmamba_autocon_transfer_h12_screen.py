#!/usr/bin/env python3
"""Strict reader for the BiPatchMamba AutoCon H12 transfer screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "electricity", "traffic", "weather")
SEEDS = (2401, 2402)
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest(root: Path, directory: str) -> list[dict]:
    path = root / directory / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise ValueError("missing manifest: {}".format(path))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_record(record: dict) -> Path:
    if record.get("status") != "ok" or record.get("returncode") != 0:
        raise ValueError("failed job: {}".format(record.get("job_id")))
    before = record.get("source_tree_sha256_before")
    after = record.get("source_tree_sha256_after")
    if not before or before != after:
        raise ValueError("source drift: {}".format(record.get("job_id")))
    expected = Path(record.get("expected_artifact", ""))
    result_dir = expected.parent
    if not (result_dir / "training_complete.json").is_file():
        raise ValueError("missing completion marker: {}".format(result_dir))
    if any((result_dir / artifact).exists() for artifact in TEST_ARTIFACTS):
        raise ValueError("test artifact present: {}".format(result_dir))
    command = record.get("command", [])
    if "--checkpoints" not in command:
        raise ValueError("checkpoint option missing: {}".format(record.get("job_id")))
    checkpoint_root = Path(command[command.index("--checkpoints") + 1])
    loss_path = checkpoint_root / result_dir.name / "loss_logs.json"
    if not loss_path.is_file():
        raise ValueError("missing loss logs: {}".format(loss_path))
    return result_dir


def _loss(record: dict, result_dir: Path) -> float:
    command = record["command"]
    loss_path = Path(command[command.index("--checkpoints") + 1]) / result_dir.name / "loss_logs.json"
    values = [float(value) for value in _json(loss_path).get("val_loss", [])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid validation loss: {}".format(loss_path))
    return min(values)


def _channels(dataset: str) -> int:
    return {"ETTh1": 7, "electricity": 321, "traffic": 862, "weather": 21}[dataset]


def _common(config: dict, model: str, dataset: str, seed: int) -> None:
    channels = _channels(dataset)
    required = {
        "model": model, "data": dataset, "features": "M", "seq_len": 96,
        "pred_len": 12, "enc_in": channels, "dec_in": channels,
        "c_out": channels, "seed": seed, "skip_test": True,
        "d_model": 24, "d_ff": 48, "patch_len": 24, "patch_stride": 24,
    }
    mismatches = [key for key, value in required.items() if config.get(key) != value]
    if mismatches:
        raise ValueError("common config mismatch {}: {}".format(mismatches, config.get("plan_job_id")))


def _read_phase(root: Path, directory: str, expected_models: set[str], phase: str) -> tuple[list[dict], set[str]]:
    expected = {(dataset, model, seed) for dataset in DATASETS for model in expected_models for seed in SEEDS}
    seen, rows, hashes = set(), [], set()
    for record in _manifest(root, directory):
        coordinate = (str(record.get("dataset")), str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError("invalid or duplicate coordinate: {}".format(coordinate))
        result_dir = _validate_record(record)
        config = _json(result_dir / "config.json")
        _common(config, coordinate[1], coordinate[0], coordinate[2])
        if phase == "strong_baselines_h12":
            if config.get("AutoCon") is not False or config.get("mamba_acf_relation_loss") is not False:
                raise ValueError("candidate option leaked into baseline: {}".format(result_dir))
        elif phase == "bipatch_plain_h12":
            if config.get("AutoCon") is not False:
                raise ValueError("plain phase unexpectedly enables AutoCon: {}".format(result_dir))
        else:
            if config.get("AutoCon") is not True or config.get("AutoCon_wnorm") != "MeanStd":
                raise ValueError("AutoCon config mismatch: {}".format(result_dir))
            expected_lambda = 0.0 if phase == "bipatch_autocon_null_h12" else 0.001
            if config.get("AutoCon_lambda") != expected_lambda:
                raise ValueError("lambda mismatch: {}".format(result_dir))
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        rows.append({
            "phase": phase, "dataset": coordinate[0], "model": coordinate[1], "seed": coordinate[2],
            "best_val_mse": _loss(record, result_dir), "best_epoch": record.get("best_epoch"),
            "epochs_ran": record.get("epochs_ran"), "elapsed_sec": record.get("elapsed_sec"),
        })
    if seen != expected:
        raise ValueError("incomplete coverage for {}: expected {} got {}".format(phase, len(expected), len(seen)))
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
    baseline_rows, baseline_hashes = _read_phase(root, "baselines", set(BASELINES), "strong_baselines_h12")
    plain_rows, plain_hashes = _read_phase(root, "plain", {"BiPatchMamba"}, "bipatch_plain_h12")
    null_rows, null_hashes = _read_phase(root, "null", {"BiPatchMambaAutoCon"}, "bipatch_autocon_null_h12")
    active_rows, active_hashes = _read_phase(root, "active", {"BiPatchMambaAutoCon"}, "bipatch_autocon_active_h12")
    hashes = baseline_hashes | plain_hashes | null_hashes | active_hashes
    if len(hashes) != 1:
        raise ValueError("source snapshot differs across phases")

    rows = baseline_rows + plain_rows + null_rows + active_rows
    summary = []
    for dataset in DATASETS:
        def values(phase: str, model: str | None = None) -> list[float]:
            return [row["best_val_mse"] for row in rows if row["phase"] == phase and row["dataset"] == dataset and (model is None or row["model"] == model)]
        baseline_means = {model: statistics.fmean(values("strong_baselines_h12", model)) for model in BASELINES}
        strongest = min(baseline_means, key=baseline_means.get)
        active, null, plain = values("bipatch_autocon_active_h12"), values("bipatch_autocon_null_h12"), values("bipatch_plain_h12")
        active_mean, null_mean, plain_mean = statistics.fmean(active), statistics.fmean(null), statistics.fmean(plain)
        summary.append({
            "dataset": dataset, "horizon": 12, "active_mean_val_mse": active_mean,
            "null_mean_val_mse": null_mean, "plain_mean_val_mse": plain_mean,
            "active_vs_null_percent": 100.0 * (active_mean / null_mean - 1.0),
            "active_vs_plain_percent": 100.0 * (active_mean / plain_mean - 1.0),
            "active_wins_vs_null": sum(a < b for a, b in zip(active, null)),
            "active_wins_vs_plain": sum(a < b for a, b in zip(active, plain)),
            "strongest_baseline": strongest, "strongest_baseline_mean_val_mse": baseline_means[strongest],
            "active_vs_strongest_percent": 100.0 * (active_mean / baseline_means[strongest] - 1.0),
            "passes_external_gate": active_mean <= baseline_means[strongest] * 1.02,
            "passes_autocon_noninferiority": active_mean <= null_mean * 1.01,
            "passes_autocon_improvement": active_mean < null_mean and sum(a < b for a, b in zip(active, null)) >= 2,
            "source_snapshot": next(iter(hashes)), "test_metrics_produced": False,
        })
    transfer_pass = all(row["passes_external_gate"] and row["passes_autocon_noninferiority"] for row in summary)
    for row in summary:
        row["passes_transfer"] = transfer_pass
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "summary.csv", summary)
    lines = ["# BiPatchMamba AutoCon H12 Transfer Readout", "", "Validation-only. No held-out test metric was generated or read.", "", "| Dataset | Active | Null | Plain | Active vs null | wins/null | Strongest baseline | Active vs baseline | External gate | Transfer gate |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |"]
    for row in summary:
        lines.append("| {dataset} | {active_mean_val_mse:.6f} | {null_mean_val_mse:.6f} | {plain_mean_val_mse:.6f} | {active_vs_null_percent:+.3f}% | {active_wins_vs_null}/2 | {strongest_baseline} ({strongest_baseline_mean_val_mse:.6f}) | {active_vs_strongest_percent:+.3f}% | {passes_external_gate} | {passes_transfer} |".format(**row))
    lines.extend(["", "Source snapshot: `{}`.".format(summary[0]["source_snapshot"])])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passes_transfer": transfer_pass, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
