#!/usr/bin/env python3
"""Strict reader for the independent METR-LA / PEMS-BAY MambaTSv3 screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("metr_la", "pems_bay")
SEEDS = (2501, 2502)
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3")
CHANNELS = {"metr_la": 207, "pems_bay": 325}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest(root: Path, directory: str) -> list[dict]:
    path = root / directory / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validated(record: dict) -> tuple[Path, dict, float]:
    if record.get("status") != "ok" or record.get("returncode") != 0:
        raise ValueError(f"failed job: {record.get('job_id')}")
    before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
    if not before or before != after:
        raise ValueError(f"source drift: {record.get('job_id')}")
    result_dir = Path(record.get("expected_artifact", "")).parent
    if not (result_dir / "training_complete.json").is_file():
        raise ValueError(f"missing completion marker: {result_dir}")
    if any((result_dir / item).exists() for item in TEST_ARTIFACTS):
        raise ValueError(f"test artifact present: {result_dir}")
    command = record.get("command", [])
    if "--checkpoints" not in command:
        raise ValueError(f"checkpoint option missing: {record.get('job_id')}")
    loss_path = Path(command[command.index("--checkpoints") + 1]) / result_dir.name / "loss_logs.json"
    values = [float(item) for item in _json(loss_path).get("val_loss", [])]
    if not values or not all(math.isfinite(item) for item in values):
        raise ValueError(f"invalid validation loss: {loss_path}")
    return result_dir, _json(result_dir / "config.json"), min(values)


def _read_phase(root: Path, directory: str, phase: str, models: set[str]) -> tuple[list[dict], set[str]]:
    expected = {(dataset, model, seed) for dataset in DATASETS for model in models for seed in SEEDS}
    seen, rows, hashes = set(), [], set()
    for record in _manifest(root, directory):
        coordinate = (str(record.get("dataset")), str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError(f"invalid or duplicate coordinate: {coordinate}")
        result_dir, config, loss = _validated(record)
        dataset, model, seed = coordinate
        required = {"data": dataset, "model": model, "features": "M", "seq_len": 96,
                    "pred_len": 12, "enc_in": CHANNELS[dataset], "dec_in": CHANNELS[dataset],
                    "c_out": CHANNELS[dataset], "seed": seed, "skip_test": True,
                    "d_model": 24, "d_ff": 48, "patch_len": 24, "patch_stride": 24,
                    "mamba_causal_embedding": True}
        mismatch = [name for name, value in required.items() if config.get(name) != value]
        if mismatch:
            raise ValueError(f"common config mismatch {mismatch}: {result_dir}")
        if phase == "baselines":
            if config.get("AutoCon") is not False or config.get("mamba_acf_relation_loss") is not False:
                raise ValueError(f"candidate option leaked into baseline: {result_dir}")
        else:
            expected_lambda = 0.0 if phase == "null" else 0.001
            if config.get("AutoCon") is not True or config.get("AutoCon_wnorm") != "MeanStd":
                raise ValueError(f"AutoCon config mismatch: {result_dir}")
            if config.get("AutoCon_lambda") != expected_lambda:
                raise ValueError(f"lambda mismatch: {result_dir}")
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        rows.append({"phase": phase, "dataset": dataset, "model": model, "seed": seed,
                     "best_val_mse": loss, "best_epoch": record.get("best_epoch"),
                     "epochs_ran": record.get("epochs_ran"), "elapsed_sec": record.get("elapsed_sec")})
    if seen != expected:
        raise ValueError(f"incomplete coverage for {phase}: expected {len(expected)} got {len(seen)}")
    return rows, hashes


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    baseline_rows, baseline_hashes = _read_phase(root, "baselines", "baselines", set(BASELINES))
    null_rows, null_hashes = _read_phase(root, "null", "null", {"MambaTSv3AutoCon"})
    active_rows, active_hashes = _read_phase(root, "active", "active", {"MambaTSv3AutoCon"})
    hashes = baseline_hashes | null_hashes | active_hashes
    if len(hashes) != 1:
        raise ValueError(f"source snapshot differs across phases: {hashes}")
    rows, summary = baseline_rows + null_rows + active_rows, []
    for dataset in DATASETS:
        values = lambda phase, model=None: [row["best_val_mse"] for row in rows if row["phase"] == phase and row["dataset"] == dataset and (model is None or row["model"] == model)]
        baseline_means = {model: statistics.fmean(values("baselines", model)) for model in BASELINES}
        strongest = min(baseline_means, key=baseline_means.get)
        active, null, plain = values("active"), values("null"), values("baselines", "MambaTSv3")
        active_mean, null_mean, plain_mean = statistics.fmean(active), statistics.fmean(null), statistics.fmean(plain)
        summary.append({"dataset": dataset, "horizon": 12, "active_mean_val_mse": active_mean,
                        "null_mean_val_mse": null_mean, "plain_mean_val_mse": plain_mean,
                        "active_vs_null_percent": 100 * (active_mean / null_mean - 1),
                        "active_vs_plain_percent": 100 * (active_mean / plain_mean - 1),
                        "active_wins_vs_null": sum(a < b for a, b in zip(active, null)),
                        "active_wins_vs_plain": sum(a < b for a, b in zip(active, plain)),
                        "strongest_baseline": strongest, "strongest_baseline_mean_val_mse": baseline_means[strongest],
                        "active_vs_strongest_percent": 100 * (active_mean / baseline_means[strongest] - 1),
                        "passes_external_gate": active_mean <= baseline_means[strongest] * 1.02,
                        "passes_autocon_noninferiority": active_mean <= null_mean * 1.01,
                        "passes_autocon_improvement": active_mean < null_mean and sum(a < b for a, b in zip(active, null)) >= 2,
                        "source_snapshot": next(iter(hashes)), "test_metrics_produced": False})
    transfer_pass = all(row["passes_external_gate"] and row["passes_autocon_noninferiority"] for row in summary)
    for row in summary:
        row["passes_transfer"] = transfer_pass
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "summary.csv", summary)
    lines = ["# METR-LA / PEMS-BAY MambaTSv3 AutoCon H12 Readout", "", "Validation-only. No held-out test metric was generated or read.", "", "| Dataset | Active | Null | Plain | Active vs null | wins/null | Strongest baseline | Active vs baseline | External gate |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
    for row in summary:
        lines.append("| {dataset} | {active_mean_val_mse:.6f} | {null_mean_val_mse:.6f} | {plain_mean_val_mse:.6f} | {active_vs_null_percent:+.3f}% | {active_wins_vs_null}/2 | {strongest_baseline} ({strongest_baseline_mean_val_mse:.6f}) | {active_vs_strongest_percent:+.3f}% | {passes_external_gate} |".format(**row))
    lines.extend(["", f"Source snapshot: `{next(iter(hashes))}`."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passes_transfer": transfer_pass, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
