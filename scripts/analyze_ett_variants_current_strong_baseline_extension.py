#!/usr/bin/env python3
"""Strict validation-only readout for the ETTh2/ETTm2 baseline extension."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

DATASETS = ("ETTh2", "ETTm2")
SEEDS = (2021, 2022, 2023)
MODELS = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = root / "logs" / "job_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(dataset, seed, model) for dataset in DATASETS for seed in SEEDS for model in MODELS}
    found, hashes, rows = set(), set(), []
    for record in records:
        coordinate = (record.get("dataset"), int(record.get("seed")), record.get("model"))
        if coordinate not in expected or coordinate in found:
            raise ValueError("unexpected or duplicate manifest record: {}".format(record.get("job_id")))
        if record.get("status") != "ok" or record.get("returncode") != 0 or int(record.get("pred_len")) != 96:
            raise ValueError("failed or mismatched run: {}".format(record.get("job_id")))
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file() or any((result_dir / item).exists() for item in TEST_ARTIFACTS):
            raise ValueError("invalid validation-only boundary: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        if config.get("model") != coordinate[2] or config.get("data") != coordinate[0] or config.get("seed") != coordinate[1] or config.get("skip_test") is not True:
            raise ValueError("config mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        losses = [float(value) for value in _load(checkpoint_root / result_dir.name / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("missing/non-finite validation loss: {}".format(result_dir))
        found.add(coordinate)
        rows.append({"dataset": coordinate[0], "horizon": 96, "seed": coordinate[1], "model": coordinate[2], "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"])})
    if found != expected or len(hashes) != 1:
        raise ValueError("coverage/source mismatch")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", rows)
    summary = []
    for dataset in DATASETS:
        means = {model: statistics.fmean(row["best_val_mse"] for row in rows if row["dataset"] == dataset and row["model"] == model) for model in MODELS}
        fastest = {model: statistics.fmean(row["elapsed_sec"] for row in rows if row["dataset"] == dataset and row["model"] == model) for model in MODELS}
        winner = min(MODELS, key=means.get)
        summary.append({"dataset": dataset, "horizon": 96, "best_model": winner, "best_mean_val_mse": means[winner], "best_model_mean_elapsed_sec": fastest[winner], **{"{}_mean_val_mse".format(model): means[model] for model in MODELS}})
    _write_csv(out / "summary.csv", summary)
    lines = ["# ETTh2/ETTm2 current strong-baseline extension", "", "Validation-only results. No test metric was read or produced.", "", "| Dataset | DLinear | NLinear | PatchTST | TimesNet | iTransformer | Best |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        lines.append("| {dataset} | {DLinear_mean_val_mse:.6f} | {NLinear_mean_val_mse:.6f} | {PatchTST_mean_val_mse:.6f} | {TimesNet_mean_val_mse:.6f} | {iTransformer_mean_val_mse:.6f} | {best_model} ({best_mean_val_mse:.6f}) |".format(**row))
    lines.extend(["", "Source snapshot: `{}`.".format(next(iter(hashes)))])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": next(iter(hashes)), "test_metrics_produced": False})


if __name__ == "__main__":
    main()
