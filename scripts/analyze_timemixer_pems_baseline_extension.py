#!/usr/bin/env python3
"""Strict validation-only reader for the TimeMixer PEMS baseline extension.

The comparison is descriptive: the TimeMixer run has its own source snapshot,
while the reference rows come from the already frozen PEMS confirmation matrix.
No test artifact or test metric is accepted by this reader.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("pems03", "pems04", "pems07", "pems08")
HORIZONS = (12, 96)
SEEDS = (2021, 2022, 2023, 2024, 2025)
TIMEMIXER_MODEL = "TimeMixer"
REFERENCE_MODELS = (
    "dlinear",
    "patchtst",
    "timesnet",
    "mamba2",
    "mambatsv3_null",
)
MODEL_LABELS = {
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
    "mamba2": "Mamba2",
    "mambatsv3_null": "MambaTSv3AutoCon-null",
}


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _best_validation_mse(checkpoint_dir: Path) -> float:
    values = [float(value) for value in _load(checkpoint_dir / "loss_logs.json").get("val_loss", [])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid validation loss: {}".format(checkpoint_dir))
    return min(values)


def _read_timemixer(phase_root: Path):
    phase_dir = phase_root / "timemixer"
    provenance = _load(phase_dir / "provenance.json")
    source_hash = provenance.get("source_tree_sha256")
    if not source_hash:
        raise ValueError("missing TimeMixer source snapshot")
    expected = {(dataset, horizon, seed) for dataset in DATASETS for horizon in HORIZONS for seed in SEEDS}
    rows, seen = [], set()
    manifest = phase_dir / "logs" / "job_manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid TimeMixer manifest row")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError("TimeMixer source snapshot mismatch")
        result_dir = Path(record["expected_artifact"]).parent
        config = _load(result_dir / "config.json")
        if config.get("model") != TIMEMIXER_MODEL or config.get("skip_test") is not True:
            raise ValueError("TimeMixer model or test boundary mismatch")
        if any((result_dir / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("TimeMixer test artifact present")
        command = record.get("command", [])
        if "--checkpoints" not in command:
            raise ValueError("missing checkpoint root in TimeMixer command")
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result_dir.name
        row = {
            "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]),
            "model": TIMEMIXER_MODEL,
            "phase": "timemixer",
            "best_val_mse": _best_validation_mse(checkpoint),
            "elapsed_sec": float(record["elapsed_sec"]),
            "best_epoch": int(record.get("best_epoch", -1)),
        }
        rows.append(row)
    found = {(row["dataset"], row["horizon"], row["seed"]) for row in rows}
    if len(rows) != len(expected) or found != expected:
        raise ValueError("TimeMixer coordinate coverage mismatch: {}".format(len(rows)))
    return rows, source_hash, provenance.get("dataset_hashes", {})


def _read_references(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["dataset"] = str(row["dataset"])
            row["horizon"] = int(row["horizon"])
            row["seed"] = int(row["seed"])
            row["model"] = str(row.get("phase", ""))
            row["best_val_mse"] = float(row["best_val_mse"])
            row["elapsed_sec"] = float(row["elapsed_sec"])
            rows.append(row)
    expected = {(dataset, horizon, seed) for dataset in DATASETS for horizon in HORIZONS for seed in SEEDS}
    # The source readout also contains the stopped adapter phase.  It is
    # intentionally excluded from this baseline comparison, but its presence
    # is checked so a silently changed reference file cannot pass.
    allowed_phases = set(REFERENCE_MODELS) | {"mambatsv3_adapter"}
    models = set(row.get("phase") for row in rows)
    if models != allowed_phases:
        raise ValueError("reference phase set mismatch: {}".format(sorted(models)))
    rows = [row for row in rows if row.get("phase") in REFERENCE_MODELS]
    found = {(row["dataset"], row["horizon"], row["seed"], row["phase"]) for row in rows}
    expected_rows = {(dataset, horizon, seed, model) for dataset, horizon, seed in expected for model in REFERENCE_MODELS}
    if len(rows) != len(expected_rows) or found != expected_rows:
        raise ValueError("reference coordinate coverage mismatch: {}".format(len(rows)))
    if not all(math.isfinite(row["best_val_mse"]) for row in rows):
        raise ValueError("non-finite reference validation loss")
    return rows


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        for row in rows[1:]:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timemixer-root", type=Path, required=True)
    parser.add_argument("--reference-rows", type=Path, required=True)
    parser.add_argument("--reference-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    timemixer_rows, timemixer_source, timemixer_hashes = _read_timemixer(args.timemixer_root.resolve())
    reference_rows = _read_references(args.reference_rows.resolve())
    reference_provenance = _load(args.reference_provenance.resolve())
    reference_hashes = reference_provenance.get("dataset_hashes", {})
    if timemixer_hashes != reference_hashes:
        raise ValueError("dataset hashes differ between TimeMixer and reference matrix")

    all_rows = reference_rows + timemixer_rows
    keyed = {(row["model"], row["dataset"], row["horizon"], row["seed"]): row for row in all_rows}
    summaries, pairwise = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            tm = [keyed[(TIMEMIXER_MODEL, dataset, horizon, seed)] for seed in SEEDS]
            means = {
                model: statistics.fmean(keyed[(model, dataset, horizon, seed)]["best_val_mse"] for seed in SEEDS)
                for model in REFERENCE_MODELS
            }
            tm_mean = statistics.fmean(row["best_val_mse"] for row in tm)
            best_model = min(means, key=means.get)
            best_non_autocon = min((model for model in REFERENCE_MODELS if model != "mambatsv3_null"), key=means.get)
            summaries.append({
                "dataset": dataset,
                "horizon": horizon,
                "timemixer_mean_val_mse": tm_mean,
                "timemixer_std_val_mse": statistics.stdev(row["best_val_mse"] for row in tm),
                "best_reference_model": MODEL_LABELS[best_model],
                "best_reference_mean_val_mse": means[best_model],
                "best_non_autocon_model": MODEL_LABELS[best_non_autocon],
                "best_non_autocon_mean_val_mse": means[best_non_autocon],
                "timemixer_mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in tm),
                "best_reference_mean_elapsed_sec": statistics.fmean(keyed[(best_model, dataset, horizon, seed)]["elapsed_sec"] for seed in SEEDS),
            })
            for model in REFERENCE_MODELS:
                ref = [keyed[(model, dataset, horizon, seed)] for seed in SEEDS]
                deltas = [100.0 * (a["best_val_mse"] - b["best_val_mse"]) / b["best_val_mse"] for a, b in zip(tm, ref)]
                elapsed = [100.0 * (a["elapsed_sec"] - b["elapsed_sec"]) / b["elapsed_sec"] for a, b in zip(tm, ref)]
                pairwise.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "reference_model": MODEL_LABELS[model],
                    "timemixer_mean_val_mse": tm_mean,
                    "reference_mean_val_mse": means[model],
                    "timemixer_vs_reference_pct": statistics.fmean(deltas),
                    "timemixer_wins": sum(delta < 0.0 for delta in deltas),
                    "timemixer_mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in tm),
                    "reference_mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in ref),
                    "timemixer_elapsed_vs_reference_pct": statistics.fmean(elapsed),
                    "n": len(SEEDS),
                })

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", all_rows)
    _write_csv(out / "summary.csv", summaries)
    _write_csv(out / "pairwise_summary.csv", pairwise)
    lines = [
        "# TimeMixer PEMS baseline extension",
        "",
        "Validation-only: 40 TimeMixer jobs plus the five-seed reference rows; no test metrics were read or produced.",
        "",
        "TimeMixer and the reference matrix use identical dataset hashes, but different source snapshots; comparisons are descriptive and are not a same-freeze significance claim.",
        "",
        "| Dataset | H | TimeMixer MSE | Best reference | Best ref. MSE | vs. best ref. | Non-AutoCon best |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in summaries:
        best_delta = 100.0 * (row["timemixer_mean_val_mse"] - row["best_reference_mean_val_mse"]) / row["best_reference_mean_val_mse"]
        lines.append("| {dataset} | {horizon} | {timemixer_mean_val_mse:.6f} | {best_reference_model} | {best_reference_mean_val_mse:.6f} | {delta:+.3f}% | {best_non_autocon_model} |".format(delta=best_delta, **row))
    lines += [
        "",
        "Negative accuracy percentage favors TimeMixer. Negative elapsed percentage favors TimeMixer.",
        "",
        "TimeMixer source snapshot: `{}`".format(timemixer_source),
        "Reference source snapshot: `{}`".format(reference_provenance.get("source_tree_sha256", "missing")),
        "The reference rows are consumed from the independently audited frozen PEMS confirmation readout.",
    ]
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"timemixer_rows": len(timemixer_rows), "reference_rows": len(reference_rows), "cells": len(summaries), "pairwise_cells": len(pairwise), "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
