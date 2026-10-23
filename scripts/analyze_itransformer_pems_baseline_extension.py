#!/usr/bin/env python3
"""Read the iTransformer PEMS extension without accepting test artifacts.

The iTransformer, frozen Mamba reference, and TimeMixer extension have
different source snapshots.  Their identical PEMS data hashes permit a
descriptive paired validation comparison, but not a same-freeze inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("pems03", "pems04", "pems07", "pems08")
HORIZONS = (12, 96)
SEEDS = (2021, 2022, 2023, 2024, 2025)
MAMBA_PHASES = ("dlinear", "patchtst", "timesnet", "mamba2", "mambatsv3_null")
MAMBA_LABELS = {
    "dlinear": "DLinear", "patchtst": "PatchTST", "timesnet": "TimesNet",
    "mamba2": "Mamba2", "mambatsv3_null": "MambaTSv3AutoCon-null",
}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _best_loss(checkpoint: Path) -> float:
    values = [float(value) for value in _load(checkpoint / "loss_logs.json").get("val_loss", [])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid validation loss: {}".format(checkpoint))
    return min(values)


def _read_phase(phase_root: Path, phase: str, expected_model: str):
    source = _load(phase_root / "provenance.json")
    source_hash = source.get("source_tree_sha256")
    if not source_hash:
        raise ValueError("missing source snapshot for {}".format(phase))
    expected = {(d, h, s) for d in DATASETS for h in HORIZONS for s in SEEDS}
    rows, seen = [], set()
    for line in (phase_root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row in {}".format(phase))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError("source snapshot mismatch in {}".format(phase))
        result = Path(record["expected_artifact"]).parent
        config = _load(result / "config.json")
        if config.get("model") != expected_model or config.get("skip_test") is not True:
            raise ValueError("model/test boundary mismatch in {}".format(job_id))
        if any((result / artifact).exists() for artifact in TEST_ARTIFACTS):
            raise ValueError("test artifact present in {}".format(job_id))
        command = record.get("command", [])
        if "--checkpoints" not in command:
            raise ValueError("checkpoint root missing in {}".format(job_id))
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]), "model": expected_model, "phase": phase,
            "best_val_mse": _best_loss(checkpoint), "elapsed_sec": float(record["elapsed_sec"]),
            "best_epoch": int(record.get("best_epoch", -1)),
        })
    found = {(row["dataset"], row["horizon"], row["seed"]) for row in rows}
    if len(rows) != len(expected) or found != expected:
        raise ValueError("coordinate coverage mismatch in {}: {}".format(phase, len(rows)))
    return rows, source_hash, source.get("dataset_hashes", {})


def _read_mamba_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase = str(row["phase"])
            if phase not in MAMBA_PHASES:
                continue
            rows.append({
                "dataset": str(row["dataset"]), "horizon": int(row["horizon"]),
                "seed": int(row["seed"]), "model": MAMBA_LABELS[phase], "phase": phase,
                "best_val_mse": float(row["best_val_mse"]), "elapsed_sec": float(row["elapsed_sec"]),
            })
    expected = {(d, h, s, phase) for d in DATASETS for h in HORIZONS for s in SEEDS for phase in MAMBA_PHASES}
    found = {(row["dataset"], row["horizon"], row["seed"], row["phase"]) for row in rows}
    if len(rows) != len(expected) or found != expected:
        raise ValueError("Mamba reference coordinate coverage mismatch")
    return rows


def _read_timemixer_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "TimeMixer":
                continue
            rows.append({
                "dataset": str(row["dataset"]), "horizon": int(row["horizon"]),
                "seed": int(row["seed"]), "model": "TimeMixer", "phase": "timemixer",
                "best_val_mse": float(row["best_val_mse"]), "elapsed_sec": float(row["elapsed_sec"]),
            })
    expected = {(d, h, s) for d in DATASETS for h in HORIZONS for s in SEEDS}
    if len(rows) != len(expected) or {(r["dataset"], r["horizon"], r["seed"]) for r in rows} != expected:
        raise ValueError("TimeMixer reference coordinate coverage mismatch")
    return rows


def _write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itransformer-root", type=Path, required=True)
    parser.add_argument("--mamba-reference-rows", type=Path, required=True)
    parser.add_argument("--mamba-reference-provenance", type=Path, required=True)
    parser.add_argument("--timemixer-rows", type=Path, required=True)
    parser.add_argument("--timemixer-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate, candidate_source, candidate_hashes = _read_phase(
        args.itransformer_root.resolve() / "itransformer", "itransformer", "iTransformer"
    )
    mamba = _read_mamba_rows(args.mamba_reference_rows.resolve())
    timemixer = _read_timemixer_rows(args.timemixer_rows.resolve())
    mamba_provenance = _load(args.mamba_reference_provenance.resolve())
    timemixer_provenance = _load(args.timemixer_provenance.resolve())
    if candidate_hashes != mamba_provenance.get("dataset_hashes") or candidate_hashes != timemixer_provenance.get("dataset_hashes"):
        raise ValueError("PEMS dataset hash mismatch across provenance roots")

    all_rows = mamba + timemixer + candidate
    keyed = {(row["model"], row["dataset"], row["horizon"], row["seed"]): row for row in all_rows}
    references = tuple(MAMBA_LABELS.values()) + ("TimeMixer",)
    summary, pairwise = [], []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            arm = [keyed[("iTransformer", dataset, horizon, seed)] for seed in SEEDS]
            means = {model: statistics.fmean(keyed[(model, dataset, horizon, seed)]["best_val_mse"] for seed in SEEDS) for model in references}
            winner = min(means, key=means.get)
            candidate_mean = statistics.fmean(row["best_val_mse"] for row in arm)
            summary.append({
                "dataset": dataset, "horizon": horizon,
                "itransformer_mean_val_mse": candidate_mean,
                "itransformer_std_val_mse": statistics.stdev(row["best_val_mse"] for row in arm),
                "best_existing_reference": winner,
                "best_existing_reference_mean_val_mse": means[winner],
                "itransformer_mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in arm),
            })
            for reference in references:
                ref = [keyed[(reference, dataset, horizon, seed)] for seed in SEEDS]
                deltas = [100.0 * (a["best_val_mse"] - b["best_val_mse"]) / b["best_val_mse"] for a, b in zip(arm, ref)]
                pairwise.append({
                    "dataset": dataset, "horizon": horizon, "reference_model": reference,
                    "itransformer_vs_reference_pct": statistics.fmean(deltas),
                    "itransformer_wins": sum(value < 0.0 for value in deltas), "n": len(SEEDS),
                    "itransformer_mean_val_mse": candidate_mean,
                    "reference_mean_val_mse": means[reference],
                    "itransformer_mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in arm),
                    "reference_mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in ref),
                })

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "validation_rows.csv", all_rows)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "pairwise_summary.csv", pairwise)
    lines = [
        "# iTransformer PEMS baseline extension", "",
        "Validation-only descriptive comparison. No test metrics were read or produced.", "",
        "The iTransformer, frozen Mamba reference, and TimeMixer reference have different source snapshots but identical PEMS data hashes. This is not a same-freeze statistical comparison.", "",
        "| Dataset | H | iTransformer MSE | Existing best | Existing best MSE | vs. existing best |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary:
        delta = 100.0 * (row["itransformer_mean_val_mse"] - row["best_existing_reference_mean_val_mse"]) / row["best_existing_reference_mean_val_mse"]
        lines.append("| {dataset} | {horizon} | {itransformer_mean_val_mse:.6f} | {best_existing_reference} | {best_existing_reference_mean_val_mse:.6f} | {delta:+.3f}% |".format(delta=delta, **row))
    lines += ["", "Negative percentage favors iTransformer.", "", "iTransformer source snapshot: `{}`".format(candidate_source), "Frozen Mamba reference snapshot: `{}`".format(mamba_provenance.get("source_tree_sha256")), "TimeMixer reference snapshot: `{}`".format(timemixer_provenance.get("source_tree_sha256"))]
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"itransformer_rows": len(candidate), "reference_rows": len(mamba) + len(timemixer), "cells": len(summary), "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
