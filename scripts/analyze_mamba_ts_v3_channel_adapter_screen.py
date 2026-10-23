#!/usr/bin/env python3
"""Strict validation-only reader for the MambaTS v3 channel-adapter screen."""

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
SEEDS = (2022, 2023)
ARMS = {"adapter_null": False, "adapter_rank8": True}


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_from_command(command, result_name):
    values = list(command)
    return Path(values[values.index("--checkpoints") + 1]) / result_name


def read_phase(root: Path, phase: str, expected_adapter: bool, source_hash: str):
    manifest = root / phase / "logs" / "job_manifest.jsonl"
    if not manifest.is_file():
        raise ValueError(f"missing manifest: {manifest}")
    rows, seen = [], set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "ok":
            raise ValueError(f"non-ok manifest row in {phase}: {record.get('job_id')}")
        job_id = record.get("job_id")
        if not job_id or job_id in seen:
            raise ValueError(f"duplicate/missing job id in {phase}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError(f"source hash mismatch in {phase}: {job_id}")
        result_dir = Path(record["expected_artifact"]).parent
        config = load(result_dir / "config.json")
        if config.get("model") != "MambaTSv3AutoCon" or config.get("AutoCon") is not True:
            raise ValueError(f"model contract mismatch: {job_id}")
        if bool(config.get("mamba_ts_channel_adapter")) != expected_adapter:
            raise ValueError(f"adapter contract mismatch: {job_id}")
        if expected_adapter and int(config.get("mamba_ts_channel_adapter_rank", 0)) != 8:
            raise ValueError(f"adapter rank mismatch: {job_id}")
        if config.get("skip_test") is not True:
            raise ValueError(f"test was not disabled: {job_id}")
        if any((result_dir / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError(f"test artifact present: {job_id}")
        checkpoint = checkpoint_from_command(record["command"], result_dir.name)
        losses = [float(x) for x in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(x) for x in losses):
            raise ValueError(f"invalid validation trajectory: {job_id}")
        rows.append({
            "phase": phase, "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "adapter": expected_adapter, "best_val_mse": min(losses),
            "elapsed_sec": float(record["elapsed_sec"]), "job_id": job_id,
        })
    expected = {(d, h, s) for d in DATASETS for h in HORIZONS for s in SEEDS}
    found = {(r["dataset"], r["horizon"], r["seed"]) for r in rows}
    if len(rows) != 16 or found != expected:
        raise ValueError(f"coverage mismatch in {phase}: {len(rows)} rows")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.phase_root.resolve()
    plan = load(root / "adapter_null" / "provenance" / "final_training_plan.json")
    source_hashes = {
        load(root / phase / "provenance.json")["source_tree_sha256"] for phase in ARMS
    }
    if len(source_hashes) != 1:
        raise ValueError(f"phase source snapshots differ: {source_hashes}")
    source_hash = next(iter(source_hashes))
    rows = read_phase(root, "adapter_null", False, source_hash) + read_phase(root, "adapter_rank8", True, source_hash)
    by_key = {(r["dataset"], r["horizon"], r["seed"], r["adapter"]): r for r in rows}
    paired = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            nulls = [by_key[(dataset, horizon, seed, False)] for seed in SEEDS]
            active = [by_key[(dataset, horizon, seed, True)] for seed in SEEDS]
            changes = [100.0 * (a["best_val_mse"] - n["best_val_mse"]) / n["best_val_mse"] for n, a in zip(nulls, active)]
            paired.append({
                "dataset": dataset, "horizon": horizon,
                "null_mse_mean": statistics.fmean(r["best_val_mse"] for r in nulls),
                "active_mse_mean": statistics.fmean(r["best_val_mse"] for r in active),
                "mean_change_pct": statistics.fmean(changes),
                "active_wins": sum(x < 0 for x in changes), "n": len(changes),
                "null_elapsed_mean_sec": statistics.fmean(r["elapsed_sec"] for r in nulls),
                "active_elapsed_mean_sec": statistics.fmean(r["elapsed_sec"] for r in active),
                "latency_proxy_change_pct": 100.0 * (statistics.fmean(r["elapsed_sec"] for r in active) - statistics.fmean(r["elapsed_sec"] for r in nulls)) / statistics.fmean(r["elapsed_sec"] for r in nulls),
            })
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    lines = ["# MambaTS v3 AutoCon channel adapter validation screen", "", "Validation-only; no test metrics were produced.", "", "| Dataset | H | Null MSE | Adapter MSE | Change | Wins / 2 | Adapter time change |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in paired:
        lines.append("| {dataset} | {horizon} | {null_mse_mean:.6f} | {active_mse_mean:.6f} | {mean_change_pct:+.3f}% | {active_wins}/{n} | {latency_proxy_change_pct:+.2f}% |".format(**row))
    lines += ["", f"Source snapshot: `{source_hash}`", "Negative change favors the adapter. This is exploratory validation evidence, not a test or publication claim."]
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_cells": len(paired), "test_metrics_produced": False, "source_tree_sha256": source_hash})


if __name__ == "__main__":
    main()
