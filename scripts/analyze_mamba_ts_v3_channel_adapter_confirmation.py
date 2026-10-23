#!/usr/bin/env python3
"""Strict validation-only reader for the five-seed channel-adapter matrix."""

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
PHASES = {
    "dlinear": ("DLinear", False), "patchtst": ("PatchTST", False),
    "timesnet": ("TimesNet", False), "mamba2": ("Mamba2", False),
    "mambatsv3_null": ("MambaTSv3AutoCon", False),
    "mambatsv3_adapter": ("MambaTSv3AutoCon", True),
}


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root, phase, model, adapter, source):
    manifest = root / phase / "logs" / "job_manifest.jsonl"
    rows, seen = [], set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line); job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row in {phase}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source mismatch in {phase}: {job_id}")
        result = Path(record["expected_artifact"]).parent; config = load(result / "config.json")
        if config.get("model") != model or config.get("skip_test") is not True:
            raise ValueError(f"configuration boundary mismatch: {job_id}")
        if model == "MambaTSv3AutoCon":
            if config.get("AutoCon") is not True or bool(config.get("mamba_ts_channel_adapter")) != adapter:
                raise ValueError(f"adapter configuration mismatch: {job_id}")
            if adapter and int(config.get("mamba_ts_channel_adapter_rank", 0)) != 8:
                raise ValueError(f"adapter rank mismatch: {job_id}")
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError(f"test artifact present: {job_id}")
        cmd = record["command"]; checkpoint = Path(cmd[cmd.index("--checkpoints") + 1]) / result.name
        losses = [float(x) for x in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(x) for x in losses):
            raise ValueError(f"invalid validation losses: {job_id}")
        rows.append({"phase": phase, "model": model, "dataset": config["data"], "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    expected = {(d, h, s) for d in DATASETS for h in HORIZONS for s in SEEDS}
    found = {(r["dataset"], r["horizon"], r["seed"]) for r in rows}
    if len(rows) != 40 or found != expected:
        raise ValueError(f"coverage mismatch in {phase}: {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    root = args.phase_root.resolve()
    sources = {load(root / phase / "provenance.json")["source_tree_sha256"] for phase in PHASES}
    if len(sources) != 1: raise ValueError(f"source snapshots differ: {sources}")
    source = next(iter(sources)); rows = []
    for phase, (model, adapter) in PHASES.items(): rows.extend(read_phase(root, phase, model, adapter, source))
    keyed = {(r["phase"], r["dataset"], r["horizon"], r["seed"]): r for r in rows}; summary = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            arms = {phase: [keyed[(phase, dataset, horizon, seed)] for seed in SEEDS] for phase in PHASES}
            means = {phase: statistics.fmean(r["best_val_mse"] for r in values) for phase, values in arms.items()}
            best_baseline = min((phase for phase in PHASES if phase != "mambatsv3_adapter"), key=means.get)
            null = arms["mambatsv3_null"]; adapter = arms["mambatsv3_adapter"]; baseline = arms[best_baseline]
            null_delta = [100 * (a["best_val_mse"] - n["best_val_mse"]) / n["best_val_mse"] for a, n in zip(adapter, null)]
            base_delta = [100 * (a["best_val_mse"] - b["best_val_mse"]) / b["best_val_mse"] for a, b in zip(adapter, baseline)]
            summary.append({"dataset": dataset, "horizon": horizon, "best_baseline": best_baseline, "best_baseline_mse": means[best_baseline], "adapter_mse": means["mambatsv3_adapter"], "null_mse": means["mambatsv3_null"], "adapter_vs_best_pct": statistics.fmean(base_delta), "adapter_vs_null_pct": statistics.fmean(null_delta), "adapter_vs_null_wins": sum(x < 0 for x in null_delta), "n": len(SEEDS), "adapter_elapsed_sec": statistics.fmean(r["elapsed_sec"] for r in adapter), "null_elapsed_sec": statistics.fmean(r["elapsed_sec"] for r in null)})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    for name, records in (("validation_rows.csv", rows), ("summary.csv", summary)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    lines = ["# MambaTS v3 channel adapter five-seed confirmation", "", "Validation-only; no test metrics were produced.", "", "| Dataset | H | Best baseline | Adapter MSE | vs. best | vs. null | Null wins / 5 |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in summary: lines.append("| {dataset} | {horizon} | {best_baseline} | {adapter_mse:.6f} | {adapter_vs_best_pct:+.3f}% | {adapter_vs_null_pct:+.3f}% | {adapter_vs_null_wins}/{n} |".format(**r))
    lines += ["", f"Source snapshot: `{source}`", "Negative percentage favors the adapter. This readout does not conduct test selection or make a publication claim."]
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "cells": len(summary), "test_metrics_produced": False, "source_tree_sha256": source})


if __name__ == "__main__": main()
