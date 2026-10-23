#!/usr/bin/env python3
"""Guarded prediction-only PEMS03 Channel-Patch and PatchTST benchmark."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_pems03_channel_patch_25pct_external_replication_20261007"
OUTPUT = CANDIDATE / "inference_benchmark"
SEED = 6060


def guard(command, prefix):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main():
    candidate = CANDIDATE / f"s{SEED}_channel_patch_frozen_core.pt"
    patch = PATCH / f"pems03_h192_s{SEED}_official_patchtst.pt"
    if not candidate.is_file() or not patch.is_file(): raise FileNotFoundError("missing frozen benchmark checkpoint")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_benchmark", "validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "few_shot_fraction": .25, "seed": SEED, "batch_size": 4, "warmup": 25, "iterations": 100, "candidate": str(candidate), "patchtst": str(patch), "contract": "synthetic device-resident inputs; no dataset rows; prediction only"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    guard([sys.executable, str(ROOT / "scripts/benchmark_mambatsv3_checkpoint_inference.py"), "--checkpoint", str(candidate), "--output", str(OUTPUT / "channel_patch.json"), "--batch-size", "4", "--warmup", "25", "--iterations", "100", "--hybrid"], OUTPUT / "channel_patch")
    guard([sys.executable, str(ROOT / "scripts/benchmark_official_patchtst_checkpoint_inference.py"), "--checkpoint", str(patch), "--output", str(OUTPUT / "official_patchtst.json"), "--batch-size", "4", "--warmup", "25", "--iterations", "100"], OUTPUT / "official_patchtst")


if __name__ == "__main__": main()
