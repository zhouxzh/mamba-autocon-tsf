#!/usr/bin/env python3
"""Guarded PEMS03 fixed-shape device-only CUDA Graph batch grid."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/s6060_channel_patch_frozen_core.pt"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_pems03_channel_patch_25pct_external_replication_20261007/pems03_h192_s6060_official_patchtst.pt"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/cuda_graph_batch_grid"
BATCHES = (1, 4, 8)


def guard(command, prefix):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    if not CANDIDATE.is_file() or not PATCH.is_file(): raise FileNotFoundError("missing benchmark checkpoint")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_benchmark", "test_metrics_produced": False, "batches": list(BATCHES), "warmup": 25, "iterations": 100, "candidate": str(CANDIDATE), "patchtst": str(PATCH), "candidate_execution": "fixed-shape CUDA Graph including device-side input copy", "patchtst_execution": "official eager", "scope": "device-only fixed-shape latency; excludes host I/O and graph construction"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for batch in BATCHES:
        candidate = [sys.executable, str(ROOT / "scripts/benchmark_mambatsv3_checkpoint_inference.py"), "--checkpoint", str(CANDIDATE), "--output", str(OUTPUT / f"channel_patch_b{batch}.json"), "--batch-size", str(batch), "--warmup", "25", "--iterations", "100", "--hybrid", "--cuda-graph"]
        patch = [sys.executable, str(ROOT / "scripts/benchmark_official_patchtst_checkpoint_inference.py"), "--checkpoint", str(PATCH), "--output", str(OUTPUT / f"official_patchtst_b{batch}.json"), "--batch-size", str(batch), "--warmup", "25", "--iterations", "100"]
        guard(candidate, OUTPUT / f"channel_patch_b{batch}")
        guard(patch, OUTPUT / f"official_patchtst_b{batch}")


if __name__ == "__main__": main()
