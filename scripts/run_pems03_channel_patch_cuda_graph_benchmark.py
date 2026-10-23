#!/usr/bin/env python3
"""Guarded fixed-shape CUDA Graph benchmark for frozen PEMS03 Channel-Patch."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/s6060_channel_patch_frozen_core.pt"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark"


def main():
    if not CHECKPOINT.is_file(): raise FileNotFoundError(CHECKPOINT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_benchmark", "test_metrics_produced": False, "checkpoint": str(CHECKPOINT), "batch_size": 4, "warmup": 25, "iterations": 100, "execution_mode": "fixed-shape CUDA Graph replay including device-side input copy", "contract": "same frozen prediction-only checkpoint as eager benchmark; no dataset reads, no AutoCon representation, no relation loss"}
    (OUTPUT / "cuda_graph_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    prefix = OUTPUT / "channel_patch_cuda_graph"
    command = [sys.executable, str(ROOT / "scripts/benchmark_mambatsv3_checkpoint_inference.py"), "--checkpoint", str(CHECKPOINT), "--output", str(OUTPUT / "channel_patch_cuda_graph.json"), "--batch-size", "4", "--warmup", "25", "--iterations", "100", "--hybrid", "--cuda-graph"]
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


if __name__ == "__main__": main()
