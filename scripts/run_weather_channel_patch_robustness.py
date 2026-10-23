#!/usr/bin/env python3
"""Read-only Weather validation stresses for channel-patch SSL checkpoints."""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006"
OUTPUT = CHECKPOINTS / "robustness_validation"
SEEDS = (5991, 5992, 5993, 5994, 5995)

def guard(command, prefix):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)

def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({"status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seeds": list(SEEDS), "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"], "comparison": "same-seed existing forecast/null and official PatchTST robustness results"}, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        checkpoint = CHECKPOINTS / f"s{seed}_channel_patch_frozen_core.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        prefix = OUTPUT / f"s{seed}_channel_patch"
        guard([sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"), "--dataset", "dataset/weather/weather.csv", "--checkpoint", str(checkpoint), "--output", str(prefix.with_suffix(".json")), "--batch-size", "4", "--seed", str(seed), "--hybrid"], prefix)
    return 0

if __name__ == "__main__": raise SystemExit(main())
