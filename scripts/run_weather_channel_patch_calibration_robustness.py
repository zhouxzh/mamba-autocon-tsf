#!/usr/bin/env python3
"""Read-only six-condition robustness matrix for calibrated Weather Channel-Patch."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_weather_channel_patch_calibration_20261007"
OUTPUT = CANDIDATE / "robustness_validation"
SEEDS = (6073, 6074, 6075)
CONDITIONS = ["in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"]


def guard(command, prefix):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seeds": list(SEEDS), "conditions": CONDITIONS, "arms": ["forecast_calibration", "channel_patch_calibration", "official_patchtst"], "same_seed_pairing": True, "source_checkpoints": str(CANDIDATE), "note": "read-only frozen checkpoints; target-prefix calibration occurred before checkpoint creation"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ("forecast_calibration", "channel_patch_calibration"):
            checkpoint = CANDIDATE / f"s{seed}_{arm}_frozen_core.pt"; prefix = OUTPUT / f"s{seed}_{arm}"
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            guard([sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"), "--dataset", "dataset/weather/weather.csv", "--checkpoint", str(checkpoint), "--output", str(prefix.with_suffix(".json")), "--batch-size", "4", "--seed", str(seed), "--hybrid"], prefix)
        checkpoint = PATCH / f"weather_h192_s{seed}_official_patchtst.pt"; prefix = OUTPUT / f"s{seed}_official_patchtst"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        guard([sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_robustness.py"), "--dataset", "dataset/weather/weather.csv", "--checkpoint", str(checkpoint), "--output", str(prefix.with_suffix(".json")), "--batch-size", "4", "--seed", str(seed)], prefix)


if __name__ == "__main__": main()
