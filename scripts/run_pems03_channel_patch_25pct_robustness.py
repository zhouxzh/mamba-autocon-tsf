#!/usr/bin/env python3
"""Read-only six-condition PEMS03 H192/25% robustness matrix."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_pems03_channel_patch_25pct_external_replication_20261007"
OUTPUT = CANDIDATE / "robustness_validation"
SEEDS = (6060, 6061, 6062)
CONDITIONS = ["in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"]


def guard(command, prefix):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_evaluation", "validation_only": True,
        "test_metrics_produced": False, "target": "PEMS03", "horizon": 192,
        "few_shot_fraction": 0.25, "seeds": list(SEEDS), "conditions": CONDITIONS,
        "arms": ["null", "forecast", "channel_patch", "official_patchtst"],
        "same_seed_pairing": True, "target_excluded_from_source_and_initialization": True,
        "source_artifact": str(CANDIDATE),
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ("null", "forecast", "channel_patch"):
            checkpoint = CANDIDATE / f"s{seed}_{arm}_frozen_core.pt"; prefix = OUTPUT / f"s{seed}_{arm}"
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            guard([sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"), "--dataset", "dataset/PEMS03/PEMS03.npz", "--checkpoint", str(checkpoint), "--output", str(prefix.with_suffix(".json")), "--batch-size", "4", "--seed", str(seed), "--hybrid"], prefix)
        checkpoint = PATCH / f"pems03_h192_s{seed}_official_patchtst.pt"; prefix = OUTPUT / f"s{seed}_official_patchtst"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        guard([sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_robustness.py"), "--dataset", "dataset/PEMS03/PEMS03.npz", "--checkpoint", str(checkpoint), "--output", str(prefix.with_suffix(".json")), "--batch-size", "4", "--seed", str(seed)], prefix)


if __name__ == "__main__": main()
