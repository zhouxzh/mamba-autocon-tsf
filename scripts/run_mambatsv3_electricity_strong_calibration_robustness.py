#!/usr/bin/env python3
"""Validation stress matrix for Electricity strong-calibration Mamba-1 checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
OUTPUT = CHECKPOINTS / "robustness_validation"
SEEDS = (5948, 5949, 5950)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seeds": list(seeds),
        "protocol": "Traffic relation fraction 0.1 plus Electricity train-prefix relation fraction 0.3",
        "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10",
                       "missing_channels_30", "noise_0p05", "noise_0p10"],
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            checkpoint = CHECKPOINTS / f"s{seed}_{arm}_frozen_core.pt"
            output = OUTPUT / f"s{seed}_{arm}.json"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            _guard([
                sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                "--dataset", "dataset/electricity/electricity.csv", "--checkpoint", str(checkpoint),
                "--output", str(output), "--batch-size", "4", "--seed", str(seed), "--hybrid",
            ], OUTPUT / f"s{seed}_{arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
