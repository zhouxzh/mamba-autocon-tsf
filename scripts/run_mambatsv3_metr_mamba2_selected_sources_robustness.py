#!/usr/bin/env python3
"""Validation stress matrix for ACF-selected-source Mamba-2 METR-LA transfer."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006"
OUTPUT = CHECKPOINT_ROOT / "robustness_validation"
SEEDS = (5891, 5892, 5893)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="skip already completed per-checkpoint readouts")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False,
        "target": "METR-LA", "horizon": 192, "few_shot_fraction": .25, "seeds": list(SEEDS),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10",
                       "missing_channels_30", "noise_0p05", "noise_0p10"],
        "pairing": "same seed and stress realization for active/null", 
    }, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ("active", "null"):
            checkpoint = CHECKPOINT_ROOT / f"s{seed}_{arm}_frozen_core.pt"
            output = OUTPUT / f"s{seed}_{arm}.json"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            event = OUTPUT / f"s{seed}_{arm}_event.json"
            if args.resume and output.is_file() and event.is_file():
                completed = json.loads(event.read_text(encoding="utf-8"))
                if completed.get("status") == "completed" and completed.get("trigger") is None:
                    continue
            command = [
                sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                "--dataset", "dataset/METR-LA/METR-LA.csv", "--checkpoint", str(checkpoint),
                "--output", str(output), "--batch-size", "4", "--seed", str(seed), "--hybrid",
            ]
            _guard(command, OUTPUT / f"s{seed}_{arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
