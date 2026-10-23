#!/usr/bin/env python3
"""Validation stress matrix for METR-LA H192 10%-label Mamba-1 replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006"
OUTPUT = CHECKPOINTS / "robustness_validation"
SEEDS = (5970, 5971, 5972)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False,
        "target": "METR-LA", "horizon": 192, "few_shot_fraction": .10, "seeds": list(SEEDS),
        "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"],
        "source_selection": "target 10%-prefix ACF selection: PEMS-BAY, PEMS07, PEMS08",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ("active", "null"):
            checkpoint = CHECKPOINTS / f"s{seed}_{arm}_frozen_core.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            _guard([
                sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                "--dataset", "dataset/METR-LA/METR-LA.csv", "--checkpoint", str(checkpoint),
                "--output", str(OUTPUT / f"s{seed}_{arm}.json"), "--batch-size", "4", "--seed", str(seed), "--hybrid",
            ], OUTPUT / f"s{seed}_{arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
