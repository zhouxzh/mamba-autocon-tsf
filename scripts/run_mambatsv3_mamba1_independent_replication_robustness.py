#!/usr/bin/env python3
"""Validation stress matrix for the frozen Mamba-1 independent replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATION = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006"
OUTPUT = REPLICATION / "robustness_validation"
PROTOCOLS = {
    "pems_bay": "dataset/PEMS-BAY/PEMS-BAY.csv",
    "metr_la": "dataset/METR-LA/METR-LA.csv",
    "weather": "dataset/weather/weather.csv",
}
SEEDS = (5930, 5931, 5932)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(PROTOCOLS), action="append", default=None)
    args = parser.parse_args()
    targets = tuple(args.target or sorted(PROTOCOLS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False,
        "mixer": "mamba1", "replication_root": str(REPLICATION), "targets": list(targets),
        "seeds": list(SEEDS), "conditions": ["in_distribution", "late_validation_shift",
        "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"],
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            for arm in ("active", "null"):
                checkpoint = REPLICATION / target / f"s{seed}_{arm}_frozen_core.pt"
                output = target_output / f"s{seed}_{arm}.json"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                _guard([
                    sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                    "--dataset", PROTOCOLS[target], "--checkpoint", str(checkpoint),
                    "--output", str(output), "--batch-size", "4", "--seed", str(seed), "--hybrid",
                ], target_output / f"s{seed}_{arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
