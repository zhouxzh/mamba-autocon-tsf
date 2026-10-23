#!/usr/bin/env python3
"""Evaluate locked PEMS-BAY frozen-core checkpoints under validation stresses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_probe_20261003"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_robustness_20261003"
SEEDS = (5800, 5801, 5802)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/PEMS-BAY/PEMS-BAY.csv"))
    parser.add_argument("--prefix", default="s")
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_evaluation", "validation_only": True,
        "test_metrics_produced": False, "dataset": str(args.dataset), "seeds": list(seeds),
        "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"],
        "checkpoint_root": str(args.checkpoint_root),
        "invariant": "Each condition mutates validation inference inputs only; target train prefixes and trained weights are read-only.",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            checkpoint = args.checkpoint_root / "{}{}_{}_frozen_core.pt".format(args.prefix, seed, arm)
            output = args.output / "{}{}_{}.json".format(args.prefix, seed, arm)
            if args.resume and output.is_file():
                continue
            prefix = output.with_suffix("")
            command = [
                sys.executable, "scripts/evaluate_mambatsv3_lowdata_robustness.py",
                "--dataset", str(args.dataset), "--checkpoint", str(checkpoint),
                "--output", str(output), "--hybrid", "--batch-size", "4", "--seed", str(seed),
            ]
            subprocess.run([
                sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
                "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
                "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
                "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
            ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
