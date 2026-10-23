#!/usr/bin/env python3
"""Evaluate frozen Mamba-2 two-head PEMS-BAY checkpoints under validation stresses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "artifacts/exploratory/mambatsv3_pems_mamba2_twohead_probe_20261006"
DEFAULT_OUTPUT = CHECKPOINT_ROOT / "robustness_validation"
SEEDS = (5880, 5881, 5882)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True,
        "test_metrics_produced": False, "target": "PEMS-BAY", "horizon": 192,
        "few_shot_fraction": 0.25, "seeds": list(seeds),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "mamba_contract": {"mixer": "mamba2", "headdim": 12, "patch_len": 24, "patch_stride": 8},
        "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10",
                       "missing_channels_30", "noise_0p05", "noise_0p10"],
        "pairing": "same checkpoint seed and deterministic stress seed for active/null",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            checkpoint = CHECKPOINT_ROOT / f"s{seed}_{arm}_frozen_core.pt"
            output = args.output / f"s{seed}_{arm}.json"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if args.resume and output.is_file():
                continue
            command = [
                sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                "--dataset", "dataset/PEMS-BAY/PEMS-BAY.csv", "--checkpoint", str(checkpoint),
                "--output", str(output), "--batch-size", "4", "--seed", str(seed), "--hybrid",
            ]
            _guard(command, args.output / f"s{seed}_{arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
