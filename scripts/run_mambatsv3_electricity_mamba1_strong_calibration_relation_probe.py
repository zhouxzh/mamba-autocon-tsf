#!/usr/bin/env python3
"""Validation-only Electricity stronger target-calibration AutoCon probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_target_calibration_probe_20261006"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
DEFAULT_SEEDS = (5948,)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _append(command: list[str], flag: str, value: str) -> list[str]:
    return _replace(command, flag, value) if flag in command else [*command, flag, value]


def _commands(seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    prior = REFERENCE / f"s5942_{arm}"
    pretrain = _load(prior.with_name(prior.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / f"s{seed}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    pretrain = _append(pretrain, "--calibration-relation-loss-fraction", ".3" if arm == "active" else "0")
    finetune = _load(prior.with_name(prior.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("strong calibration probe cannot read test rows")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seeds": list(seeds),
        "base_protocol": str(REFERENCE),
        "relation_schedule": {"Traffic_source": .1, "Electricity_calibration_active": .3, "Electricity_calibration_null": 0.0},
        "paired_control": "both arms use identical Traffic and Electricity next-patch tasks; only active has ACF relations",
        "mamba_scan_count": 1,
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            prefix = args.output / f"s{seed}_{arm}"
            pretrain, finetune = _commands(seed, arm, args.output)
            _guard(pretrain, prefix, "pretrain")
            _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
