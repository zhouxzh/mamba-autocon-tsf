#!/usr/bin/env python3
"""Independent three-domain validation replication for the Mamba-1 speed arm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_screen_20261006"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006"
SEEDS = (5930, 5931, 5932)
TARGETS = ("pems_bay", "metr_la", "weather")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _commands(target: str, seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    prior = REFERENCE / target / f"s5920_{arm}"
    pretrain = _load(prior.with_name(prior.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / f"s{seed}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _load(prior.with_name(prior.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("replication cannot read test rows")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    targets = tuple(args.target or TARGETS)
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "replication_status": "new seeds after Mamba-1 one-seed screen", "seeds": list(seeds),
        "targets": list(targets), "reference_screen": str(REFERENCE),
        "mamba_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8,
                           "target_core_frozen": True},
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for arm in ("active", "null"):
                prefix = target_output / f"s{seed}_{arm}"
                pretrain, finetune = _commands(target, seed, arm, target_output)
                _guard(pretrain, prefix, "pretrain")
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
