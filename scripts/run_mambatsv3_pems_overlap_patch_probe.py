#!/usr/bin/env python3
"""Validation-only overlapping-patch one-scan MambaTSv3 transfer probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_overlap_patch_probe_20261003"
DEFAULT_SEEDS = (5860,)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _commands(seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    old = REFERENCE / "s5810_{}".format(arm)
    pretrain = _load(old.with_name(old.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / "s{}_{}".format(seed, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    pretrain = _replace(pretrain, "--patch-stride", "8")
    finetune = _load(old.with_name(old.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    finetune = _replace(finetune, "--patch-stride", "8")
    finetune = _replace(finetune, "--freeze-mamba-epochs", "0")
    if "--freeze-pretrained-core" not in finetune:
        finetune.append("--freeze-pretrained-core")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("probe cannot read test rows")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage))),
        "--event-log", str(prefix.with_name(prefix.name + "_{}_event.json".format(stage))), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": 0.25, "seeds": list(seeds),
        "patch_geometry": {"patch_len": 24, "patch_stride": 8, "patch_count": 10},
        "architecture": "same MambaTSv3 module invoked once per channel patch sequence; only token overlap changes",
        "adaptation": "freeze transferred shared core; train target forecast head, NLinear head, and residual scale",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            prefix = args.output / "s{}_{}".format(seed, arm)
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file()):
                pretrain, _ = _commands(seed, arm, args.output)
                _guard(pretrain, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file()):
                _, finetune = _commands(seed, arm, args.output)
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
