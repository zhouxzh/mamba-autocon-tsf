#!/usr/bin/env python3
"""Fresh three-seed, validation-only Weather frozen-core transfer matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_weather_confirmation_20261003"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_weather_frozen_core_validation_20261003"
SEEDS = (5830, 5831, 5832)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _commands(seed: int, arm: str) -> tuple[list[str], list[str]]:
    old = REFERENCE_ROOT / "s5530_{}".format(arm)
    pretrain = _load(old.with_name(old.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = OUTPUT / "s{}_{}".format(seed, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _load(old.with_name(old.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    finetune = _replace(finetune, "--freeze-mamba-epochs", "0")
    finetune.append("--freeze-pretrained-core")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("validation matrix cannot contain test evaluation")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage))),
        "--event-log", str(prefix.with_name(prefix.name + "_{}_event.json".format(stage))), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Weather", "horizon": 192, "few_shot_fraction": 0.1, "seeds": list(SEEDS),
        "source_reference": str(REFERENCE_ROOT),
        "adaptation": "freeze transferred shared core; train target forecast head, NLinear head, and residual scale",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ("active", "null"):
            prefix = OUTPUT / "s{}_{}".format(seed, arm)
            pretrain, finetune = _commands(seed, arm)
            _guard(pretrain, prefix, "pretrain")
            _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
