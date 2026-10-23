#!/usr/bin/env python3
"""Validation-only 16/8 patch geometry probe for one-scan MambaTSv3 transfer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_patch16_probe_20261003"
SEED = 5867


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _commands(arm: str) -> tuple[list[str], list[str]]:
    old = REFERENCE / "s5810_{}".format(arm)
    pretrain = _load(old.with_name(old.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = OUTPUT / "s{}_{}".format(SEED, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    pretrain = _replace(pretrain, "--patch-len", "16")
    pretrain = _replace(pretrain, "--patch-stride", "8")
    finetune = _load(old.with_name(old.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    finetune = _replace(finetune, "--patch-len", "16")
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": 0.25, "seed": SEED,
        "patch_geometry": {"patch_len": 16, "patch_stride": 8, "patch_count": 11},
        "architecture": "same one-scan shared MambaTSv3 and ACF pretraining; patch geometry matches official PatchTST token overlap",
        "adaptation": "freeze transferred shared core; train target forecast head, NLinear head, and residual scale",
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("active", "null"):
        prefix = OUTPUT / "s{}_{}".format(SEED, arm)
        pretrain, finetune = _commands(arm)
        _guard(pretrain, prefix, "pretrain")
        _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
