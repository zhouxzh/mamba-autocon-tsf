#!/usr/bin/env python3
"""Fresh validation-only target channel-adapter probe for strict PEMS transfer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_channel_adapter_probe_20261003"
SEED = 5840


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _commands(arm: str) -> tuple[list[str], list[str]]:
    old = REFERENCE_ROOT / "s5810_{}".format(arm)
    pretrain = _load(old.with_name(old.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = OUTPUT / "s{}_{}".format(SEED, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    finetune = _load(old.with_name(old.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_adapter.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    finetune = _replace(finetune, "--freeze-mamba-epochs", "0")
    if "--freeze-pretrained-core" not in finetune:
        finetune.append("--freeze-pretrained-core")
    finetune.extend(["--mamba-ts-channel-adapter", "--mamba-ts-channel-adapter-rank", "8"])
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("probe commands must not read test rows")
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
        "target_adapter": "rank-8 zero-gated channel-preserving residual from existing one-scan states",
        "adaptation": "freeze transferred shared core; train forecast head, target adapter, NLinear head, and residual scale",
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("active", "null"):
        prefix = OUTPUT / "s{}_{}".format(SEED, arm)
        pretrain, finetune = _commands(arm)
        _guard(pretrain, prefix, "pretrain")
        _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
