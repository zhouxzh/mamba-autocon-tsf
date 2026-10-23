#!/usr/bin/env python3
"""Validation-only Electricity strong-calibration learned-set-pooling probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_set_pooling_strong_calibration_probe_20261006"
SEED = 5988


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    result[result.index(flag) + 1] = value
    return result


def _append_once(command: list[str], flag: str) -> list[str]:
    return list(command) if flag in command else list(command) + [flag]


def _commands(arm: str) -> tuple[list[str], list[str], Path]:
    source_arm = "null" if arm == "pool_null" else "active"
    reference = REFERENCE / f"s5948_{source_arm}"
    prefix = OUTPUT / f"s{SEED}_{arm}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    if arm.startswith("pool_"):
        pretrain = _append_once(pretrain, "--mamba-ts-channel-set-pooling")
        finetune = _append_once(finetune, "--mamba-ts-channel-set-pooling")
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("probe cannot read test rows")
    return pretrain, finetune, prefix


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seed": SEED,
        "arms": ["plain_active", "pool_active", "pool_null"], "base_protocol": str(REFERENCE),
        "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8,
                           "target_core_frozen": True, "mamba_scan_count": 1},
        "relation_schedule": {"Traffic_source": .1, "Electricity_calibration_active": .3,
                              "Electricity_calibration_null": 0.0},
        "question": "Can learned channel set pooling improve high-dimensional strong-calibration ACF transfer?",
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("plain_active", "pool_active", "pool_null"):
        pretrain, finetune, prefix = _commands(arm)
        _guard(pretrain, prefix, "pretrain")
        _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
