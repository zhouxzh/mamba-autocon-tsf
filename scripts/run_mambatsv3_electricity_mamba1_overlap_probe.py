#!/usr/bin/env python3
"""Validation-only Electricity probe: Mamba-1 24/8 with Traffic ACF source."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_similarity_source_candidate_20261003"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_overlap_probe_20261006"
SEED = 5940


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _append(command: list[str], flag: str, value: str) -> list[str]:
    return _replace(command, flag, value) if flag in command else [*command, flag, value]


def _commands(arm: str) -> tuple[list[str], list[str]]:
    previous = REFERENCE / f"electricity_s5650_{arm}"
    pretrain = _load(previous.with_name(previous.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = OUTPUT / f"s{SEED}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    pretrain = _replace(pretrain, "--patch-stride", "8")
    pretrain = _append(pretrain, "--mamba-ts-mixer-type", "mamba1")
    pretrain = _append(pretrain, "--mamba-headdim", "12")
    finetune = _load(previous.with_name(previous.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    finetune = _replace(finetune, "--patch-stride", "8")
    finetune = _replace(finetune, "--freeze-mamba-epochs", "0")
    finetune = _append(finetune, "--mamba-ts-mixer-type", "mamba1")
    finetune = _append(finetune, "--mamba-headdim", "12")
    if "--freeze-pretrained-core" not in finetune:
        finetune.append("--freeze-pretrained-core")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("probe cannot read test rows")
    return pretrain, finetune


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
        "source": "Traffic only, selected by prior target-prefix ACF similarity screen",
        "change_from_failed_screen": "Mamba-1 shared encoder, 24/8 overlapping patches, frozen transferred core",
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("active", "null"):
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pretrain, finetune = _commands(arm)
        _guard(pretrain, prefix, "pretrain")
        _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
