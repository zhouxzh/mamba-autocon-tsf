#!/usr/bin/env python3
"""Validation-only Electricity full-patch shared target-adapter probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_target_calibration_probe_20261006"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_target_mlp_probe_20261006"
SEED = 5944


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _commands(arm: str) -> tuple[list[str], list[str]]:
    prior = REFERENCE / f"s5942_{arm}"
    pretrain = _load(prior.with_name(prior.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = OUTPUT / f"s{SEED}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    finetune = _load(prior.with_name(prior.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_target_mlp.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    if "--mamba-ts-target-mlp-adapter" not in finetune:
        finetune.extend(["--mamba-ts-target-mlp-adapter", "--mamba-ts-target-mlp-adapter-rank", "32"])
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("target MLP probe cannot read test rows")
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
        "base_protocol": str(REFERENCE),
        "adapter": "rank-32 shared full-patch target decoder, zero-gated, no additional Mamba scan",
        "paired_control": "both active and null use the same Traffic source, Electricity train-prefix calibration, and target adapter; active alone uses ACF relations",
        "mamba_scan_count": 1,
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("active", "null"):
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pretrain, finetune = _commands(arm)
        _guard(pretrain, prefix, "pretrain")
        _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
