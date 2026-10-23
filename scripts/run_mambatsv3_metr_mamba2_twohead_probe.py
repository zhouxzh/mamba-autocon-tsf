#!/usr/bin/env python3
"""Validation-only METR-LA Mamba-2 two-head frozen-core transfer probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_metr_25_confirmation_20261003"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_twohead_probe_20261006"
DEFAULT_SEEDS = (5890,)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    output = list(command)
    position = output.index(flag)
    output[position + 1] = value
    return output


def _append_unique(command: list[str], flag: str, value: str) -> list[str]:
    return _replace(command, flag, value) if flag in command else [*command, flag, value]


def _commands(seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    previous = REFERENCE / f"s5580_{arm}"
    pretrain = _load(previous.with_name(previous.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / f"s{seed}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    pretrain = _replace(pretrain, "--patch-stride", "8")
    pretrain = _append_unique(pretrain, "--mamba-ts-mixer-type", "mamba2")
    pretrain = _append_unique(pretrain, "--mamba-headdim", "12")
    finetune = _load(previous.with_name(previous.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    finetune = _replace(finetune, "--patch-stride", "8")
    finetune = _replace(finetune, "--freeze-mamba-epochs", "0")
    finetune = _append_unique(finetune, "--mamba-ts-mixer-type", "mamba2")
    finetune = _append_unique(finetune, "--mamba-headdim", "12")
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
        "target": "METR-LA", "horizon": 192, "few_shot_fraction": 0.25, "seeds": list(seeds),
        "source_reference": str(REFERENCE), "source_pool": ["PEMS-BAY", "PEMS07", "PEMS08"],
        "backbone_family": "Mamba-2", "mamba_headdim": 12,
        "patch_geometry": {"patch_len": 24, "patch_stride": 8, "patch_count": 10},
        "adaptation": "freeze source-pretrained shared Mamba-2 core; train target forecast head, NLinear head, and residual scale",
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
