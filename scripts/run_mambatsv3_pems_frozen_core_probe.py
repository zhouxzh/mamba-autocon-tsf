#!/usr/bin/env python3
"""Fresh validation-only PEMS-BAY frozen-core representation probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_confirmation_20261003"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_probe_20261003"
DEFAULT_SEEDS = (5800,)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command)
    index = command.index(flag)
    command[index + 1] = value
    return command


def _commands(arm: str, output: Path, seed: int, few_shot_fraction: float) -> tuple[list[str], list[str]]:
    old = REFERENCE_ROOT / "s5510_{}".format(arm)
    pretrain = _load(old.with_name(old.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / "s{}_{}".format(seed, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _load(old.with_name(old.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    finetune = _replace(finetune, "--few-shot-fraction", str(few_shot_fraction))
    finetune = _replace(finetune, "--freeze-mamba-epochs", "0")
    finetune.append("--freeze-pretrained-core")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("probe commands must not evaluate the test split")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage))),
        "--event-log", str(prefix.with_name(prefix.name + "_{}_event.json".format(stage))),
        "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25], default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "PEMS-BAY", "seeds": list(seeds),
        "few_shot_fraction": args.few_shot_fraction,
        "relation_view": "forecast_contrastive", "source_reference": str(REFERENCE_ROOT),
        "adaptation": "freeze transferred core; train target forecast head, NLinear head, and residual scale",
        "gate": "This one-seed probe cannot open test evaluation; a separate fresh three-seed validation matrix is required.",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            prefix = args.output / "s{}_{}".format(seed, arm)
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file()):
                pretrain, _ = _commands(arm, args.output, seed, args.few_shot_fraction)
                _guard(pretrain, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file()):
                _, finetune = _commands(arm, args.output, seed, args.few_shot_fraction)
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
