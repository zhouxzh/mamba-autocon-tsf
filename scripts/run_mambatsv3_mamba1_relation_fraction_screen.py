#!/usr/bin/env python3
"""Validation-only source-relation-fraction screen for frozen Mamba-1 transfer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_relation_fraction_screen_20261006"
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
SEED = 5980
TARGETS = {
    "pems_bay": {
        "reference": ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006",
        "reference_seed": 5960,
        "dataset": "PEMS-BAY",
        "source_rule": "fixed PEMS07/PEMS08/PEMS04 pool",
    },
    "metr_la": {
        "reference": ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006",
        "reference_seed": 5970,
        "dataset": "METR-LA",
        "source_rule": "frozen PEMS-BAY/PEMS07/PEMS08 train-prefix ACF selection",
    },
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _tag(fraction: float) -> str:
    return "r" + str(fraction).replace(".", "p")


def _commands(target: str, fraction: float | None, seed: int, output: Path) -> tuple[list[str], list[str], str]:
    spec = TARGETS[target]
    arm = "null" if fraction is None else "active"
    reference = spec["reference"] / f"s{spec['reference_seed']}_{arm}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    tag = "null" if fraction is None else _tag(fraction)
    prefix = output / f"s{seed}_{tag}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    if fraction is not None:
        pretrain = _replace(pretrain, "--relation-loss-fraction", str(fraction))
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("relation-fraction screen cannot read test rows")
    return pretrain, finetune, tag


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def _complete(path: Path) -> bool:
    return path.is_file() and _load(path).get("status") == "completed" and _load(path).get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    parser.add_argument("--fraction", type=float, action="append", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    fractions = tuple(args.fraction or FRACTIONS)
    if (not fractions or len(set(fractions)) != len(fractions)
            or any(fraction <= 0 or fraction > 1 for fraction in fractions)):
        raise ValueError("--fraction values must be unique and in (0, 1]")
    if args.seed <= 0:
        raise ValueError("--seed must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "question": "Can a source-only AutoCon relation-loss fraction improve frozen-core 10% transfer without changing inference?",
        "targets": {name: {"dataset": TARGETS[name]["dataset"], "source_rule": TARGETS[name]["source_rule"]}
                    for name in targets},
        "selection_seed": args.seed, "relation_loss_fractions": list(fractions),
        "null": {"mamba_lambda": 0.0, "relation_loss_fraction": 0.0},
        "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8,
                           "few_shot_fraction": .10, "target_core_frozen": True,
                           "source_next_patch_and_acf_relation": True,
                           "inference_change": "none"},
        "selection_rule": "A fraction is replication-eligible only if it beats or ties its exact null on both targets and improves the mean MSE over fraction 0.10 on both targets. No test reader is authorized by this screen.",
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for fraction in (None, *fractions):
            pretrain, finetune, tag = _commands(target, fraction, args.seed, target_output)
            prefix = target_output / f"s{args.seed}_{tag}"
            checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
            event = prefix.with_name(prefix.name + "_finetune_event.json")
            if args.resume and checkpoint.is_file() and _complete(event):
                continue
            pretrain_checkpoint = prefix.with_name(prefix.name + "_pretrain.pt")
            pretrain_event = prefix.with_name(prefix.name + "_pretrain_event.json")
            if not (args.resume and pretrain_checkpoint.is_file() and _complete(pretrain_event)):
                _guard(pretrain, prefix, "pretrain")
            _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
