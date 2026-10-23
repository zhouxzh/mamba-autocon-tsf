#!/usr/bin/env python3
"""Validation-only probe for ACF-confidence-weighted forecast InfoNCE."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_acf_confidence_probe_20261006"
SEED = 5981
TARGETS = {
    "pems_bay": (ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006", 5960, "PEMS-BAY"),
    "metr_la": (ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006", 5970, "METR-LA"),
}
ARMS = ("null", "plain", "confidence")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    output = list(command)
    index = output.index(flag)
    output[index + 1] = value
    return output


def _append_once(command: list[str], flag: str) -> list[str]:
    return list(command) if flag in command else list(command) + [flag]


def _commands(target: str, arm: str) -> tuple[list[str], list[str], Path]:
    reference_root, reference_seed, _ = TARGETS[target]
    source_arm = "null" if arm == "null" else "active"
    reference = reference_root / f"s{reference_seed}_{source_arm}"
    prefix = OUTPUT / target / f"s{SEED}_{arm}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    if arm == "confidence":
        pretrain = _append_once(pretrain, "--forecast-contrastive-acf-weighting")
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("confidence probe cannot read test rows")
    return pretrain, finetune, prefix


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
    parser.add_argument("--arm", choices=ARMS, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    arms = tuple(args.arm or ARMS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "question": "Does weighting positive forecast pairs by train-only ACF magnitude improve the fixed 0.10 AutoCon pretraining?",
        "seed": SEED, "targets": {name: TARGETS[name][2] for name in targets}, "arms": list(arms),
        "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8,
                           "few_shot_fraction": .10, "relation_loss_fraction": .10,
                           "target_core_frozen": True, "inference_change": "none"},
        "active_change": "ForecastLagPairContrastiveLoss weights positive terms by train-only |ACF(lag)|; null and plain arms disable it",
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        (OUTPUT / target).mkdir(parents=True, exist_ok=True)
        for arm in arms:
            pretrain, finetune, prefix = _commands(target, arm)
            checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
            event = prefix.with_name(prefix.name + "_finetune_event.json")
            pretrain_checkpoint = prefix.with_name(prefix.name + "_pretrain.pt")
            pretrain_event = prefix.with_name(prefix.name + "_pretrain_event.json")
            if not (args.resume and pretrain_checkpoint.is_file() and _complete(pretrain_event)):
                _guard(pretrain, prefix, "pretrain")
            if not (args.resume and checkpoint.is_file() and _complete(event)):
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
