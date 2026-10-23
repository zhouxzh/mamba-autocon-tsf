#!/usr/bin/env python3
"""Validation-only channelwise patch-representation ACF pretraining probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambats_channel_patch_representation_probe_20261006"
SEED = 6025
TARGETS = {
    "pems_bay": (ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006", 6001),
    "metr_la": (ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_independent_replication_20261006", 6007),
}
ARMS = ("null", "forecast", "channel_patch")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    result[result.index(flag) + 1] = value
    return result


def _commands(target: str, arm: str) -> tuple[list[str], list[str], Path]:
    root, reference_seed = TARGETS[target]
    source_arm = "null" if arm == "null" else "active"
    reference = root / f"s{reference_seed}_{source_arm}"
    prefix = OUTPUT / target / f"s{SEED}_{arm}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    pretrain[0] = sys.executable
    finetune[0] = sys.executable
    if arm == "channel_patch":
        pretrain.extend(("--channel-patch-relation-fraction", ".05"))
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("channel-patch representation probe cannot read test rows")
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "seed": SEED, "targets": list(targets),
        "question": "Does channelwise full patch-trajectory ACF InfoNCE improve transfer beyond forecast-level InfoNCE?",
        "arms": {"null": "next-patch only", "forecast": "released forecast-level ACF InfoNCE",
                 "channel_patch": "forecast-level ACF InfoNCE plus channel patch representation ACF InfoNCE"},
        "fixed_contract": {"mixer": "mamba1", "mamba_ts_variant": "v3", "patch_len": 24,
                           "patch_stride": 8, "d_model": 24, "d_ff": 48, "few_shot_fraction": .1,
                           "target_core_frozen": True, "mamba_scan_count": 1,
                           "channel_patch_relation_fraction": .05, "inference_change": "none"},
        "continuation_rule": "replicate only if channel_patch is lower than forecast on both targets and its pretraining telemetry has active ACF anchors",
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        (OUTPUT / target).mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            pretrain, finetune, prefix = _commands(target, arm)
            pre_event = prefix.with_name(prefix.name + "_pretrain_event.json")
            checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
            fin_event = prefix.with_name(prefix.name + "_finetune_event.json")
            if not (args.resume and pre_event.is_file() and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and _complete(pre_event)):
                _guard(pretrain, prefix, "pretrain")
            if not (args.resume and checkpoint.is_file() and _complete(fin_event)):
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
