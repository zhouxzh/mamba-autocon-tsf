#!/usr/bin/env python3
"""Strict PEMS03 validation screen for a shared Channel-Patch SSL projector."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_projector_probe_20261007"
SEED = 6089
ARMS = ("identity", "mlp")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace(command, flag, value):
    result = list(command)
    result[result.index(flag) + 1] = value
    return result


def append_pair(command, flag, value):
    return [*command, flag, value]


def complete(path):
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def assert_target_excluded(command):
    inputs = [command[index + 1] for index, item in enumerate(command[:-1])
              if item in {"--source", "--initial-checkpoint"}]
    if not inputs or any("pems03" in item.lower() for item in inputs):
        raise ValueError("PEMS03 source/initialization leakage")


def guarded(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")),
        "--", *command,
    ], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "PEMS03", "horizon": 192,
        "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE),
        "arms": list(ARMS),
        "question": "Does a shared train-only Channel-Patch projection head improve target-excluded SSL transfer?",
        "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24,
                           "d_ff": 48, "mamba_scan_count": 1, "target_core_frozen": True,
                           "channel_patch_relation_fraction": .05},
        "target_excluded_from_source_and_initialization": True,
        "only_difference": "Channel-Patch InfoNCE applies identity or one shared D-to-D MLP projector; prediction, source data, ACF pairs, core, and inference path are fixed.",
        "screen_rule": "expand MLP only if validation MSE is at least 0.25 percent lower than identity",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s6060_channel_patch"
    for arm in ARMS:
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
        finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable
        finetune[0] = sys.executable
        pre = append_pair(pre, "--channel-patch-projector", arm)
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        pre = replace(pre, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        finetune = replace(finetune, "--seed", str(SEED))
        assert_target_excluded(pre)
        if "--evaluate-test" in pre + finetune:
            raise ValueError("projector screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
            guarded(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guarded(finetune, prefix, "finetune")


if __name__ == "__main__":
    main()
