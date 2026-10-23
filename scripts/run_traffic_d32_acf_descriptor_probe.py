#!/usr/bin/env python3
"""Validation-only Traffic screen for train-prefix ACF-conditioned patch tokens."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_probe_20261007"
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_probe_20261007"
SEED = 6095
ARMS = ("forecast_calibration", "channel_patch", "descriptor_channel_patch")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace(command, flag, value):
    result = list(command)
    result[result.index(flag) + 1] = value
    return result


def append_once(command, *items):
    return command if items[0] in command else [*command, *items]


def append_pair(command, flag, value):
    return [*command, flag, value]


def complete(path):
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")),
        "--", *command,
    ], cwd=ROOT, check=True)


def require_contract(command, fraction, descriptor):
    required = {
        "--channel-patch-relation-fraction": fraction,
        "--calibration-channel-patch-relation-fraction": fraction,
        "--calibration-source": "Traffic:dataset/traffic/traffic.csv:csv",
        "--calibration-fraction": ".25",
        "--calibration-relation-loss-fraction": ".3",
    }
    for flag, value in required.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError(f"invalid {flag}")
    if ("--mamba-ts-acf-channel-descriptor" in command) != descriptor:
        raise ValueError("invalid ACF descriptor flag")
    if "--evaluate-test" in command:
        raise ValueError("Traffic ACF descriptor screen cannot read test")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Traffic", "horizon": 96,
        "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE),
        "arms": list(ARMS),
        "question": "Does a fixed train-prefix ACF channel descriptor make shared patch representations transferable on high-dimensional Traffic?",
        "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 32,
                           "d_ff": 64, "mamba_scan_count": 1, "target_core_frozen": True,
                           "source_and_calibration_channel_patch_fraction": .05,
                           "source_forecast_relation_fraction": .1,
                           "calibration_forecast_relation_fraction": .3},
        "target_calibration_boundary": "only earliest 60% x 25% Traffic rows; validation and test excluded",
        "only_difference": "descriptor candidate adds a zero-gated shared 6-to-D projection of per-channel train-prefix ACF summaries after patch embedding; descriptors are rebuilt from each role's train prefix and not transferred across domains",
        "screen_rule": "expand descriptor Channel-Patch only if validation MSE is at least 0.25 percent below both forecast calibration and ordinary Channel-Patch",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s6090_channel_patch_calibration"
    for arm in ARMS:
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
        finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable
        finetune[0] = sys.executable
        fraction = ".05" if arm != "forecast_calibration" else "0"
        descriptor = arm == "descriptor_channel_patch"
        pre = replace(pre, "--channel-patch-relation-fraction", fraction)
        pre = replace(pre, "--calibration-channel-patch-relation-fraction", fraction)
        if descriptor:
            pre = append_once(pre, "--mamba-ts-acf-channel-descriptor")
            finetune = append_once(finetune, "--mamba-ts-acf-channel-descriptor")
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        pre = replace(pre, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        finetune = replace(finetune, "--seed", str(SEED))
        require_contract(pre, fraction, descriptor)
        if "--evaluate-test" in finetune:
            raise ValueError("Traffic ACF descriptor screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
            guard(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guard(finetune, prefix, "finetune")


if __name__ == "__main__":
    main()
