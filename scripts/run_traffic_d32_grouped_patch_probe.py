#!/usr/bin/env python3
"""Validation-only Traffic screen for train-prefix ACF-grouped patch SSL."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_probe_20261007"
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_grouped_patch_probe_20261007"
SEED = 6094
ARMS = ("forecast_calibration", "grouped_patch_calibration")


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


def guard(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")),
        "--", *command,
    ], cwd=ROOT, check=True)


def require_contract(command, grouped_fraction):
    required = {
        "--channel-patch-relation-fraction": "0",
        "--calibration-channel-patch-relation-fraction": "0",
        "--grouped-patch-relation-fraction": grouped_fraction,
        "--calibration-grouped-patch-relation-fraction": grouped_fraction,
        "--grouped-patch-group-count": "4",
        "--calibration-source": "Traffic:dataset/traffic/traffic.csv:csv",
        "--calibration-fraction": ".25",
        "--calibration-relation-loss-fraction": ".3",
    }
    for flag, value in required.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError(f"invalid {flag}")
    if "--evaluate-test" in command:
        raise ValueError("Traffic grouped-patch screen cannot read test")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Traffic", "horizon": 96,
        "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE),
        "arms": list(ARMS),
        "question": "Can fixed train-prefix ACF groups produce a transferable high-dimensional patch representation?",
        "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 32,
                           "d_ff": 64, "mamba_scan_count": 1, "target_core_frozen": True,
                           "source_forecast_relation_fraction": .1,
                           "calibration_forecast_relation_fraction": .3,
                           "grouped_patch_group_count": 4},
        "target_calibration_boundary": "only earliest 60% x 25% Traffic rows; validation and test excluded",
        "only_difference": "candidate replaces per-channel Channel-Patch with four fixed train-prefix dominant-lag groups, group-mean patch states, and group ACF positives; encoder, forecast loss, data, and inference graph are fixed",
        "screen_rule": "expand grouped-patch only if validation MSE is at least 0.25 percent lower than forecast calibration",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s6090_channel_patch_calibration"
    for arm in ARMS:
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
        finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable
        finetune[0] = sys.executable
        fraction = ".05" if arm == "grouped_patch_calibration" else "0"
        pre = replace(pre, "--channel-patch-relation-fraction", "0")
        pre = replace(pre, "--calibration-channel-patch-relation-fraction", "0")
        pre = append_pair(pre, "--grouped-patch-relation-fraction", fraction)
        pre = append_pair(pre, "--calibration-grouped-patch-relation-fraction", fraction)
        pre = append_pair(pre, "--grouped-patch-group-count", "4")
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        pre = replace(pre, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        finetune = replace(finetune, "--seed", str(SEED))
        require_contract(pre, fraction)
        if "--evaluate-test" in finetune:
            raise ValueError("Traffic grouped-patch screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
            guard(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guard(finetune, prefix, "finetune")


if __name__ == "__main__":
    main()
