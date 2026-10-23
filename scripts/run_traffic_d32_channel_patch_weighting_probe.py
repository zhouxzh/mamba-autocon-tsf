#!/usr/bin/env python3
"""Validation-only Traffic screen for ACF-strength Channel-Patch aggregation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_probe_20261007"
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_weighting_probe_20261007"
SEED = 6091
ARMS = ("uniform", "acf_strength")


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


def require_contract(command, weighting):
    required = {
        "--channel-patch-relation-fraction": ".05",
        "--calibration-channel-patch-relation-fraction": ".05",
        "--channel-patch-channel-weighting": weighting,
        "--calibration-source": "Traffic:dataset/traffic/traffic.csv:csv",
        "--calibration-fraction": ".25",
        "--calibration-relation-loss-fraction": ".3",
    }
    for flag, value in required.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError(f"invalid {flag}")
    if "--evaluate-test" in command:
        raise ValueError("Traffic weighting screen cannot read test")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Traffic", "horizon": 96,
        "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE),
        "arms": list(ARMS),
        "question": "Does train-prefix ACF-strength channel aggregation improve high-dimensional Channel-Patch SSL?",
        "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 32,
                           "d_ff": 64, "mamba_scan_count": 1, "target_core_frozen": True,
                           "source_and_calibration_channel_patch_fraction": .05,
                           "source_forecast_relation_fraction": .1,
                           "calibration_forecast_relation_fraction": .3},
        "target_calibration_boundary": "only earliest 60% x 25% Traffic rows; validation and test excluded",
        "only_difference": "uniform averaging of active channel losses versus normalized train-prefix selected-peak ACF-strength averaging; pairs, model, losses, sources, and inference graph are fixed",
        "screen_rule": "expand acf_strength only if validation MSE is at least 0.25 percent lower than uniform Channel-Patch",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s6090_channel_patch_calibration"
    for arm in ARMS:
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
        finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable
        finetune[0] = sys.executable
        weighting = "acf_strength" if arm == "acf_strength" else "none"
        pre = append_pair(pre, "--channel-patch-channel-weighting", weighting)
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        pre = replace(pre, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        finetune = replace(finetune, "--seed", str(SEED))
        require_contract(pre, weighting)
        if "--evaluate-test" in finetune:
            raise ValueError("Traffic weighting screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
            guard(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file()
                and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guard(finetune, prefix, "finetune")


if __name__ == "__main__":
    main()
