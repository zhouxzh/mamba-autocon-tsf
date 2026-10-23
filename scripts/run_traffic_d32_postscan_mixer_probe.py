#!/usr/bin/env python3
"""Validation-only Traffic gate for transferable post-scan channel mixing."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_probe_20261007"
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_postscan_mixer_probe_rerun_20261007"
SEED = 6098
ARMS = ("forecast_calibration", "channel_patch", "postscan_channel_patch")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace(command, flag, value):
    result = list(command)
    result[result.index(flag) + 1] = value
    return result


def append_once(command, *items):
    return command if items[0] in command else [*command, *items]


def complete(path):
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Traffic", "horizon": 96,
        "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE), "arms": list(ARMS),
        "question": "Can a channel-count-independent post-scan FiLM state mixer make Channel-Patch ACF pretraining transfer on high-dimensional Traffic?",
        "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 32,
                           "d_ff": 64, "mamba_scan_count": 1, "target_core_frozen": True,
                           "source_and_calibration_channel_patch_fraction": .05,
                           "source_forecast_relation_fraction": .1,
                           "calibration_forecast_relation_fraction": .3, "postscan_mixer_rank": 4},
        "target_calibration_boundary": "only earliest 60% x 25% Traffic rows; validation and test excluded",
        "only_difference": "candidate enables one D-width shared-state FiLM mixer after the existing Mamba scan; it has no channel-indexed parameter, no second scan, and both forecast/Channel-Patch consume the same mixed states",
        "screen_rule": "expand only if postscan Channel-Patch is at least 0.25 percent below both forecast calibration and ordinary Channel-Patch",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s6090_channel_patch_calibration"
    for arm in ARMS:
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pretrain = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
        finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pretrain[0] = finetune[0] = sys.executable
        fraction = ".05" if arm != "forecast_calibration" else "0"
        pretrain = replace(pretrain, "--channel-patch-relation-fraction", fraction)
        pretrain = replace(pretrain, "--calibration-channel-patch-relation-fraction", fraction)
        if arm == "postscan_channel_patch":
            pretrain = append_once(pretrain, "--mamba-ts-shared-postscan-mixer", "--mamba-ts-shared-postscan-mixer-rank", "4")
            finetune = append_once(finetune, "--mamba-ts-shared-postscan-mixer", "--mamba-ts-shared-postscan-mixer-rank", "4")
        pretrain = replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        pretrain = replace(pretrain, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        finetune = replace(finetune, "--seed", str(SEED))
        if "--evaluate-test" in pretrain + finetune:
            raise ValueError("postscan mixer probe cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
            guard(pretrain, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guard(finetune, prefix, "finetune")


if __name__ == "__main__":
    main()
