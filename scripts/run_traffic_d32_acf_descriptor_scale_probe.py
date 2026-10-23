#!/usr/bin/env python3
"""Validation-only Traffic screen for target-tuned ACF descriptor gate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_probe_20261007"
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_scale_probe_20261007"
SEED = 6096


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    result = list(command); result[result.index(flag) + 1] = value; return result
def append_once(command, *items): return command if items[0] in command else [*command, *items]
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")),
        "--", *command,
    ], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Traffic", "horizon": 96,
        "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE),
        "arms": ["forecast_calibration", "descriptor_frozen", "descriptor_target_scale"],
        "question": "Can a target-local scalar calibrate a transferred train-prefix ACF descriptor without unfreezing the one-scan core?",
        "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 32,
                           "d_ff": 64, "mamba_scan_count": 1, "target_core_frozen": True,
                           "source_and_calibration_channel_patch_fraction": .05,
                           "source_forecast_relation_fraction": .1,
                           "calibration_forecast_relation_fraction": .3},
        "target_calibration_boundary": "only earliest 60% x 25% Traffic rows; validation and test excluded",
        "only_difference": "descriptor_target_scale makes only the transferred ACF descriptor ReZero scalar trainable during target adaptation; projection, encoder, Mamba scan, prediction head protocol, and target descriptor buffer are unchanged",
        "screen_rule": "expand only if target-scale tuning is at least 0.25 percent below both frozen descriptor and forecast calibration",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    ref = REFERENCE / "s6095_descriptor_channel_patch"
    pre = load(ref.with_name(ref.name + "_pretrain_event.json"))["command"]
    pre[0] = sys.executable
    pre = replace(pre, "--output", str(OUTPUT / f"s{SEED}_descriptor_pretrain.pt"))
    pre = replace(pre, "--seed", str(SEED))
    pre = append_once(pre, "--mamba-ts-acf-channel-descriptor")
    if "--evaluate-test" in pre: raise ValueError("descriptor pretraining cannot read test")
    descriptor_prefix = OUTPUT / f"s{SEED}_descriptor"
    if not ((OUTPUT / f"s{SEED}_descriptor_pretrain.pt").is_file() and complete(descriptor_prefix.with_name(descriptor_prefix.name + "_pretrain_event.json"))):
        guard(pre, descriptor_prefix, "pretrain")
    for arm, tuned in (("descriptor_frozen", False), ("descriptor_target_scale", True)):
        finetune = load(ref.with_name(ref.name + "_finetune_event.json"))["command"]
        finetune[0] = sys.executable
        prefix = OUTPUT / f"s{SEED}_{arm}"
        finetune = replace(finetune, "--source-checkpoint", str(OUTPUT / f"s{SEED}_descriptor_pretrain.pt"))
        finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        finetune = replace(finetune, "--seed", str(SEED))
        finetune = append_once(finetune, "--mamba-ts-acf-channel-descriptor")
        if tuned: finetune = append_once(finetune, "--mamba-ts-acf-channel-descriptor-tune-scale")
        if "--evaluate-test" in finetune: raise ValueError("descriptor target adaptation cannot read test")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guard(finetune, prefix, "finetune")
    forecast_ref = REFERENCE / "s6095_forecast_calibration"
    pre = load(forecast_ref.with_name(forecast_ref.name + "_pretrain_event.json"))["command"]
    ft = load(forecast_ref.with_name(forecast_ref.name + "_finetune_event.json"))["command"]
    pre[0] = ft[0] = sys.executable
    prefix = OUTPUT / f"s{SEED}_forecast_calibration"
    pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(SEED))
    ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); ft = replace(ft, "--seed", str(SEED))
    if "--evaluate-test" in pre + ft: raise ValueError("forecast control cannot read test")
    if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
    if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(ft, prefix, "finetune")


if __name__ == "__main__": main()
