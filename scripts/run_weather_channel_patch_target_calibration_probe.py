#!/usr/bin/env python3
"""Weather H192/10% target-prefix calibration screen for Channel-Patch SSL."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006"
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_target_calibration_probe_20261007"
SEED = 6070
ARMS = ("forecast_calibration", "channel_patch_calibration")


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def append_pair(command, flag, value): return [*command, flag, value]
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def target_prefix_only(command):
    if "--calibration-source" not in command or "--calibration-fraction" not in command: raise ValueError("Weather target calibration absent")
    if command[command.index("--calibration-source") + 1] != "Weather:dataset/weather/weather.csv:csv": raise ValueError("unexpected calibration source")
    if command[command.index("--calibration-fraction") + 1] != ".1": raise ValueError("unexpected calibration fraction")
    if "--evaluate-test" in command: raise ValueError("calibration screen cannot read test")


def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seed": SEED, "reference": str(REFERENCE), "arms": list(ARMS), "question": "Does train-prefix Weather SSL calibration recover Channel-Patch transfer under a frozen one-scan shared core?", "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24, "d_ff": 48, "mamba_scan_count": 1, "target_core_frozen": True, "source_relation_fraction": .1, "calibration_relation_fraction": .1, "calibration_fraction": .1}, "only_difference": "channel_patch arm adds full channel-patch trajectory InfoNCE fraction .05 during source and target-prefix calibration", "source_pool": ["METR-LA", "PEMS-BAY", "Electricity"], "target_calibration_boundary": "only earliest 60% x 10% Weather rows; validation and test excluded", "screen_rule": "expand only if channel-patch calibration is at least 0.25 percent below forecast calibration and below uncalibrated forecast reference at the same seed"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s5991_active"
    for arm in ARMS:
        prefix = OUTPUT / f"s{SEED}_{arm}"; pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable; finetune[0] = sys.executable
        pre = append_pair(pre, "--calibration-source", "Weather:dataset/weather/weather.csv:csv")
        pre = append_pair(pre, "--calibration-fraction", ".1")
        pre = append_pair(pre, "--calibration-relation-loss-fraction", ".1")
        if arm == "channel_patch_calibration": pre = append_pair(pre, "--channel-patch-relation-fraction", ".05")
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(SEED))
        target_prefix_only(pre)
        if "--evaluate-test" in finetune: raise ValueError("calibration screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")


if __name__ == "__main__": main()
