#!/usr/bin/env python3
"""Weather target-only versus source-and-target Channel-Patch SSL screen."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006"
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_locality_probe_20261007"
SEED = 6072
ARMS = {"forecast_calibration": ("0", "0"), "source_and_target_channel_patch": (".05", ".05"), "target_only_channel_patch": ("0", ".05")}


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def append_pair(command, flag, value): return [*command, flag, value]
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def calibration_contract(command, expected_source_fraction, expected_calibration_fraction):
    expected = {"--calibration-source": "Weather:dataset/weather/weather.csv:csv", "--calibration-fraction": ".1", "--calibration-relation-loss-fraction": ".1", "--channel-patch-relation-fraction": expected_source_fraction, "--calibration-channel-patch-relation-fraction": expected_calibration_fraction}
    for flag, value in expected.items():
        if flag not in command or command[command.index(flag) + 1] != value: raise ValueError(f"invalid {flag}")
    if "--evaluate-test" in command: raise ValueError("locality screen cannot read test")


def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seed": SEED, "reference": str(REFERENCE), "arms": ARMS, "question": "Does restricting full patch-trajectory ACF InfoNCE to the Weather train-prefix calibration reduce cross-domain representation interference?", "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24, "d_ff": 48, "mamba_scan_count": 1, "target_core_frozen": True, "source_forecast_relation_fraction": .1, "calibration_forecast_relation_fraction": .1, "calibration_fraction": .1}, "target_calibration_boundary": "only earliest 60% x 10% Weather rows; validation and test excluded", "screen_rule": "expand target-only Channel-Patch only if it is at least 0.25 percent below both calibrated forecast and source-and-target Channel-Patch"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s5991_active"
    for arm, (source_fraction, calibration_fraction) in ARMS.items():
        prefix = OUTPUT / f"s{SEED}_{arm}"; pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable; finetune[0] = sys.executable
        pre = append_pair(pre, "--calibration-source", "Weather:dataset/weather/weather.csv:csv"); pre = append_pair(pre, "--calibration-fraction", ".1"); pre = append_pair(pre, "--calibration-relation-loss-fraction", ".1")
        pre = append_pair(pre, "--channel-patch-relation-fraction", source_fraction); pre = append_pair(pre, "--calibration-channel-patch-relation-fraction", calibration_fraction)
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(SEED))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(SEED))
        calibration_contract(pre, source_fraction, calibration_fraction)
        if "--evaluate-test" in finetune: raise ValueError("locality screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")


if __name__ == "__main__": main()
