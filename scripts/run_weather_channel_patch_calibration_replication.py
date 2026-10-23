#!/usr/bin/env python3
"""Independent Weather target-calibration Channel-Patch causal replication."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006"
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
SEEDS = (6073, 6074, 6075)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def append_pair(command, flag, value): return [*command, flag, value]
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def check_pretrain(command, channel_patch):
    required = {"--calibration-source": "Weather:dataset/weather/weather.csv:csv", "--calibration-fraction": ".1", "--calibration-relation-loss-fraction": ".1", "--channel-patch-relation-fraction": ".05" if channel_patch else "0", "--calibration-channel-patch-relation-fraction": ".05" if channel_patch else "0"}
    for flag, value in required.items():
        if flag not in command or command[command.index(flag) + 1] != value: raise ValueError(f"invalid {flag}")
    if "--evaluate-test" in command: raise ValueError("Weather replication cannot read test")


def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seeds": list(SEEDS), "reference": str(REFERENCE), "question": "Does source-and-target Channel-Patch SSL improve target-prefix calibrated Weather transfer on independent seeds?", "arms": ["forecast_calibration", "channel_patch_calibration"], "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24, "d_ff": 48, "mamba_scan_count": 1, "target_core_frozen": True, "source_forecast_relation_fraction": .1, "calibration_forecast_relation_fraction": .1, "source_and_calibration_channel_patch_fraction": .05, "calibration_fraction": .1}, "target_calibration_boundary": "only earliest 60% x 10% Weather rows; validation and test excluded", "promotion_rule": "external baseline/robustness only if Channel-Patch wins all three pairs and one-SE ratio upper is at most one", "no_result_combination": "fresh replication seeds are reported separately from exploratory 6070/6072 screens"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s5991_active"
    for seed in SEEDS:
        for arm in ("forecast_calibration", "channel_patch_calibration"):
            prefix = OUTPUT / f"s{seed}_{arm}"; pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
            pre[0] = sys.executable; finetune[0] = sys.executable
            pre = append_pair(pre, "--calibration-source", "Weather:dataset/weather/weather.csv:csv"); pre = append_pair(pre, "--calibration-fraction", ".1"); pre = append_pair(pre, "--calibration-relation-loss-fraction", ".1")
            fraction = ".05" if arm == "channel_patch_calibration" else "0"
            pre = append_pair(pre, "--channel-patch-relation-fraction", fraction); pre = append_pair(pre, "--calibration-channel-patch-relation-fraction", fraction)
            pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
            finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(seed))
            check_pretrain(pre, arm == "channel_patch_calibration")
            if "--evaluate-test" in finetune: raise ValueError("Weather replication cannot read test")
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")


if __name__ == "__main__": main()
