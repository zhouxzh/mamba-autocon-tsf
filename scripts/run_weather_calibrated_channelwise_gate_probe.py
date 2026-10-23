#!/usr/bin/env python3
"""One-seed Weather calibrated Channel-Patch target-channel gate screen."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
TIMEMIXER = ROOT / "artifacts/exploratory/official_timemixer_weather_channel_patch_calibration_20261007"
OUTPUT = ROOT / "artifacts/exploratory/weather_calibrated_channelwise_gate_probe_20261007"
SEED = 6073


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = REFERENCE / f"s{SEED}_channel_patch_calibration_pretrain.pt"; event = REFERENCE / f"s{SEED}_channel_patch_calibration_finetune_event.json"
    if not source.is_file(): raise FileNotFoundError(source)
    command = load(event)["command"]; command[0] = sys.executable; prefix = OUTPUT / f"s{SEED}_channelwise_gate"
    command = replace(command, "--source-checkpoint", str(source)); command = replace(command, "--output", str(prefix.with_suffix(".pt"))); command.append("--mamba-channelwise-residual-scale")
    if "--evaluate-test" in command: raise ValueError("Weather gate screen cannot read test")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seed": SEED, "source_pretraining": str(source), "controls": {"scalar_channel_patch": str(REFERENCE / f"s{SEED}_channel_patch_calibration_frozen_core.pt"), "official_timemixer": str(TIMEMIXER / f"weather_h192_s{SEED}_official_timemixer.json")}, "mechanism": "zero-initialized target-channel Mamba correction gate after frozen calibrated Channel-Patch core", "mamba_scan_count": 1, "shared_encoder_change": False, "screen_rule": "expand only if channelwise gate is at least 0.25 percent below scalar Channel-Patch and below same-seed TimeMixer"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


if __name__ == "__main__": main()
