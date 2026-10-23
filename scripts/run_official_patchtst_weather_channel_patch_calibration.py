#!/usr/bin/env python3
"""Official PatchTST comparator for Weather calibrated Channel-Patch seeds."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/official_patchtst_mamba1_crossdomain_replication_20261006/weather"
OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_weather_channel_patch_calibration_20261007"
SEEDS = (6073, 6074, 6075)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(command, prefix):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seeds": list(SEEDS), "candidate": "Weather target-prefix calibrated Channel-Patch", "official": "PatchTST commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9", "same_seed_pairing": True, "no_test_reader": True}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        reference = REFERENCE / "weather_h192_s5930_official_patchtst_event.json"; command = load(reference)["command"]; command[0] = sys.executable; prefix = OUTPUT / f"weather_h192_s{seed}_official_patchtst"; command = replace(command, "--output", str(prefix.with_suffix(".pt"))); command = replace(command, "--seed", str(seed))
        if "--evaluate-test" in command: raise ValueError("PatchTST comparator cannot read test")
        if not (prefix.with_suffix(".pt").is_file() and complete(prefix.with_name(prefix.name + "_event.json"))): guard(command, prefix)


if __name__ == "__main__": main()
