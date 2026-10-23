#!/usr/bin/env python3
"""Independent Weather evaluation of the PEMS/METR-selected channel-patch SSL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006"
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006"
SEEDS = (5991, 5992, 5993, 5994, 5995)


def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out
def complete(path: Path) -> bool: return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Weather", "horizon": 192, "few_shot_fraction": .1, "seeds": list(SEEDS),
        "question": "Does the PEMS/METR-selected channelwise patch ACF representation mechanism transfer to independent Weather?",
        "selection_boundary": "channel-patch loss and fraction selected using PEMS-BAY/METR-LA only; Weather is not used in selection",
        "reference_controls": str(REFERENCE),
        "fixed_contract": {"mixer": "mamba1", "mamba_ts_variant": "v3", "patch": "24/8", "d_model": 24,
                           "d_ff": 48, "target_core_frozen": True, "mamba_scan_count": 1,
                           "channel_patch_relation_fraction": .05, "inference_change": "none"},
    }, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        reference = REFERENCE / f"s{seed}_active"
        prefix = OUTPUT / f"s{seed}_channel_patch"
        pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
        ft = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable; ft[0] = sys.executable
        pre += ["--channel-patch-relation-fraction", ".05"]
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
        ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
        if "--evaluate-test" in pre + ft: raise ValueError("Weather channel-patch replication cannot read test rows")
        pre_event = prefix.with_name(prefix.name + "_pretrain_event.json"); ft_event = prefix.with_name(prefix.name + "_finetune_event.json")
        if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(pre_event)): guard(pre, prefix, "pretrain")
        if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(ft_event)): guard(ft, prefix, "finetune")
    return 0


if __name__ == "__main__": raise SystemExit(main())
