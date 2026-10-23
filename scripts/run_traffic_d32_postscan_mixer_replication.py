#!/usr/bin/env python3
"""Fresh-seed validation replication for the transferable post-scan mixer."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/traffic_d32_postscan_mixer_probe_rerun_20261007"
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_postscan_mixer_replication_20261007"
SEEDS = (6099, 6100, 6101)
ARMS = ("forecast_calibration", "channel_patch", "postscan_channel_patch")


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    result = list(command); result[result.index(flag) + 1] = value; return result
def append_once(command, *items): return command if items[0] in command else [*command, *items]
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None
def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Traffic", "horizon": 96, "few_shot_fraction": .25, "seeds": list(SEEDS), "reference": str(REFERENCE), "arms": list(ARMS), "question": "Does the transferable post-scan FiLM Channel-Patch candidate replicate on fresh Traffic seeds?", "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 32, "d_ff": 64, "mamba_scan_count": 1, "target_core_frozen": True, "source_and_calibration_channel_patch_fraction": .05, "postscan_mixer_rank": 4}, "no_result_combination": "fresh seeds are disjoint from development seed 6098", "promotion_rule": "postscan must win all three fresh pairs versus forecast and ordinary Channel-Patch, with one-SE ratio upper at most one for each contrast"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ARMS:
            reference = REFERENCE / f"s6098_{arm}"
            prefix = OUTPUT / f"s{seed}_{arm}"
            pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
            finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
            pre[0] = finetune[0] = sys.executable
            pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
            finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(seed))
            if arm == "postscan_channel_patch":
                pre = append_once(pre, "--mamba-ts-shared-postscan-mixer", "--mamba-ts-shared-postscan-mixer-rank", "4")
                finetune = append_once(finetune, "--mamba-ts-shared-postscan-mixer", "--mamba-ts-shared-postscan-mixer-rank", "4")
            if "--evaluate-test" in pre + finetune: raise ValueError("replication cannot read test")
            if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
            if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")


if __name__ == "__main__": main()
