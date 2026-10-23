#!/usr/bin/env python3
"""Frozen three-seed validation replication for learned channel-set pooling."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_probe_20261006"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_replication_20261006"
SEEDS = (5985, 5986, 5987)
TARGETS = ("pems_bay", "metr_la")
ARMS = ("pool_active", "pool_null", "plain_active")

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def replace(c, flag, value):
    out = list(c); out[out.index(flag) + 1] = value; return out
def remove(c, flag):
    out = list(c)
    while flag in out: out.remove(flag)
    return out
def commands(target, seed, arm):
    source_arm = "active" if arm != "pool_null" else "null"
    reference = REFERENCE / target / f"s5984_{source_arm}"
    prefix = OUTPUT / target / f"s{seed}_{arm}"
    pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; pre[0] = sys.executable
    ft = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]; ft[0] = sys.executable
    if arm == "plain_active":
        pre, ft = remove(pre, "--mamba-ts-channel-set-pooling"), remove(ft, "--mamba-ts-channel-set-pooling")
    pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
    ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); ft = replace(ft, "--seed", str(seed))
    if "--evaluate-test" in pre + ft: raise ValueError("replication cannot read test rows")
    return pre, ft, prefix
def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)
def complete(p): return p.is_file() and load(p).get("status") == "completed" and load(p).get("trigger") is None
def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "seeds": list(SEEDS), "targets": list(TARGETS), "arms": list(ARMS), "frozen_after_probe": str(REFERENCE), "question": "Does learned channel-set pooling replicate against both exact null and ordinary active InfoNCE?", "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8, "few_shot_fraction": .1, "relation_loss_fraction": .1, "target_core_frozen": True, "mamba_scan_count": 1}}, indent=2) + "\n", encoding="utf-8")
    for target in TARGETS:
        (OUTPUT / target).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            for arm in ARMS:
                pre, ft, prefix = commands(target, seed, arm)
                if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
                if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(ft, prefix, "finetune")
    return 0
if __name__ == "__main__": raise SystemExit(main())
