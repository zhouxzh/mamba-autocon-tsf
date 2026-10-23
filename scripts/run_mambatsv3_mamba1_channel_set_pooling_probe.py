#!/usr/bin/env python3
"""Validation-only learned channel-set pooling probe for frozen Mamba-1 transfer."""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_probe_20261006"
SEED = 5984
TARGETS = {
    "pems_bay": (ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006", 5960, "PEMS-BAY"),
    "metr_la": (ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006", 5970, "METR-LA"),
}

def load(path): return json.loads(path.read_text(encoding="utf-8"))
def replace(command, flag, value):
    out = list(command); out[out.index(flag) + 1] = value; return out
def append(command, flag): return list(command) if flag in command else list(command) + [flag]

def commands(target, arm):
    root, seed, _ = TARGETS[target]; reference = root / f"s{seed}_{arm}"
    prefix = OUTPUT / target / f"s{SEED}_{arm}"
    pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; pre[0] = sys.executable
    pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(SEED)); pre = append(pre, "--mamba-ts-channel-set-pooling")
    ft = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]; ft[0] = sys.executable
    ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); ft = replace(ft, "--seed", str(SEED)); ft = append(ft, "--mamba-ts-channel-set-pooling")
    if "--evaluate-test" in pre + ft: raise ValueError("channel-set probe cannot read test rows")
    return pre, ft, prefix

def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "seed": SEED, "targets": {k: v[2] for k, v in TARGETS.items()}, "question": "Does learned channel-set pooling improve channel-count-agnostic ACF transfer over the fixed shared encoder?", "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8, "few_shot_fraction": .1, "relation_loss_fraction": .1, "target_core_frozen": True, "mamba_scan_count": 1}, "active_change": "shared D-to-1 channel attention plus weighted mean/std patch context before the existing scan"}, indent=2) + "\n", encoding="utf-8")
    for target in TARGETS:
        (OUTPUT / target).mkdir(parents=True, exist_ok=True)
        for arm in ("active", "null"):
            pre, ft, prefix = commands(target, arm)
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(ft, prefix, "finetune")
    return 0
if __name__ == "__main__": raise SystemExit(main())
