#!/usr/bin/env python3
"""Strict target-excluded PEMS03 H192/25% channel-patch replication."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay"
PATCH_REFERENCE = ROOT / "artifacts/exploratory/official_patchtst_channel_patch_replication_20261006/pems_bay"
INITIAL_REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems03_external_confirmation_20261003/s5570_active_pretrain_event.json"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
PATCH_OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_pems03_channel_patch_25pct_external_replication_20261007"
SEEDS = (6060, 6061, 6062)
ARMS = ("null", "forecast", "channel_patch")
SOURCES = ("PEMS04:dataset/PEMS04/PEMS04.npz:npz", "PEMS08:dataset/PEMS08/PEMS08.npz:npz", "PEMS07:dataset/PEMS07/PEMS07.npz:npz")


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated


def values(command, flag):
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == flag]


def replace_repeated(command, flag, replacements):
    updated = list(command); positions = [index for index, item in enumerate(updated[:-1]) if item == flag]
    if len(positions) != len(replacements): raise ValueError(f"{flag} count mismatch")
    for position, value in zip(positions, replacements): updated[position + 1] = value
    return updated


def set_single(command, flag, value):
    found = values(command, flag)
    if len(found) > 1: raise ValueError(f"{flag} must occur at most once")
    return replace(command, flag, value) if found else [*command, flag, value]


def replace_sources(command):
    updated = []; index = 0; seen = 0
    while index < len(command):
        if command[index] == "--source":
            updated.extend(("--source", SOURCES[seen])); seen += 1; index += 2
        else:
            updated.append(command[index]); index += 1
    if seen != len(SOURCES): raise ValueError("source count mismatch")
    return updated


def complete(path):
    return Path(path).is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def assert_target_excluded(command):
    inputs = values(command, "--source") + values(command, "--initial-checkpoint")
    if not inputs or any("pems03" in item.lower() for item in inputs):
        raise ValueError("PEMS03 is forbidden in external source or initialization inputs")


def guard(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True); PATCH_OUTPUT.mkdir(parents=True, exist_ok=True)
    initial = values(load(INITIAL_REFERENCE)["command"], "--initial-checkpoint")
    if len(initial) != 2 or any("pems03" in item.lower() for item in initial):
        raise ValueError("target-excluded initial checkpoints unavailable")
    plan = {
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "PEMS03", "horizon": 192, "few_shot_fraction": 0.25, "seeds": list(SEEDS),
        "question": "Does current 24/8 channel-patch SSL transfer to PEMS03 at 25 percent labels with target excluded from sources and initialization?",
        "source_pool": ["PEMS04", "PEMS08", "PEMS07"], "initialization_sources": ["PEMS04", "PEMS08"],
        "initialization_checkpoints": initial, "target_excluded_from_source_and_initialization": True,
        "arms": [*ARMS, "official_patchtst"],
        "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24, "d_ff": 48, "target_core_frozen": True, "mamba_scan_count": 1, "channel_patch_relation_fraction": 0.05},
    }
    for root in (OUTPUT, PATCH_OUTPUT):
        (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ARMS:
            source_arm = "null" if arm == "null" else "channel_patch" if arm == "channel_patch" else "forecast"
            ref = REFERENCE / f"s6026_{source_arm}"; prefix = OUTPUT / f"s{seed}_{arm}"
            pre = load(ref.with_name(ref.name + "_pretrain_event.json"))["command"]
            finetune = load(ref.with_name(ref.name + "_finetune_event.json"))["command"]
            pre[0] = sys.executable; finetune[0] = sys.executable
            pre = replace_sources(pre); pre = replace_repeated(pre, "--initial-checkpoint", initial)
            pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
            finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
            finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(seed))
            finetune = replace(finetune, "--dataset", "dataset/PEMS03/PEMS03.npz"); finetune = replace(finetune, "--few-shot-fraction", ".25")
            if arm == "channel_patch": pre = set_single(pre, "--channel-patch-relation-fraction", ".05")
            assert_target_excluded(pre)
            if "--evaluate-test" in pre + finetune: raise ValueError("PEMS03 replication cannot read test")
            if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
            if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")
        prefix = PATCH_OUTPUT / f"pems03_h192_s{seed}_official_patchtst"
        patch = load(PATCH_REFERENCE / "pems_bay_h192_s6026_official_patchtst_event.json")["command"]
        patch[0] = sys.executable; patch = replace(patch, "--output", str(prefix.with_suffix(".pt"))); patch = replace(patch, "--seed", str(seed))
        patch = replace(patch, "--dataset", "dataset/PEMS03/PEMS03.npz"); patch = replace(patch, "--few-shot-fraction", ".25")
        if "--evaluate-test" in patch: raise ValueError("PatchTST cannot read test")
        if not (prefix.with_suffix(".pt").is_file() and complete(prefix.with_name(prefix.name + "_event.json"))): guard(patch, prefix, "patch")


if __name__ == "__main__": main()
