#!/usr/bin/env python3
"""Strict target-excluded PEMS03 screen for channel-patch SSL strength."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_strength_probe_20261007"
SEED = 6067
ARMS = {"fraction_0p05": ".05", "fraction_0p10": ".10"}


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def assert_target_excluded(command):
    inputs = [command[index + 1] for index, item in enumerate(command[:-1]) if item in {"--source", "--initial-checkpoint"}]
    if not inputs or any("pems03" in item.lower() for item in inputs): raise ValueError("PEMS03 source/initialization leakage")


def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "horizon": 192, "few_shot_fraction": .25, "seed": SEED, "reference": str(REFERENCE), "arms": ARMS, "question": "Does increasing prediction-scaled full channel-patch trajectory InfoNCE from 0.05 to 0.10 strengthen the learned representation?", "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24, "d_ff": 48, "mamba_scan_count": 1, "forecast_relation_fraction": .1, "target_core_frozen": True}, "target_excluded_from_source_and_initialization": True, "only_difference": "channel-patch-relation-fraction; all ACF positives and forecast InfoNCE settings fixed", "screen_rule": "expand 0.10 only if validation MSE is at least 0.25 percent lower than 0.05"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = REFERENCE / "s6060_channel_patch"
    for arm, fraction in ARMS.items():
        prefix = OUTPUT / f"s{SEED}_{arm}"; pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable; finetune[0] = sys.executable
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(SEED)); pre = replace(pre, "--channel-patch-relation-fraction", fraction)
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(SEED))
        assert_target_excluded(pre)
        if "--evaluate-test" in pre + finetune: raise ValueError("strength screen cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")


if __name__ == "__main__": main()
