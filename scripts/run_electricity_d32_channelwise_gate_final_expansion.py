#!/usr/bin/env python3
"""Final capped five-seed Electricity d32 channelwise-gate precision expansion."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_REFERENCE = ROOT / "artifacts/exploratory/electricity_d32_channel_patch_probe_20261006"
GATE_REFERENCE = ROOT / "artifacts/exploratory/electricity_d32_channelwise_residual_gate_probe_20261006"
PATCH_REFERENCE = ROOT / "artifacts/exploratory/official_patchtst_electricity_d32_channelwise_gate_20261006"
OUTPUT = ROOT / "artifacts/exploratory/electricity_d32_channelwise_gate_final_expansion_20261007"
SEEDS = (6094, 6095, 6096, 6097, 6098)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None
def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "fresh_seeds": list(SEEDS), "prior_seeds": [6047, 6048, 6049, 6050, 6051], "question": "Final capped precision expansion for the five-seed d32/24 channelwise-gate versus official PatchTST near-tie.", "architecture": {"mixer": "mamba1", "patch": "24/24", "d_model": 32, "d_ff": 64, "mamba_scan_count": 1, "channelwise_residual_scale": True}, "training_contract": {"candidate_batch_size": 4, "candidate_epochs": 5, "patchtst_batch_size": 4, "patchtst_epochs": 5, "max_train_windows": 64}, "same_seed_pairing": True, "aggregation_rule": "combine exactly five fresh seeds with the prior five matched pairs; this is the final Electricity d32 gate precision expansion regardless of outcome", "no_test_reader": True, "stop_rule": "no further Electricity d32 channelwise-gate seed expansion after this artifact"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        pre_prefix = OUTPUT / f"s{seed}_channel_patch"
        pre = load(PRETRAIN_REFERENCE / "s6043_channel_patch_pretrain_event.json")["command"]; pre[0] = sys.executable; pre = replace(pre, "--output", str(pre_prefix.with_name(pre_prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
        if "--evaluate-test" in pre: raise ValueError("candidate cannot read test")
        if not (args.resume and pre_prefix.with_name(pre_prefix.name + "_pretrain.pt").is_file() and complete(pre_prefix.with_name(pre_prefix.name + "_pretrain_event.json"))): guard(pre, pre_prefix, "pretrain")
        gate_prefix = OUTPUT / f"s{seed}_channelwise_gate"
        gate = load(GATE_REFERENCE / "s6044_channelwise_gate_event.json")["command"]; gate[0] = sys.executable; gate = replace(gate, "--source-checkpoint", str(pre_prefix.with_name(pre_prefix.name + "_pretrain.pt"))); gate = replace(gate, "--output", str(gate_prefix.with_suffix(".pt"))); gate = replace(gate, "--seed", str(seed))
        if "--evaluate-test" in gate: raise ValueError("candidate cannot read test")
        if not (args.resume and gate_prefix.with_suffix(".pt").is_file() and complete(gate_prefix.with_name(gate_prefix.name + "_finetune_event.json"))): guard(gate, gate_prefix, "finetune")
        patch_prefix = OUTPUT / f"electricity_h96_s{seed}_official_patchtst"
        patch = load(PATCH_REFERENCE / "electricity_h96_s6047_official_patchtst_event.json")["command"]; patch[0] = sys.executable; patch = replace(patch, "--output", str(patch_prefix.with_suffix(".pt"))); patch = replace(patch, "--seed", str(seed))
        if "--evaluate-test" in patch: raise ValueError("PatchTST cannot read test")
        if not (args.resume and patch_prefix.with_suffix(".pt").is_file() and complete(patch_prefix.with_name(patch_prefix.name + "_event.json"))): guard(patch, patch_prefix, "patch")


if __name__ == "__main__": main()
