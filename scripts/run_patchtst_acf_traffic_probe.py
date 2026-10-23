#!/usr/bin/env python3
"""Validation-only PatchTST ACF-pretraining Traffic 25% gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/patchtst_acf_traffic_probe_20261007"
SEED = 6097
ARMS = {"forecast": ("0", "0"), "channel_patch": (".05", ".05")}


def complete(path):
    return path.is_file() and json.loads(path.read_text()).get("status") == "completed" and json.loads(path.read_text()).get("trigger") is None


def guard(command, prefix, stage):
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "Traffic", "horizon": 96, "few_shot_fraction": .25, "seed": SEED, "arms": list(ARMS), "question": "Does train-only ACF Channel-Patch pretraining transfer through a shared PatchTST encoder on high-dimensional Traffic?", "fixed_contract": {"backbone": "shared channel-independent PatchTST", "patch": "24/24", "d_model": 32, "d_ff": 64, "encoder_layers": 1, "target_encoder_frozen": True, "transformer_encoder_passes": 1}, "target_calibration_boundary": "only earliest 60% x 25% Traffic rows; validation and test excluded", "only_difference": "candidate enables source and target-calibration Channel-Patch InfoNCE at .05; forecast InfoNCE and next-patch prediction are fixed", "screen_rule": "expand only if Channel-Patch is at least 0.25 percent below forecast pretraining"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for arm, (source_fraction, calibration_fraction) in ARMS.items():
        prefix = OUTPUT / f"s{SEED}_{arm}"
        pretrain = [sys.executable, "scripts/pretrain_patchtst_acf_core.py", "--source", "Electricity:dataset/electricity/electricity.csv:csv", "--calibration-source", "Traffic:dataset/traffic/traffic.csv:csv", "--calibration-fraction", ".25", "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")), "--seq-len", "96", "--pred-len", "96", "--patch-len", "24", "--patch-stride", "24", "--d-model", "32", "--d-ff", "64", "--n-heads", "4", "--e-layers", "1", "--train-fraction", ".6", "--batch-size", "4", "--epochs", "3", "--max-windows", "192", "--relation-loss-fraction", ".1", "--calibration-relation-loss-fraction", ".3", "--channel-patch-relation-fraction", source_fraction, "--calibration-channel-patch-relation-fraction", calibration_fraction, "--seed", str(SEED)]
        finetune = [sys.executable, "scripts/finetune_patchtst_acf_head.py", "--dataset", "dataset/traffic/traffic.csv", "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")), "--output", str(prefix.with_name(prefix.name + "_head.pt")), "--seq-len", "96", "--pred-len", "96", "--patch-len", "24", "--patch-stride", "24", "--d-model", "32", "--d-ff", "64", "--n-heads", "4", "--e-layers", "1", "--few-shot-fraction", ".25", "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "4", "--epochs", "5", "--max-train-windows", "64", "--seed", str(SEED)]
        if "--evaluate-test" in pretrain + finetune:
            raise ValueError("PatchTST probe cannot read test")
        if not (prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
            guard(pretrain, prefix, "pretrain")
        if not (prefix.with_name(prefix.name + "_head.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
            guard(finetune, prefix, "finetune")


if __name__ == "__main__":
    main()
