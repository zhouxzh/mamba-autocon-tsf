#!/usr/bin/env python3
"""Batch-geometry-matched PEMS03 Channel-Patch finetune versus TimeMixer."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
EXPANSION = ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007"
TIMEMIXER_BASE = ROOT / "artifacts/exploratory/official_timemixer_pems03_channel_patch_25pct_20261007"
TIMEMIXER_EXPANSION = ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_20261007"
SEEDS = (6060, 6061, 6062, 6065, 6066)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def source_paths(seed):
    root = BASE if seed in (6060, 6061, 6062) else EXPANSION
    return root / f"s{seed}_channel_patch_pretrain.pt", root / f"s{seed}_channel_patch_finetune_event.json"


def timemixer_path(seed):
    root = TIMEMIXER_BASE if seed in (6060, 6061, 6062) else TIMEMIXER_EXPANSION
    return root / f"pems03_h192_s{seed}_official_timemixer.json"


def guard(command, prefix):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "horizon": 192, "few_shot_fraction": .25, "seeds": list(SEEDS), "candidate_batch_size": 2, "candidate_epochs": 5, "candidate_max_train_windows": 64, "candidate_updates": 160, "timemixer_batch_size": 2, "timemixer_epochs": 5, "timemixer_max_train_windows": 64, "timemixer_updates": 160, "question": "Does matching TimeMixer's actual minibatch geometry resolve the Channel-Patch comparison?", "candidate_pretraining": "locked completed Channel-Patch pretraining checkpoint for each same seed", "same_seed_pairing": True, "no_test_reader": True, "promotion_rule": "only treat batch geometry as explanatory if five-seed noninferiority passes; otherwise stop budget matching"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        pretrain, event = source_paths(seed); timemixer = timemixer_path(seed)
        if not pretrain.is_file() or not timemixer.is_file(): raise FileNotFoundError("missing locked source artifact")
        if load(timemixer).get("test_metrics_produced"): raise ValueError("locked TimeMixer is test-exposed")
        prefix = OUTPUT / f"s{seed}_channel_patch_batch2"
        command = load(event)["command"]; command[0] = sys.executable
        command = replace(command, "--source-checkpoint", str(pretrain)); command = replace(command, "--output", str(prefix.with_suffix(".pt"))); command = replace(command, "--batch-size", "2")
        if "--evaluate-test" in command: raise ValueError("batch-matched run cannot read test")
        if not (args.resume and prefix.with_suffix(".pt").is_file() and complete(prefix.with_name(prefix.name + "_event.json"))): guard(command, prefix)


if __name__ == "__main__": main()
