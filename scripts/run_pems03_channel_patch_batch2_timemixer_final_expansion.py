#!/usr/bin/env python3
"""Final capped five-seed precision expansion for batch-matched PEMS03 TimeMixer."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
PRIOR = ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_expansion_20261007"
TIMEMIXER_ROOT = Path("/tmp/timemixer-adapter-ZctRwq/kwuking-TimeMixer-e246105")
TIMEMIXER_COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_final_expansion_20261007"
SEEDS = (6089, 6090, 6091, 6092, 6093)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None
def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    if not (TIMEMIXER_ROOT / "models/TimeMixer.py").is_file(): raise FileNotFoundError(TIMEMIXER_ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "horizon": 192, "few_shot_fraction": .25, "fresh_seeds": list(SEEDS), "prior_batch_matched_seeds": [6060, 6061, 6062, 6065, 6066, 6084, 6085, 6086, 6087, 6088], "question": "Final capped precision expansion after ten batch/update-matched pairs yielded ratio upper 1.00399.", "candidate_batch_size": 2, "candidate_epochs": 5, "candidate_max_train_windows": 64, "candidate_updates": 160, "timemixer_batch_size": 2, "timemixer_epochs": 5, "timemixer_max_train_windows": 64, "timemixer_updates": 160, "same_seed_pairing": True, "aggregation_rule": "combine exactly five fresh seeds with the fixed ten-seed batch-matched cohort; this is the final TimeMixer precision expansion regardless of outcome", "no_test_reader": True, "stop_rule": "no additional TimeMixer seed expansion after this artifact"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = BASE / "s6060_channel_patch"
    for seed in SEEDS:
        prefix = OUTPUT / f"s{seed}_channel_patch_batch2"; pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable; finetune[0] = sys.executable
        pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
        finetune = replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(prefix.with_name(prefix.name + "_batch2.pt"))); finetune = replace(finetune, "--batch-size", "2"); finetune = replace(finetune, "--seed", str(seed))
        if "--evaluate-test" in pre + finetune: raise ValueError("final expansion cannot read test")
        if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): guard(pre, prefix, "pretrain")
        if not (args.resume and prefix.with_name(prefix.name + "_batch2.pt").is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))): guard(finetune, prefix, "finetune")
        tm_prefix = OUTPUT / f"pems03_h192_s{seed}_official_timemixer"
        tm = [sys.executable, str(ROOT / "scripts/official_timemixer_adapter.py"), "--official-root", str(TIMEMIXER_ROOT), "--official-commit", TIMEMIXER_COMMIT, "--dataset", "dataset/PEMS03/PEMS03.npz", "--output", str(tm_prefix.with_suffix(".pt")), "--seq-len", "96", "--pred-len", "192", "--few-shot-fraction", ".25", "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "2", "--epochs", "5", "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--e-layers", "1", "--dropout", ".1", "--down-sampling-window", "2", "--down-sampling-layers", "1", "--channel-independence", "0", "--seed", str(seed)]
        if "--evaluate-test" in tm: raise ValueError("TimeMixer cannot read test")
        if not (args.resume and tm_prefix.with_suffix(".pt").is_file() and complete(tm_prefix.with_name(tm_prefix.name + "_timemixer_event.json"))): guard(tm, tm_prefix, "timemixer")


if __name__ == "__main__": main()
