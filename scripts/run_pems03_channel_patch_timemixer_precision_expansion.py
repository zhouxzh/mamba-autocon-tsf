#!/usr/bin/env python3
"""Pre-registered fresh-seed precision expansion for PEMS03 vs TimeMixer."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_REFERENCE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
TIMEMIXER_ROOT = Path("/tmp/timemixer-adapter-ZctRwq/kwuking-TimeMixer-e246105")
TIMEMIXER_COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007"
SEEDS = (6065, 6066)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command, flag, value):
    updated = list(command); updated[updated.index(flag) + 1] = value; return updated
def complete(path): return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(command, prefix, stage):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def assert_target_excluded(command):
    inputs = [command[index + 1] for index, item in enumerate(command[:-1]) if item in {"--source", "--initial-checkpoint"}]
    if not inputs or any("pems03" in item.lower() for item in inputs): raise ValueError("PEMS03 source/initialization leakage")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not (TIMEMIXER_ROOT / "models/TimeMixer.py").is_file(): raise FileNotFoundError(TIMEMIXER_ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "horizon": 192, "few_shot_fraction": .25, "seeds": list(SEEDS), "existing_seed_set": [6060, 6061, 6062], "question": "Does two fresh paired seeds resolve the TimeMixer noninferiority uncertainty after candidate mean advantage but upper ratio 1.02374?", "arms": ["channel_patch", "official_timemixer"], "candidate_reference": str(CANDIDATE_REFERENCE), "official_root": str(TIMEMIXER_ROOT), "official_commit": TIMEMIXER_COMMIT, "fixed_contract": {"candidate": "Mamba1 24/8 full channel-patch trajectory SSL; source/initialization target excluded; frozen target core", "timemixer": "official code, d_model 24, d_ff 48, one layer, batch 2", "epochs": 5, "max_train_windows": 64}, "aggregation_rule": "combine only these two fresh pairs with the completed 6060--6062 same-coordinate pairs; pass if five-seed one-SE ratio upper is at most one", "no_result_combination": "no comparison to other coordinates or baselines"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    reference = CANDIDATE_REFERENCE / "s6060_channel_patch"
    for seed in SEEDS:
        candidate_prefix = OUTPUT / f"s{seed}_channel_patch"
        pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; finetune = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
        pre[0] = sys.executable; finetune[0] = sys.executable
        pre = replace(pre, "--output", str(candidate_prefix.with_name(candidate_prefix.name + "_pretrain.pt"))); pre = replace(pre, "--seed", str(seed))
        finetune = replace(finetune, "--source-checkpoint", str(candidate_prefix.with_name(candidate_prefix.name + "_pretrain.pt"))); finetune = replace(finetune, "--output", str(candidate_prefix.with_name(candidate_prefix.name + "_frozen_core.pt"))); finetune = replace(finetune, "--seed", str(seed))
        assert_target_excluded(pre)
        if "--evaluate-test" in pre + finetune: raise ValueError("precision expansion cannot read test")
        if not (candidate_prefix.with_name(candidate_prefix.name + "_pretrain.pt").is_file() and complete(candidate_prefix.with_name(candidate_prefix.name + "_pretrain_event.json"))): guard(pre, candidate_prefix, "pretrain")
        if not (candidate_prefix.with_name(candidate_prefix.name + "_frozen_core.pt").is_file() and complete(candidate_prefix.with_name(candidate_prefix.name + "_finetune_event.json"))): guard(finetune, candidate_prefix, "finetune")
        baseline_prefix = OUTPUT / f"pems03_h192_s{seed}_official_timemixer"
        baseline = [sys.executable, str(ROOT / "scripts/official_timemixer_adapter.py"), "--official-root", str(TIMEMIXER_ROOT), "--official-commit", TIMEMIXER_COMMIT, "--dataset", "dataset/PEMS03/PEMS03.npz", "--output", str(baseline_prefix.with_suffix(".pt")), "--seq-len", "96", "--pred-len", "192", "--few-shot-fraction", ".25", "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "2", "--epochs", "5", "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--e-layers", "1", "--dropout", ".1", "--down-sampling-window", "2", "--down-sampling-layers", "1", "--channel-independence", "0", "--seed", str(seed)]
        if "--evaluate-test" in baseline: raise ValueError("TimeMixer baseline cannot read test")
        if not (baseline_prefix.with_suffix(".pt").is_file() and complete(baseline_prefix.with_name(baseline_prefix.name + "_event.json"))): guard(baseline, baseline_prefix, "timemixer")


if __name__ == "__main__": main()
