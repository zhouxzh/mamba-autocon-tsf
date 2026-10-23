#!/usr/bin/env python3
"""Pinned official iTransformer PEMS03 H192/25% external baseline."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = Path("/tmp/itransformer-adapter-kShbod/iTransformer-c2426e68ca13f74aaec08045c5c724d8ad328124")
COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"
CANDIDATE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
OUTPUT = ROOT / "artifacts/exploratory/official_itransformer_pems03_channel_patch_25pct_20261007"
SEEDS = (6060, 6061, 6062)


def guard(command, prefix):
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)


def complete(path):
    if not path.is_file(): return False
    event = json.loads(path.read_text(encoding="utf-8"))
    return event.get("status") == "completed" and event.get("trigger") is None


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not (OFFICIAL_ROOT / "model/iTransformer.py").is_file(): raise FileNotFoundError(OFFICIAL_ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "horizon": 192, "few_shot_fraction": .25, "seeds": list(SEEDS), "candidate": str(CANDIDATE), "official_root": str(OFFICIAL_ROOT), "official_commit": COMMIT, "adapter_contract": "official model with local chronological adapter", "budget": {"seq_len": 96, "epochs": 5, "max_train_windows": 64, "d_model": 24, "d_ff": 48, "n_heads": 4, "e_layers": 1, "batch_size": 4}, "same_seed_pairing": True}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        candidate = CANDIDATE / f"s{seed}_channel_patch_frozen_core.json"
        if not candidate.is_file() or json.loads(candidate.read_text(encoding="utf-8")).get("test_metrics_produced"):
            raise ValueError(f"candidate checkpoint boundary failure: {seed}")
        prefix = OUTPUT / f"pems03_h192_s{seed}_official_itransformer"
        command = [sys.executable, str(ROOT / "scripts/official_itransformer_adapter.py"), "--official-root", str(OFFICIAL_ROOT), "--official-commit", COMMIT, "--dataset", "dataset/PEMS03/PEMS03.npz", "--output", str(prefix.with_suffix(".pt")), "--seq-len", "96", "--pred-len", "192", "--few-shot-fraction", ".25", "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "4", "--epochs", "5", "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--n-heads", "4", "--e-layers", "1", "--dropout", ".1", "--use-norm", "1", "--seed", str(seed)]
        if "--evaluate-test" in command: raise ValueError("iTransformer baseline cannot read test")
        if not (prefix.with_suffix(".pt").is_file() and complete(prefix.with_name(prefix.name + "_event.json"))): guard(command, prefix)


if __name__ == "__main__": main()
