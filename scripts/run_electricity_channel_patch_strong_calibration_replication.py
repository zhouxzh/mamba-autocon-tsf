#!/usr/bin/env python3
"""Independent Electricity replication for channel-patch strong calibration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006"
SEEDS = (6033, 6034, 6035)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out


def commands(seed: int, arm: str) -> tuple[list[str], list[str], Path]:
    reference = REFERENCE / "s5948_active"
    prefix = OUTPUT / f"s{seed}_{arm}"
    pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    ft = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    pre[0] = sys.executable; ft[0] = sys.executable
    if arm == "channel_patch": pre += ["--channel-patch-relation-fraction", ".05"]
    pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pre = replace(pre, "--seed", str(seed))
    ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    ft = replace(ft, "--seed", str(seed))
    if "--evaluate-test" in pre + ft: raise ValueError("replication cannot read test rows")
    return pre, ft, prefix


def guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def complete(path: Path) -> bool:
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seeds": list(SEEDS),
        "question": "Does the Electricity channel-patch strong-calibration gain replicate on independent seeds?",
        "arms": ["forecast", "channel_patch"],
        "fixed_contract": {"Traffic_source_relation": .1, "Electricity_calibration_relation": .3,
                           "channel_patch_relation": .05, "mixer": "mamba1", "patch": "24/8",
                           "target_core_frozen": True, "mamba_scan_count": 1},
        "no_result_combination": "fresh independent seeds; not pooled with seed 6032 gate",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        for arm in ("forecast", "channel_patch"):
            pre, ft, prefix = commands(seed, arm)
            pre_event = prefix.with_name(prefix.name + "_pretrain_event.json")
            ft_event = prefix.with_name(prefix.name + "_finetune_event.json")
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(pre_event)):
                guard(pre, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and complete(ft_event)):
                guard(ft, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
