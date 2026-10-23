#!/usr/bin/env python3
"""Independent three-seed validation replication of channel-patch ACF SSL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006"
SEEDS = {"pems_bay": (6026, 6027, 6028), "metr_la": (6029, 6030, 6031)}
TARGETS = {
    "pems_bay": (ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006", 6001),
    "metr_la": (ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_independent_replication_20261006", 6007),
}
ARMS = ("null", "forecast", "channel_patch")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command)
    out[out.index(flag) + 1] = value
    return out


def commands(target: str, seed: int, arm: str) -> tuple[list[str], list[str], Path]:
    root, reference_seed = TARGETS[target]
    source_arm = "null" if arm == "null" else "active"
    reference = root / f"s{reference_seed}_{source_arm}"
    prefix = OUTPUT / target / f"s{seed}_{arm}"
    pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    ft = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    pre[0] = sys.executable; ft[0] = sys.executable
    if arm == "channel_patch":
        pre += ["--channel-patch-relation-fraction", ".05"]
    pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pre = replace(pre, "--seed", str(seed))
    ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    ft = replace(ft, "--seed", str(seed))
    if "--evaluate-test" in pre + ft:
        raise ValueError("replication cannot read test rows")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "targets": list(targets),
        "seeds": {target: list(SEEDS[target]) for target in targets},
        "question": "Does channelwise full patch-trajectory ACF InfoNCE replicate beyond one-seed screening?",
        "arms": {"null": "next-patch only", "forecast": "forecast-level ACF InfoNCE",
                 "channel_patch": "forecast-level plus full channel-patch ACF InfoNCE"},
        "fixed_contract": {"mixer": "mamba1", "mamba_ts_variant": "v3", "patch_len": 24,
                           "patch_stride": 8, "d_model": 24, "d_ff": 48, "few_shot_fraction": .1,
                           "target_core_frozen": True, "mamba_scan_count": 1,
                           "channel_patch_relation_fraction": .05},
        "no_result_combination": "independent fresh seeds; separate from seed 6025 screen",
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        (OUTPUT / target).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS[target]:
            for arm in ARMS:
                pre, ft, prefix = commands(target, seed, arm)
                pre_event = prefix.with_name(prefix.name + "_pretrain_event.json")
                ft_event = prefix.with_name(prefix.name + "_finetune_event.json")
                checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
                if not (args.resume and pre_event.is_file() and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and complete(pre_event)):
                    guard(pre, prefix, "pretrain")
                if not (args.resume and ft_event.is_file() and checkpoint.is_file() and complete(ft_event)):
                    guard(ft, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
