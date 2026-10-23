#!/usr/bin/env python3
"""Validation-only source-domain optimizer-order probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_source_update_policy_probe_20261006"
SEED = 5983
TARGETS = {
    "pems_bay": (ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006", 5960, "PEMS-BAY"),
    "metr_la": (ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006", 5970, "METR-LA"),
}
ARMS = ("null_sequential", "active_sequential", "null_cyclic", "active_cyclic")


def _load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command); result[result.index(flag) + 1] = value; return result


def _commands(target: str, arm: str) -> tuple[list[str], list[str], Path]:
    root, seed, _ = TARGETS[target]
    active = arm.startswith("active")
    policy = "cyclic" if arm.endswith("cyclic") else "sequential"
    reference = root / f"s{seed}_{'active' if active else 'null'}"
    prefix = OUTPUT / target / f"s{SEED}_{arm}"
    pre = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]; pre[0] = sys.executable
    pre = _replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt"))); pre = _replace(pre, "--seed", str(SEED))
    pre += ["--source-update-policy", policy]
    ft = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]; ft[0] = sys.executable
    ft = _replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt"))); ft = _replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt"))); ft = _replace(ft, "--seed", str(SEED))
    if "--evaluate-test" in pre + ft: raise ValueError("source-order probe cannot read test rows")
    return pre, ft, prefix


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"), "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5", "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")), "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command], cwd=ROOT, check=True)


def _complete(path: Path) -> bool:
    return path.is_file() and _load(path).get("status") == "completed" and _load(path).get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--resume", action="store_true"); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({"status": "registered_before_training", "validation_only": True, "test_metrics_produced": False, "seed": SEED, "targets": {k: v[2] for k, v in TARGETS.items()}, "arms": list(ARMS), "question": "Does cyclic source-domain optimizer ordering improve transfer over fixed sequential ordering?", "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8, "few_shot_fraction": .1, "relation_loss_fraction": .1, "target_core_frozen": True, "inference_change": "none"}}, indent=2) + "\n", encoding="utf-8")
    for target in TARGETS:
        (OUTPUT / target).mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            pre, ft, prefix = _commands(target, arm)
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and _complete(prefix.with_name(prefix.name + "_pretrain_event.json"))): _guard(pre, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_frozen_core.pt").is_file() and _complete(prefix.with_name(prefix.name + "_finetune_event.json"))): _guard(ft, prefix, "finetune")
    return 0


if __name__ == "__main__": raise SystemExit(main())
