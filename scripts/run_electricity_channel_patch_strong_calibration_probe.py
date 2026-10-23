#!/usr/bin/env python3
"""Validation-only Electricity gate for channel-patch ACF representation SSL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_probe_20261006"
SEED = 6032


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out


def commands(arm: str) -> tuple[list[str], list[str], Path]:
    reference = REFERENCE / "s5948_active"
    prefix = OUTPUT / f"s{SEED}_{arm}"
    pre = load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    ft = load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    pre[0] = sys.executable; ft[0] = sys.executable
    if arm == "channel_patch":
        pre += ["--channel-patch-relation-fraction", ".05"]
    pre = replace(pre, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pre = replace(pre, "--seed", str(SEED))
    ft = replace(ft, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    ft = replace(ft, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    ft = replace(ft, "--seed", str(SEED))
    if "--evaluate-test" in pre + ft:
        raise ValueError("Electricity channel-patch gate cannot read test rows")
    return pre, ft, prefix


def guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seed": SEED,
        "question": "Does channel-patch ACF InfoNCE improve the positive strong-calibration active protocol?",
        "arms": {"forecast": "Traffic source relation=.1 plus Electricity calibration relation=.3",
                 "channel_patch": "same forecast protocol plus patch-trajectory relation=.05"},
        "fixed_contract": {"mixer": "mamba1", "mamba_ts_variant": "v3", "patch_len": 24,
                           "patch_stride": 8, "d_model": 24, "d_ff": 48, "target_core_frozen": True,
                           "mamba_scan_count": 1, "inference_change": "none"},
        "continuation_rule": "run independent seeds only if channel_patch validation MSE is lower than forecast active and channel-patch anchors are active",
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("forecast", "channel_patch"):
        pre, ft, prefix = commands(arm)
        guard(pre, prefix, "pretrain")
        guard(ft, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
