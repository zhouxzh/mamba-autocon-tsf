#!/usr/bin/env python3
"""Validation-only rank-8 target adapter probe on channel-patch SSL checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_pretrain.pt"
REFERENCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_frozen_core.json"
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_adapter_probe_20261006"


def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out


def main() -> int:
    event = SOURCE.with_name("s6033_channel_patch_finetune_event.json")
    command = list(load(event)["command"]); command[0] = sys.executable
    prefix = OUTPUT / "s6033_channel_patch_adapter"
    command = replace(command, "--source-checkpoint", str(SOURCE))
    command = replace(command, "--output", str(prefix.with_suffix(".pt")))
    command += ["--mamba-ts-channel-adapter", "--mamba-ts-channel-adapter-rank", "8"]
    if "--evaluate-test" in command: raise ValueError("adapter probe cannot read test rows")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "seed": 6033, "source_checkpoint": str(SOURCE),
        "adapter": "rank-8 shared channel adapter after channel-patch ACF representation pretraining",
        "mamba_scan_count": 1, "inference_change": "no additional scan",
    }, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
