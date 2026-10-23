#!/usr/bin/env python3
"""Validation-only target RevIN-affine gate on channel-patch SSL transfer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_pretrain.pt"
REFERENCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_frozen_core.json"
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_revin_probe_20261006"


def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out


def main() -> int:
    command = list(load(SOURCE.with_name("s6033_channel_patch_finetune_event.json"))["command"])
    command[0] = sys.executable
    prefix = OUTPUT / "s6033_channel_patch_revin"
    command = replace(command, "--source-checkpoint", str(SOURCE))
    command = replace(command, "--output", str(prefix.with_suffix(".pt")))
    command += ["--mamba-ts-revin-affine"]
    if "--evaluate-test" in command: raise ValueError("RevIN probe cannot read test rows")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "seed": 6033, "source_checkpoint": str(SOURCE),
        "mechanism": "identity-initialized target-only reversible affine calibration after MeanStd normalization",
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
