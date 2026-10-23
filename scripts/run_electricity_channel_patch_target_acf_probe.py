#!/usr/bin/env python3
"""Validation-only target-prefix ACF adaptation probe for channel-patch SSL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_pretrain.pt"
REFERENCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_frozen_core.json"
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_target_acf_probe_20261006"
SEED = 6033


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out


def guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-affine-fraction", type=float, default=.1)
    args = parser.parse_args()
    if not 0.0 < args.target_affine_fraction <= 1.0: raise ValueError("target affine fraction must be in (0,1]")
    reference = load(REFERENCE)
    # The frozen replication manifest stores the exact source command only in
    # its event sidecar; derive it from the paired replication directory.
    event = SOURCE.with_name("s6033_channel_patch_finetune_event.json")
    event_payload = load(event)
    command = list(event_payload["command"])
    command[0] = sys.executable
    prefix = OUTPUT / f"s{SEED}_target_acf"
    command = replace(command, "--source-checkpoint", str(SOURCE))
    command = replace(command, "--output", str(prefix.with_suffix(".pt")))
    command = replace(command, "--seed", str(SEED))
    command += ["--target-affine-fraction", str(args.target_affine_fraction), "--target-affine-pairing"]
    if "--evaluate-test" in command: raise ValueError("target ACF probe cannot read test rows")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "seed": SEED,
        "source_checkpoint": str(SOURCE), "channel_patch_pretraining": True,
        "target_affine_fraction": args.target_affine_fraction,
        "question": "Does train-prefix target ACF adaptation improve the frozen channel-patch representation transfer?",
        "inference_change": "none", "mamba_scan_count": 1,
    }, indent=2) + "\n", encoding="utf-8")
    guard(command, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
