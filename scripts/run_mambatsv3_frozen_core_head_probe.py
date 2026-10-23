#!/usr/bin/env python3
"""Validate frozen transferred core with a target-trainable forecast head.

The active/null source checkpoints are the completed Traffic pooled-patch
anti-collapse probe. This runner only changes target adaptation: the
transferred shared core stays frozen and the target forecast head, NLinear
head, and residual scale remain trainable. It is validation-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "artifacts/exploratory/mambatsv3_pooled_patch_variance_probe_20261003"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_frozen_core_head_probe_20261003"
SEED = 5770


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command)
    index = command.index(flag)
    command[index + 1] = value
    return command


def _command(arm: str, output: Path) -> list[str]:
    old = SOURCE_ROOT / "traffic_s{}_pooled_patch_{}".format(SEED, arm)
    event = _load(old.with_name(old.name + "_finetune_event.json"))
    command = list(event["command"])
    command[0] = sys.executable
    command = _replace(command, "--output", str(output / "traffic_s{}_{}_frozen_core.pt".format(SEED, arm)))
    command = _replace(command, "--freeze-mamba-epochs", "0")
    command.append("--freeze-pretrained-core")
    if "--evaluate-test" in command:
        raise ValueError("source event unexpectedly contains test evaluation")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Traffic", "seed": SEED,
        "source_root": str(SOURCE_ROOT),
        "adaptation": "freeze transferred shared core; train target forecast head, NLinear head, and residual scale",
        "arms": ["active", "null"],
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for arm in ("active", "null"):
        checkpoint = args.output / "traffic_s{}_{}_frozen_core.pt".format(SEED, arm)
        if args.resume and checkpoint.is_file():
            continue
        prefix = checkpoint.with_suffix("")
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
            "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
            "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
            "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *_command(arm, args.output),
        ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
