#!/usr/bin/env python3
"""Validation-only staged target adaptation from locked strict source pretraining.

The one-scan MambaTSv3 architecture and source checkpoints are unchanged. The
target head is warmed up while the shared core is frozen for one epoch, then
the same core is unfrozen for the remaining four epochs. This tests whether
the strong PatchTST gap is target-adaptation capacity rather than failed ACF
pretraining. No test reader is invoked.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_staged_adaptation_probe_20261003"
DEFAULT_SEEDS = (5810,)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _command(seed: int, arm: str) -> list[str]:
    event = _load(SOURCE / "s{}_{}_finetune_event.json".format(seed, arm))
    command = list(event["command"])
    command[0] = sys.executable
    command = _replace(command, "--output", str(OUTPUT / "s{}_{}_staged.pt".format(seed, arm)))
    command = _replace(command, "--freeze-mamba-epochs", "1")
    while "--freeze-pretrained-core" in command:
        command.pop(command.index("--freeze-pretrained-core"))
    if "--evaluate-test" in command:
        raise ValueError("probe cannot read test rows")
    return command


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": 0.25, "seeds": list(seeds),
        "source_checkpoints": str(SOURCE),
        "adaptation": "freeze shared core for epoch one target-head warm-up; unfreeze same single-scan core for epochs two through five",
        "comparison": "same source pretraining, target split, seed, epochs, and windows as frozen-core strict matrix",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            prefix = OUTPUT / "s{}_{}_staged".format(seed, arm)
            checkpoint = prefix.with_suffix(".pt")
            if args.resume and checkpoint.is_file():
                continue
            subprocess.run([
                sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
                "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
                "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
                "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *_command(seed, arm),
            ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
