#!/usr/bin/env python3
"""Validation-only target temporal-MLP adapter probe from locked source checkpoints."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/exploratory/mambatsv3_pems_channel_adapter_probe_20261003"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_target_mlp_adapter_probe_20261003"
SEED = 5840


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _command(arm: str) -> list[str]:
    event = _load(SOURCE / "s{}_{}_finetune_event.json".format(SEED, arm))
    command = list(event["command"])
    command[0] = sys.executable
    command = _replace(command, "--output", str(OUTPUT / "s{}_{}_target_mlp.pt".format(SEED, arm)))
    while command.count("--freeze-pretrained-core") > 1:
        command.pop(command.index("--freeze-pretrained-core"))
    # The source checkpoints are reused; channel adapter is omitted so this
    # isolates the temporal target decoder on the same frozen source core.
    while "--mamba-ts-channel-adapter" in command:
        command.pop(command.index("--mamba-ts-channel-adapter"))
    while "--mamba-ts-channel-adapter-rank" in command:
        index = command.index("--mamba-ts-channel-adapter-rank")
        command.pop(index)
        command.pop(index)
    if "--mamba-ts-target-mlp-adapter" not in command:
        command.extend(["--mamba-ts-target-mlp-adapter", "--mamba-ts-target-mlp-adapter-rank", "32"])
    if "--evaluate-test" in command:
        raise ValueError("probe cannot read test rows")
    return command


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": 0.25, "seed": SEED,
        "source_checkpoints": str(SOURCE), "target_adapter": "rank-32 zero-gated shared nonlinear patch-state decoder",
        "adaptation": "freeze transferred shared core; train forecast head, temporal MLP adapter, NLinear head, and residual scale",
    }, indent=2) + "\n", encoding="utf-8")
    for arm in ("active", "null"):
        prefix = OUTPUT / "s{}_{}_target_mlp".format(SEED, arm)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
            "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
            "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
            "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *_command(arm),
        ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
