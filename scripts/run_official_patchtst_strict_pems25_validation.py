#!/usr/bin/env python3
"""Train pinned official PatchTST on the strict PEMS-BAY 25% validation seeds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/official_patchtst_adapter_pems_25_validation_20261003/pems_bay_h192_s5540_official_patchtst_event.json"
OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_strict_pems25_validation_20261003"
SEEDS = (5810, 5811, 5812)


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command); index = command.index(flag); command[index + 1] = value
    return command


def _command(seed: int, output: Path) -> list[str]:
    command = json.loads(REFERENCE.read_text(encoding="utf-8"))["command"]
    command[0] = sys.executable
    command = _replace(command, "--seed", str(seed))
    command = _replace(command, "--output", str(output / "pems_bay_h192_s{}_official_patchtst.pt".format(seed)))
    if "--evaluate-test" in command:
        raise ValueError("validation runner must not read test rows")
    return command


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": 0.25,
        "seeds": list(seeds), "baseline": "official PatchTST", "reference_event": str(REFERENCE),
        "comparison": "same target chronological split, labels, epochs, and max target windows as strict frozen-core PEMS-BAY 25%",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        prefix = args.output / "pems_bay_h192_s{}_official_patchtst".format(seed)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
            "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
            "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
            "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *_command(seed, args.output),
        ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
