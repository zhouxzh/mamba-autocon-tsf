#!/usr/bin/env python3
"""Run the pinned official PatchTST overlap protocol for Mamba-2 comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/official_patchtst_overlap_pems25_validation_20261003/pems_bay_h192_s5860_official_patchtst_event.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_mamba2_patch24_comparator_20261006"
DEFAULT_SEEDS = (5870,)


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    output = list(command)
    index = output.index(flag)
    output[index + 1] = value
    return output


def _command(seed: int, output: Path) -> list[str]:
    command = json.loads(REFERENCE.read_text(encoding="utf-8"))["command"]
    command[0] = sys.executable
    command = _replace(command, "--seed", str(seed))
    command = _replace(command, "--output", str(output / f"pems_bay_h192_s{seed}_official_patchtst.pt"))
    command = _replace(command, "--patch-len", "24")
    if command[command.index("--patch-stride") + 1] != "8":
        raise ValueError("comparator must retain patch_stride=8")
    if "--evaluate-test" in command:
        raise ValueError("comparator cannot read test rows")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({
        "status": "registered_before_training",
        "validation_only": True,
        "test_metrics_produced": False,
        "target": "PEMS-BAY",
        "horizon": 192,
        "few_shot_fraction": 0.25,
        "seeds": list(seeds),
        "baseline": "official PatchTST at pinned 24/8 overlap-patch protocol",
        "reference_event": str(REFERENCE),
        "comparison": "same target split, labels, 24/8 patch geometry, epochs, and maximum target windows as Mamba-2 probe",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        prefix = args.output / f"pems_bay_h192_s{seed}_official_patchtst"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
            "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
            "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
            "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--",
            *_command(seed, args.output),
        ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
