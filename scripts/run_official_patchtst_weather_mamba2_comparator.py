#!/usr/bin/env python3
"""Run pinned official PatchTST on the Weather Mamba-2 validation protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/official_patchtst_adapter_weather_h192_validation_20261003/weather_h192_s5450_official_patchtst_event.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_weather_mamba2_patch24_comparator_20261006"
DEFAULT_SEEDS = (5900, 5901, 5902)


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _command(seed: int, output: Path) -> list[str]:
    historical = json.loads(REFERENCE.read_text(encoding="utf-8"))["command"]
    def value(flag: str) -> str:
        try:
            return historical[historical.index(flag) + 1]
        except ValueError as exc:
            raise ValueError(f"reference event is missing {flag}") from exc
    # The old Weather event relied on adapter defaults. Make every comparison
    # setting explicit so this new 24/8 geometry is independently reproducible.
    return [
        sys.executable, "scripts/official_patchtst_adapter.py",
        "--official-root", value("--official-root"), "--official-commit", value("--official-commit"),
        "--dataset", "dataset/weather/weather.csv",
        "--output", str(output / f"weather_h192_s{seed}_official_patchtst.pt"),
        "--seq-len", "96", "--pred-len", "192", "--few-shot-fraction", ".1",
        "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "4", "--epochs", "5",
        "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--n-heads", "4",
        "--e-layers", "1", "--patch-len", "24", "--patch-stride", "8", "--dropout", ".1",
        "--fc-dropout", ".05", "--head-dropout", "0", "--revin", "1", "--affine", "0",
        "--subtract-last", "0", "--individual", "0", "--seed", str(seed),
    ]


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
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Weather", "horizon": 192, "few_shot_fraction": 0.1, "seeds": list(seeds),
        "baseline": "official PatchTST", "patch_geometry": {"patch_len": 24, "patch_stride": 8},
        "comparison": "same target chronological split, labels, epochs, and max target windows as Mamba-2 transfer probe",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        prefix = args.output / f"weather_h192_s{seed}_official_patchtst"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
            "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
            "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
            "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *_command(seed, args.output),
        ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
