#!/usr/bin/env python3
"""Validation-only official PatchTST comparator for Electricity H96 25% labels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = Path("/tmp/patchtst-adapter-oaFMCU/PatchTST-204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised")
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_electricity_strong_calibration_20261006"
SEEDS = (5948, 5949, 5950)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    if not OFFICIAL_ROOT.is_dir() or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("official root missing or invalid seeds")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seeds": list(seeds),
        "official_commit": COMMIT, "patch_geometry": {"patch_len": 24, "patch_stride": 8},
        "comparison": "same chronological split, seed, labels, epoch and target-window budget as strong calibration candidate",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        prefix = OUTPUT / f"electricity_h96_s{seed}_official_patchtst"
        _guard([
            sys.executable, str(ROOT / "scripts/official_patchtst_adapter.py"),
            "--official-root", str(OFFICIAL_ROOT), "--official-commit", COMMIT,
            "--dataset", "dataset/electricity/electricity.csv", "--output", str(prefix.with_suffix(".pt")),
            "--seq-len", "96", "--pred-len", "96", "--few-shot-fraction", ".25",
            "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "4", "--epochs", "5",
            "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--n-heads", "4",
            "--e-layers", "1", "--patch-len", "24", "--patch-stride", "8", "--dropout", ".1",
            "--fc-dropout", ".05", "--head-dropout", "0", "--revin", "1", "--affine", "0",
            "--subtract-last", "0", "--individual", "0", "--seed", str(seed),
        ], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
