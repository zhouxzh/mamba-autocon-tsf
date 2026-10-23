#!/usr/bin/env python3
"""Same-seed official PatchTST Electricity comparator for channel-patch SSL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = Path("/tmp/patchtst-adapter-oaFMCU/PatchTST-204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised")
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_electricity_channel_patch_replication_20261006"
SEEDS = (6033, 6034, 6035)


def guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    if not OFFICIAL_ROOT.is_dir(): raise FileNotFoundError(OFFICIAL_ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "Electricity", "horizon": 96, "few_shot_fraction": .25, "seeds": list(SEEDS),
        "official_commit": COMMIT, "patch_geometry": {"patch_len": 24, "patch_stride": 8},
        "comparison": "same seed, chronological split, labels, epochs, widths, and target-window budget as channel-patch replication",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        prefix = OUTPUT / f"electricity_h96_s{seed}_official_patchtst"
        guard([sys.executable, str(ROOT / "scripts/official_patchtst_adapter.py"),
               "--official-root", str(OFFICIAL_ROOT), "--official-commit", COMMIT,
               "--dataset", "dataset/electricity/electricity.csv", "--output", str(prefix.with_suffix(".pt")),
               "--seq-len", "96", "--pred-len", "96", "--few-shot-fraction", ".25",
               "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "4", "--epochs", "5",
               "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--n-heads", "4",
               "--e-layers", "1", "--patch-len", "24", "--patch-stride", "8", "--dropout", ".1",
               "--fc-dropout", ".05", "--head-dropout", "0", "--revin", "1", "--affine", "0",
               "--subtract-last", "0", "--individual", "0", "--seed", str(seed)], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
