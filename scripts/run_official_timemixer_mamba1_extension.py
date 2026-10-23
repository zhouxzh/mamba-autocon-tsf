#!/usr/bin/env python3
"""Post-release validation extension with hash-locked official TimeMixer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = Path("/tmp/timemixer-adapter-ZctRwq/kwuking-TimeMixer-e246105")
COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
OUTPUT = ROOT / "artifacts/exploratory/official_timemixer_mamba1_postrelease_extension_20261006"
SEEDS = (5930, 5931, 5932)
TARGETS = {
    "pems_bay": ("dataset/PEMS-BAY/PEMS-BAY.csv", "pems_bay_h192", 192, .25),
    "metr_la": ("dataset/METR-LA/METR-LA.csv", "metr_la_h192", 192, .25),
    "weather": ("dataset/weather/weather.csv", "weather_h192", 192, .10),
}


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    seeds = tuple(args.seed or SEEDS)
    if not OFFICIAL_ROOT.is_dir() or not (OFFICIAL_ROOT / "models/TimeMixer.py").is_file():
        raise FileNotFoundError(OFFICIAL_ROOT)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seed values must be unique")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "extension_status": "post-release external TimeMixer baseline; candidate checkpoints frozen",
        "official_root": str(OFFICIAL_ROOT), "official_commit": COMMIT, "targets": list(targets), "seeds": list(seeds),
        "budget": {"seq_len": 96, "epochs": 5, "max_train_windows": 64, "d_model": 24, "d_ff": 48, "e_layers": 1, "batch_size": 2},
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        dataset, stem, pred_len, few = TARGETS[target]
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            prefix = target_output / f"{stem}_s{seed}_official_timemixer"
            _guard([
                sys.executable, str(ROOT / "scripts/official_timemixer_adapter.py"),
                "--official-root", str(OFFICIAL_ROOT), "--official-commit", COMMIT,
                "--dataset", dataset, "--output", str(prefix.with_suffix(".pt")),
                "--seq-len", "96", "--pred-len", str(pred_len), "--few-shot-fraction", str(few),
                "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "2", "--epochs", "5",
                "--max-train-windows", "64", "--d-model", "24", "--d-ff", "48", "--e-layers", "1",
                "--dropout", ".1", "--down-sampling-window", "2", "--down-sampling-layers", "1",
                "--channel-independence", "0", "--seed", str(seed),
            ], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
