#!/usr/bin/env python3
"""Read-only six-condition validation stresses for channel-patch replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_channel_patch_replication_20261006"
SEEDS = {"pems_bay": (6026, 6027, 6028), "metr_la": (6029, 6030, 6031)}
DATASETS = {"pems_bay": "dataset/PEMS-BAY/PEMS-BAY.csv", "metr_la": "dataset/METR-LA/METR-LA.csv"}
CONDITIONS = ["in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10"]


def guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "completed" and payload.get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(SEEDS), action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); targets = tuple(args.target or sorted(SEEDS))
    output = CHECKPOINTS / "robustness_validation"; output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False,
        "targets": list(targets), "conditions": CONDITIONS,
        "candidate_arms": ["null", "forecast", "channel_patch"], "official_baseline": "PatchTST",
        "same_seed_pairing": True,
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        for seed in SEEDS[target]:
            for arm in ("null", "forecast", "channel_patch"):
                checkpoint = CHECKPOINTS / target / f"s{seed}_{arm}_frozen_core.pt"
                if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
                prefix = output / f"{target}_s{seed}_{arm}"
                if not (args.resume and prefix.with_suffix(".json").is_file() and complete(prefix.with_name(prefix.name + "_event.json"))):
                    guard([sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                           "--dataset", DATASETS[target], "--checkpoint", str(checkpoint),
                           "--output", str(prefix.with_suffix(".json")),
                           "--batch-size", "4", "--seed", str(seed), "--hybrid"], prefix)
            checkpoint = PATCH / target / f"{target}_h192_s{seed}_official_patchtst.pt"
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            prefix = output / f"{target}_s{seed}_official_patchtst"
            if not (args.resume and prefix.with_suffix(".json").is_file() and complete(prefix.with_name(prefix.name + "_event.json"))):
                guard([sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_robustness.py"),
                       "--dataset", DATASETS[target], "--checkpoint", str(checkpoint),
                       "--output", str(prefix.with_suffix(".json")),
                       "--batch-size", "4", "--seed", str(seed)], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
