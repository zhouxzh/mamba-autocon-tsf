#!/usr/bin/env python3
"""Validation stresses for the frozen independent cross-domain replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATION = ROOT / "artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006"
OUTPUT = REPLICATION / "robustness_validation"
PROTOCOLS = {
    "pems_bay": ("dataset/PEMS-BAY/PEMS-BAY.csv", .25),
    "metr_la": ("dataset/METR-LA/METR-LA.csv", .25),
    "weather": ("dataset/weather/weather.csv", .10),
}
SEEDS = (5910, 5911, 5912)


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def _completed(event: Path) -> bool:
    if not event.is_file():
        return False
    payload = json.loads(event.read_text(encoding="utf-8"))
    return payload.get("status") == "completed" and payload.get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(PROTOCOLS), action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(PROTOCOLS))
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seed values must be unique")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_evaluation", "validation_only": True, "test_metrics_produced": False,
        "replication_root": str(REPLICATION), "targets": list(targets), "seeds": list(seeds),
        "conditions": ["in_distribution", "late_validation_shift", "missing_channels_10",
                       "missing_channels_30", "noise_0p05", "noise_0p10"],
        "mamba_contract": {"mixer": "mamba2", "headdim": 12, "patch_len": 24, "patch_stride": 8},
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        dataset, _ = PROTOCOLS[target]
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for arm in ("active", "null"):
                checkpoint = REPLICATION / target / f"s{seed}_{arm}_frozen_core.pt"
                output = target_output / f"s{seed}_{arm}.json"
                prefix = target_output / f"s{seed}_{arm}"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if args.resume and output.is_file() and _completed(prefix.with_name(prefix.name + "_event.json")):
                    continue
                _guard([
                    sys.executable, str(ROOT / "scripts/evaluate_mambatsv3_lowdata_robustness.py"),
                    "--dataset", dataset, "--checkpoint", str(checkpoint), "--output", str(output),
                    "--batch-size", "4", "--seed", str(seed), "--hybrid",
                ], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
