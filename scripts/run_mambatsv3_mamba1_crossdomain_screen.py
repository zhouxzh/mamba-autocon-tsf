#!/usr/bin/env python3
"""One-seed Mamba-1 efficiency screen under the frozen cross-domain protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_screen_20261006"
SEED = 5920
TARGETS = {
    "pems_bay": ("PEMS-BAY", "pems_bay"),
    "metr_la": ("METR-LA", "metr_la"),
    "weather": ("Weather", "weather"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _commands(target: str, arm: str) -> tuple[list[str], list[str]]:
    reference = REFERENCE / target / f"s5910_{arm}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = OUTPUT / target / f"s{SEED}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(SEED))
    pretrain = _replace(pretrain, "--mamba-ts-mixer-type", "mamba1")
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(SEED))
    finetune = _replace(finetune, "--mamba-ts-mixer-type", "mamba1")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("Mamba-1 screen cannot read test rows")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for target in targets:
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for arm in ("active", "null"):
            prefix = target_output / f"s{SEED}_{arm}"
            pretrain, finetune = _commands(target, arm)
            _guard(pretrain, prefix, "pretrain")
            _guard(finetune, prefix, "finetune")
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "mixer": "mamba1", "seed": SEED,
        "targets": list(targets), "reference_protocol": str(REFERENCE),
        "change": "only replace Mamba-2 mixer with Mamba-1; retain ACF source pools, 24/8 geometry, and frozen core",
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
