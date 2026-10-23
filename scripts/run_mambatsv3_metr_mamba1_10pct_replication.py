#!/usr/bin/env python3
"""Independent METR-LA H192 10%-label Mamba-1 replication with ACF-selected sources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006"
SELECTION = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_source_selection_20261006/selection.json"
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006"
SEEDS = (5970, 5971, 5972)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    output = list(command)
    index = output.index(flag)
    output[index + 1] = value
    return output


def _replace_sources(command: list[str], sources: list[str]) -> list[str]:
    output = list(command)
    positions = [index for index, value in enumerate(output) if value == "--source"]
    if len(positions) != len(sources):
        raise ValueError("reference and selected source counts differ")
    for position, source in zip(positions, sources):
        output[position + 1] = source
    return output


def _commands(seed: int, arm: str, output: Path, sources: list[str]) -> tuple[list[str], list[str]]:
    previous = REFERENCE / f"s5891_{arm}"
    pretrain = _load(previous.with_name(previous.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / f"s{seed}_{arm}"
    pretrain = _replace_sources(pretrain, sources)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    pretrain = _replace(pretrain, "--mamba-ts-mixer-type", "mamba1")
    finetune = _load(previous.with_name(previous.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    finetune = _replace(finetune, "--few-shot-fraction", ".1")
    finetune = _replace(finetune, "--mamba-ts-mixer-type", "mamba1")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("10% replication cannot read test rows")
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
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    selection = _load(SELECTION)
    if selection.get("main_validation_accessed") or selection.get("test_metrics_produced"):
        raise ValueError("source selector accessed forbidden target rows")
    selected = selection["selected"]
    sources = [f"{item['name']}:{Path(item['path']).relative_to(ROOT)}:{item['format']}" for item in selected]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "METR-LA", "horizon": 192, "few_shot_fraction": .10, "seeds": list(seeds),
        "source_selection_artifact": str(SELECTION), "selected_sources": sources,
        "mamba_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8, "target_core_frozen": True},
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            prefix = OUTPUT / f"s{seed}_{arm}"
            pretrain, finetune = _commands(seed, arm, OUTPUT, sources)
            _guard(pretrain, prefix, "pretrain")
            _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
