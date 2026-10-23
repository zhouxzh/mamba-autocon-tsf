#!/usr/bin/env python3
"""Validation-only METR-LA Mamba-2 probe with few-shot ACF-selected sources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_twohead_probe_20261006"
SELECTION = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_source_selection_20261006/selection.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006"
DEFAULT_SEEDS = (5891,)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _replace_sources(command: list[str], sources: list[str]) -> list[str]:
    result = list(command)
    indices = [index for index, value in enumerate(result) if value == "--source"]
    if len(indices) != len(sources):
        raise ValueError("reference source count differs from selected source count")
    for index, source in zip(indices, sources):
        result[index + 1] = source
    return result


def _commands(seed: int, arm: str, output: Path, sources: list[str]) -> tuple[list[str], list[str]]:
    previous = REFERENCE / f"s5890_{arm}"
    pretrain = _load(previous.with_name(previous.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / f"s{seed}_{arm}"
    pretrain = _replace_sources(pretrain, sources)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _load(previous.with_name(previous.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("probe cannot read test rows")
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    selection = _load(SELECTION)
    if selection.get("main_validation_accessed") or selection.get("test_metrics_produced"):
        raise ValueError("source selector accessed forbidden target rows")
    selected = selection["selected"]
    sources = [f"{item['name']}:{Path(item['path']).relative_to(ROOT)}:{item['format']}" for item in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "target": "METR-LA", "horizon": 192, "few_shot_fraction": .25, "seeds": list(seeds),
        "source_selection_artifact": str(SELECTION), "selected_sources": sources,
        "reference_protocol": str(REFERENCE),
        "change": "replace fixed PEMS07 source with Weather according to target few-shot train-prefix ACF ranking",
    }, indent=2) + "\n", encoding="utf-8")
    for seed in seeds:
        for arm in ("active", "null"):
            prefix = args.output / f"s{seed}_{arm}"
            pretrain, finetune = _commands(seed, arm, args.output, sources)
            _guard(pretrain, prefix, "pretrain")
            _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
