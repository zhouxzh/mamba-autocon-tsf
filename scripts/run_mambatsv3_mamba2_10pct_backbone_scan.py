#!/usr/bin/env python3
"""Validation-only 10%-label Mamba-2 backbone scan on frozen Mamba-1 coordinates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba2_10pct_backbone_scan_20261006"
TARGETS = {
    "pems_bay": {
        "reference": ROOT / "artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006/pems_bay",
        "reference_seed": 5910,
        "dataset": "PEMS-BAY",
        "seeds": (5960, 5961, 5962),
        "sources": None,
        "source_rule": "fixed PEMS07/PEMS08/PEMS04 pool used by the matched Mamba-1 scan",
    },
    "metr_la": {
        "reference": ROOT / "artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006/metr_la",
        "reference_seed": 5910,
        "dataset": "METR-LA",
        "seeds": (5970, 5971, 5972),
        "sources": ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_source_selection_20261006/selection.json",
        "source_rule": "Mamba-1 10%-prefix ACF selector; no validation/test target rows",
    },
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _replace_sources(command: list[str], sources: list[str]) -> list[str]:
    result = list(command)
    positions = [index for index, value in enumerate(result) if value == "--source"]
    if len(positions) != len(sources):
        raise ValueError("reference and selected source counts differ")
    for position, source in zip(positions, sources):
        result[position + 1] = source
    return result


def _selected_sources(selection_path: Path | None) -> list[str] | None:
    if selection_path is None:
        return None
    selection = _load(selection_path)
    if selection.get("main_validation_accessed") or selection.get("test_metrics_produced"):
        raise ValueError("source selector accessed forbidden target rows")
    return [
        f"{item['name']}:{Path(item['path']).relative_to(ROOT)}:{item['format']}"
        for item in selection["selected"]
    ]


def _commands(target: str, seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    spec = TARGETS[target]
    prefix = output / f"s{seed}_{arm}"
    previous = spec["reference"] / f"s{spec['reference_seed']}_{arm}"
    pretrain = _load(previous.with_name(previous.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    sources = _selected_sources(spec["sources"])
    if sources is not None:
        pretrain = _replace_sources(pretrain, sources)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _load(previous.with_name(previous.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    finetune = _replace(finetune, "--few-shot-fraction", ".1")
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("backbone scan cannot read test rows")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def _complete(path: Path) -> bool:
    return path.is_file() and _load(path).get("status") == "completed" and _load(path).get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="override only when exactly one target is selected")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    if args.seed and len(targets) != 1:
        raise ValueError("--seed requires exactly one --target because target coordinates have distinct seed sets")
    override_seeds = tuple(args.seed or ())
    if override_seeds and (len(set(override_seeds)) != len(override_seeds) or any(seed <= 0 for seed in override_seeds)):
        raise ValueError("--seed values must be unique positive integers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / "plan.json"
    previous = _load(plan_path) if plan_path.is_file() else {}
    plan_targets = dict(previous.get("targets", {}))
    plan_seeds = dict(previous.get("seeds_by_target", {}))
    plan_targets.update({name: {"dataset": TARGETS[name]["dataset"], "source_rule": TARGETS[name]["source_rule"]}
                         for name in targets})
    plan_seeds.update({name: list(override_seeds or TARGETS[name]["seeds"]) for name in targets})
    (plan_path).write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "question": "Does Mamba-2 improve the exact frozen 10%-label Mamba-1 ACF-transfer coordinates?",
        "targets": plan_targets, "seeds_by_target": plan_seeds,
        "few_shot_fraction": .10, "horizon": 192,
        "fixed_contract": {"patch_len": 24, "patch_stride": 8, "d_model": 24, "d_ff": 48,
                           "mixer": "mamba2", "mamba_headdim": 12, "target_core_frozen": True,
                           "source_next_patch_and_acf_relation": True},
        "comparison": "same seeds and official PatchTST artifacts as Mamba-1 10% replications; no test reader",
        "excluded_unpaired_exploration": {
            "metr_la": {"seeds": [5960, 5961, 5962],
                        "reason": "not seed-paired with the frozen METR-LA Mamba-1 replication"}
        },
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        seeds = override_seeds or TARGETS[target]["seeds"]
        for seed in seeds:
            for arm in ("active", "null"):
                prefix = target_output / f"s{seed}_{arm}"
                checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
                finetune_event = prefix.with_name(prefix.name + "_finetune_event.json")
                if args.resume and checkpoint.is_file() and _complete(finetune_event):
                    continue
                pretrain, finetune = _commands(target, seed, arm, target_output)
                pretrain_event = prefix.with_name(prefix.name + "_pretrain_event.json")
                pretrain_checkpoint = prefix.with_name(prefix.name + "_pretrain.pt")
                if not (args.resume and pretrain_checkpoint.is_file() and _complete(pretrain_event)):
                    _guard(pretrain, prefix, "pretrain")
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
