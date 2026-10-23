#!/usr/bin/env python3
"""Run a fresh-seed, validation-only pooled-patch representation probe.

This is deliberately a two-target, one-seed mechanism probe after the
channelwise forecast-InfoNCE high-dimensional failures.  It preserves the
source-compatible single-source selection and matched 192-window budget.  The
only active/null difference is the AutoCon weight; both arms construct the
same pretraining-only pooled-patch head.  No test command is available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_electricity_traffic_25_boundary_20261003"
SELECTION_ROOT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_source_selection_25_20261003"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pooled_patch_probe_20261003"
MATRICES = {"electricity": {"seed": 5750, "source": "Traffic"}, "traffic": {"seed": 5760, "source": "Electricity"}}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command)
    index = command.index(flag)
    command[index + 1] = value
    return command


def _single_source(command: list[str], spec: dict) -> list[str]:
    output, index = [], 0
    while index < len(command):
        if command[index] == "--source":
            index += 2
        else:
            output.append(command[index])
            index += 1
    output[2:2] = ["--source", "{}:{}:{}".format(spec["name"], spec["path"], spec["format"])]
    return output


def _top_source(target: str) -> dict:
    ranking = _load(SELECTION_ROOT / "{}.json".format(target))["ranking"]
    if ranking[0]["name"] != MATRICES[target]["source"]:
        raise ValueError("the recorded dominant source changed for {}".format(target))
    if not float(ranking[0]["similarity"]) > 3.0 * float(ranking[1]["similarity"]):
        raise ValueError("source-dominance gate is not met for {}".format(target))
    return ranking[0]


def _commands(target: str, arm: str, output: Path, seed: int,
              variance_weight: float) -> tuple[list[str], list[str]]:
    old_seed = 5550 if target == "electricity" else 5560
    old = BASELINE_ROOT / "{}_s{}_{}_".format(target, old_seed, arm)
    pretrain = _load(old.with_name(old.name + "pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    pretrain = _single_source(pretrain, _top_source(target))
    prefix = output / "{}_s{}_pooled_patch_{}".format(target, seed, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--max-windows", "192")
    pretrain = _replace(pretrain, "--seed", str(seed))
    pretrain = _replace(pretrain, "--relation-view", "pooled_patch_contrastive")
    if "--forecast-contrastive-positive-lags" not in pretrain:
        pretrain.extend(["--forecast-contrastive-positive-lags", "coverage_only"])
    if arm == "active" and variance_weight:
        pretrain.extend(["--pooled-patch-variance-weight", str(variance_weight)])
    finetune = _load(old.with_name(old.name + "finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_hybrid.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage))),
        "--event-log", str(prefix.with_name(prefix.name + "_{}_event.json".format(stage))),
        "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matrix", action="append", choices=sorted(MATRICES), default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="optional fresh seed; applies to the selected one-target probe")
    parser.add_argument("--pooled-patch-variance-weight", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected = args.matrix or sorted(MATRICES)
    if args.seed is not None and len(selected) != 1:
        raise ValueError("--seed requires exactly one --matrix target")
    if args.pooled_patch_variance_weight < 0:
        raise ValueError("pooled-patch-variance-weight must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "candidate": "pooled patch representation ACF InfoNCE",
        "relation_view": "pooled_patch_contrastive", "single_source_windows": 192,
        "pooled_patch_variance_weight": args.pooled_patch_variance_weight,
        "source_selection": {target: _top_source(target) for target in selected},
        "seeds": {target: MATRICES[target]["seed"] for target in selected},
        "gate": "This fresh-seed probe cannot open tests. It only decides whether a separate three-seed validation matrix is justified.",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.environ.setdefault("AUTOCON_CPU_THREAD_LIMIT", "4")
    for target in selected:
        seed = MATRICES[target]["seed"] if args.seed is None else args.seed
        for arm in ("active", "null"):
            pretrain, finetune = _commands(
                target, arm, args.output, seed, args.pooled_patch_variance_weight,
            )
            prefix = args.output / "{}_s{}_pooled_patch_{}".format(target, seed, arm)
            if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file()):
                _guard(pretrain, prefix, "pretrain")
            if not (args.resume and prefix.with_name(prefix.name + "_hybrid.pt").is_file()):
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
