#!/usr/bin/env python3
"""Run a validation-only source-compatible AutoCon candidate serially.

Each target retains only its source selected from its own 25%-label training
prefix.  To keep the pretraining optimization budget matched to the former
three-source protocol, the single source contributes 192 windows rather than
64.  Active and exact-null arms share every setting apart from AutoCon's
relation weights.  Test evaluation is intentionally unavailable here.
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
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_similarity_source_candidate_20261003"
MATRICES = {
    "electricity": {"seeds": (5650, 5651, 5652), "source": "Traffic"},
    "traffic": {"seeds": (5660, 5661, 5662), "source": "Electricity"},
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    command = list(command)
    index = command.index(flag)
    command[index + 1] = value
    return command


def _remove_sources(command: list[str]) -> list[str]:
    filtered, index = [], 0
    while index < len(command):
        if command[index] == "--source":
            index += 2
        else:
            filtered.append(command[index])
            index += 1
    return filtered


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    telemetry = prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage))
    event = prefix.with_name(prefix.name + "_{}_event.json".format(stage))
    guarded = [
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(telemetry), "--event-log", str(event), "--", *command,
    ]
    subprocess.run(guarded, cwd=ROOT, check=True)


def _top_source(target: str) -> dict:
    payload = _load(SELECTION_ROOT / "{}.json".format(target))
    top = payload["ranking"][0]
    if top["name"] != MATRICES[target]["source"]:
        raise ValueError("recorded source ranking changed for {}".format(target))
    second = float(payload["ranking"][1]["similarity"])
    first = float(top["similarity"])
    if not first > 3.0 * second:
        raise ValueError("source dominance gate is not met for {}".format(target))
    return top


def _commands(target: str, seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    old = BASELINE_ROOT / "{}_s{}_{}_".format(target, 5550 if target == "electricity" else 5560, arm)
    pretrain = _load(old.with_name(old.name + "pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    top = _top_source(target)
    pretrain = _remove_sources(pretrain)
    pretrain[2:2] = ["--source", "{}:{}:{}".format(top["name"], top["path"], top["format"])]
    prefix = output / "{}_s{}_{}".format(target, seed, arm)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--max-windows", "192")
    pretrain = _replace(pretrain, "--seed", str(seed))
    if "--forecast-contrastive-positive-lags" not in pretrain:
        pretrain.extend(["--forecast-contrastive-positive-lags", "coverage_only"])
    finetune = _load(old.with_name(old.name + "finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_hybrid.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    return pretrain, finetune


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matrix", action="append", choices=sorted(MATRICES), default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected = args.matrix or sorted(MATRICES)
    source_records = {target: _top_source(target) for target in selected}
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "candidate": "source-compatible single-source pretraining",
        "selection_records": source_records, "single_source_windows": 192,
        "historical_multi_source_windows_per_source": 64,
        "budget_note": "192 windows x one selected source equals 64 windows x three historical sources per epoch.",
        "arms": {"active": {"mamba_lambda": 0.1, "relation_loss_fraction": 0.1},
                 "exact_null": {"mamba_lambda": 0.0, "relation_loss_fraction": 0.0}},
        "gate": "all three validation seeds must favor active over exact-null for a target before any test protocol is registered.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.environ.setdefault("AUTOCON_CPU_THREAD_LIMIT", "4")
    for target in selected:
        seeds = MATRICES[target]["seeds"] if args.seed is None else tuple(seed for seed in args.seed if seed in MATRICES[target]["seeds"])
        if not seeds:
            raise ValueError("no requested seed belongs to {}".format(target))
        for seed in seeds:
            for arm in ("active", "null"):
                pretrain, finetune = _commands(target, seed, arm, args.output)
                prefix = args.output / "{}_s{}_{}".format(target, seed, arm)
                if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file()):
                    _guard(pretrain, prefix, "pretrain")
                if not (args.resume and prefix.with_name(prefix.name + "_hybrid.pt").is_file()):
                    _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
