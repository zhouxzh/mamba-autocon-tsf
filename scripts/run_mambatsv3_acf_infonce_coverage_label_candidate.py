#!/usr/bin/env python3
"""Run the validation-only strict-positive-label AutoCon candidate serially.

The candidate is replayed from the recorded high-dimensional boundary commands.
It changes only the active arm by restricting InfoNCE channel positives to the
same train-prefix support-qualified lags used for source-window sampling.  The
historical exact-null checkpoints are read-only comparators.  Every child is
run under the GPU thermal/fan guard, and no test command is constructed.
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
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_coverage_label_candidate_20261003"
MATRICES = {
    "electricity": (5550, 5551, 5552),
    "traffic": (5560, 5561, 5562),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_value(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    try:
        index = updated.index(flag)
    except ValueError as exc:
        raise ValueError("recorded command lacks {}".format(flag)) from exc
    if index + 1 >= len(updated):
        raise ValueError("recorded command has no value after {}".format(flag))
    updated[index + 1] = value
    return updated


def _guarded(command: list[str], output: Path, stage: str) -> None:
    telemetry = output.with_name(output.stem + "_{}_telemetry.csv".format(stage))
    event = output.with_name(output.stem + "_{}_event.json".format(stage))
    guarded = [
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(telemetry), "--event-log", str(event), "--",
        *command,
    ]
    subprocess.run(guarded, cwd=ROOT, check=True)


def _record_plan(output: Path, selected: list[str]) -> None:
    plan = {
        "status": "registered_before_training",
        "validation_only": True,
        "test_metrics_produced": False,
        "candidate": "coverage_only InfoNCE channel-positive labels",
        "mechanism": "The positive-label lag bank is restricted to the train-prefix channel-support-qualified lags already used by source sampling.",
        "unchanged": [
            "single-scan MambaTSv3 backbone", "source pools", "chronological splits",
            "initial checkpoints", "seeds", "source/target training budget", "prediction head", "exact-null comparator",
        ],
        "matrices": {name: {"seeds": list(MATRICES[name])} for name in selected},
        "gate": "Each target requires active validation MSE lower than its historical exact-null MSE for all three seeds before any new test protocol can be registered.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def _candidate_commands(target: str, seed: int, output: Path) -> tuple[list[str], list[str]]:
    old_prefix = BASELINE_ROOT / "{}_s{}_active".format(target, seed)
    new_prefix = output / "{}_s{}_coverage_labels".format(target, seed)
    pretrain = _load(old_prefix.with_name(old_prefix.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    pretrain = _replace_value(pretrain, "--output", str(new_prefix.with_name(new_prefix.name + "_pretrain.pt")))
    if "--forecast-contrastive-positive-lags" in pretrain:
        pretrain = _replace_value(pretrain, "--forecast-contrastive-positive-lags", "coverage_only")
    else:
        pretrain.extend(["--forecast-contrastive-positive-lags", "coverage_only"])
    finetune = _load(old_prefix.with_name(old_prefix.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace_value(finetune, "--source-checkpoint", str(new_prefix.with_name(new_prefix.name + "_pretrain.pt")))
    finetune = _replace_value(finetune, "--output", str(new_prefix.with_name(new_prefix.name + "_hybrid.pt")))
    return pretrain, finetune


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matrix", action="append", choices=sorted(MATRICES), default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected = args.matrix or sorted(MATRICES)
    if not BASELINE_ROOT.is_dir():
        raise FileNotFoundError(BASELINE_ROOT)
    _record_plan(args.output, selected)
    for target in selected:
        seeds = MATRICES[target] if args.seed is None else tuple(seed for seed in args.seed if seed in MATRICES[target])
        if not seeds:
            raise ValueError("no requested seed belongs to {}".format(target))
        for seed in seeds:
            pretrain, finetune = _candidate_commands(target, seed, args.output)
            prefix = args.output / "{}_s{}_coverage_labels".format(target, seed)
            pretrain_output = prefix.with_name(prefix.name + "_pretrain.pt")
            hybrid_output = prefix.with_name(prefix.name + "_hybrid.pt")
            env = os.environ.copy()
            env.setdefault("AUTOCON_CPU_THREAD_LIMIT", "4")
            os.environ.update({"AUTOCON_CPU_THREAD_LIMIT": env["AUTOCON_CPU_THREAD_LIMIT"]})
            if not (args.resume and pretrain_output.is_file()):
                _guarded(pretrain, prefix, "pretrain")
            if not (args.resume and hybrid_output.is_file()):
                _guarded(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
