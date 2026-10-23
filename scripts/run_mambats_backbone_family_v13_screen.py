#!/usr/bin/env python3
"""Registered v3/v13 frozen-core ACF-transfer backbone-family pilot.

The candidate differs from v3 only by shared depthwise filters over states
from the existing one Mamba scan.  Each target/seed runs a paired v3 active
control and v13 lambda-zero/active arms.  This is validation-only and must
not be used to choose or read any held-out test split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambats_backbone_family_v13_screen_20261006"
TARGETS = {
    "pems_bay": {
        "reference": ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006",
        "reference_seed": 6001,
        "seeds": (6021, 6022),
        "source_rule": "fixed source pool inherited from the independent Mamba-1 replication",
    },
    "metr_la": {
        "reference": ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_independent_replication_20261006",
        "reference_seed": 6007,
        "seeds": (6023, 6024),
        "source_rule": "frozen PEMS-BAY/PEMS07/PEMS08 selector used by the independent Mamba-1 replication",
    },
}
ARMS = {
    "v3_active": ("v3", "active"),
    "v13_null": ("v13", "null"),
    "v13_active": ("v13", "active"),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    updated[updated.index(flag) + 1] = value
    return updated


def _set_variant(command: list[str], variant: str) -> list[str]:
    updated = list(command)
    if "--mamba-ts-variant" in updated:
        return _replace(updated, "--mamba-ts-variant", variant)
    updated.extend(("--mamba-ts-variant", variant))
    return updated


def _commands(target: str, seed: int, name: str, output: Path) -> tuple[list[str], list[str], Path]:
    spec = TARGETS[target]
    variant, source_arm = ARMS[name]
    reference = spec["reference"] / f"s{spec['reference_seed']}_{source_arm}"
    prefix = output / target / f"s{seed}_{name}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    pretrain[0] = sys.executable
    finetune[0] = sys.executable
    pretrain = _set_variant(pretrain, variant)
    finetune = _set_variant(finetune, variant)
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("backbone-family screen cannot read test rows")
    return pretrain, finetune, prefix


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False,
        "question": "Does channel-shared post-scan multi-scale filtering improve frozen-core ACF transfer over v3?",
        "targets": {name: {"seeds": list(TARGETS[name]["seeds"]), "source_rule": TARGETS[name]["source_rule"]} for name in targets},
        "arms": {name: {"mamba_ts_variant": variant, "relation_arm": relation} for name, (variant, relation) in ARMS.items()},
        "fixed_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8, "d_model": 24,
                           "d_ff": 48, "few_shot_fraction": .1, "target_core_frozen": True,
                           "source_next_patch_and_acf_relation": True},
        "continuation_rule": "promote only if v13 active wins v3 active and v13 null in both seed pairs; otherwise stop or report as a negative pilot",
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        for seed in TARGETS[target]["seeds"]:
            for name in ARMS:
                pretrain, finetune, prefix = _commands(target, seed, name, OUTPUT)
                pre_event = prefix.with_name(prefix.name + "_pretrain_event.json")
                checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
                fin_event = prefix.with_name(prefix.name + "_finetune_event.json")
                if not (args.resume and pre_event.is_file() and prefix.with_name(prefix.name + "_pretrain.pt").is_file() and _complete(pre_event)):
                    _guard(pretrain, prefix, "pretrain")
                if not (args.resume and checkpoint.is_file() and _complete(fin_event)):
                    _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
