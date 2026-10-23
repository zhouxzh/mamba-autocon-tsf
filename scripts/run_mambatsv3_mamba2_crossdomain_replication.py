#!/usr/bin/env python3
"""Frozen three-domain validation replication for the selected Mamba-2 protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_mamba2_crossdomain_replication_20261006"
SEEDS = (5910, 5911, 5912)
PROTOCOLS = {
    "pems_bay": (ROOT / "artifacts/exploratory/mambatsv3_pems_mamba2_twohead_probe_20261006", "PEMS-BAY", .25),
    "metr_la": (ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006", "METR-LA", .25),
    "weather": (ROOT / "artifacts/exploratory/mambatsv3_weather_mamba2_twohead_probe_20261006", "Weather", .10),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _commands(root: Path, seed: int, arm: str, output: Path) -> tuple[list[str], list[str]]:
    reference = root / f"s{(5880 if root.name.startswith('mambatsv3_pems') else 5891 if 'metr' in root.name else 5900)}_{arm}"
    pretrain = _load(reference.with_name(reference.name + "_pretrain_event.json"))["command"]
    pretrain[0] = sys.executable
    prefix = output / f"s{seed}_{arm}"
    pretrain = _replace(pretrain, "--output", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    pretrain = _replace(pretrain, "--seed", str(seed))
    finetune = _load(reference.with_name(reference.name + "_finetune_event.json"))["command"]
    finetune[0] = sys.executable
    finetune = _replace(finetune, "--source-checkpoint", str(prefix.with_name(prefix.name + "_pretrain.pt")))
    finetune = _replace(finetune, "--output", str(prefix.with_name(prefix.name + "_frozen_core.pt")))
    finetune = _replace(finetune, "--seed", str(seed))
    if "--evaluate-test" in pretrain + finetune:
        raise ValueError("replication cannot read test rows")
    return pretrain, finetune


def _guard(command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def _completed(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = _load(path)
    return payload.get("status") == "completed" and payload.get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(PROTOCOLS), action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(PROTOCOLS))
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("--seed values must be unique positive integers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "replication_status": "independent seeds selected after freezing architecture and source-selection rules",
        "seeds": list(seeds), "targets": list(targets),
        "protocols": {key: {"reference": str(value[0]), "target": value[1], "few_shot_fraction": value[2]}
                      for key, value in PROTOCOLS.items()},
        "mamba_contract": {"mixer": "mamba2", "headdim": 12, "patch_len": 24,
                           "patch_stride": 8, "target_core_frozen": True},
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        reference, _, _ = PROTOCOLS[target]
        target_output = OUTPUT / target
        target_output.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for arm in ("active", "null"):
                prefix = target_output / f"s{seed}_{arm}"
                event = prefix.with_name(prefix.name + "_finetune_event.json")
                checkpoint = prefix.with_name(prefix.name + "_frozen_core.pt")
                if args.resume and checkpoint.is_file() and _completed(event):
                    continue
                pretrain, finetune = _commands(reference, seed, arm, target_output)
                if not (args.resume and prefix.with_name(prefix.name + "_pretrain.pt").is_file()
                        and _completed(prefix.with_name(prefix.name + "_pretrain_event.json"))):
                    _guard(pretrain, prefix, "pretrain")
                _guard(finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
