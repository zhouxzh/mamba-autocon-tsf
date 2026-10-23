#!/usr/bin/env python3
"""Portable strict PEMS03 Channel-Patch replication for an independent host.

This runner deliberately builds commands from the frozen protocol rather than
reusing local event files.  It opens only chronological validation data.  A
human must verify that ``--machine-id`` identifies a physically independent
machine before its output can be used as independent-machine evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/reproducibility/pems03_channel_patch_independent_machine_20261007"
SEEDS = (6080, 6081, 6082)
SOURCES = (
    "PEMS04:dataset/PEMS04/PEMS04.npz:npz",
    "PEMS08:dataset/PEMS08/PEMS08.npz:npz",
    "PEMS07:dataset/PEMS07/PEMS07.npz:npz",
)
INITIAL = (
    "artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems04_MambaTSv3AutoCon_h12_s2131_l0p001_c22b7150b652d_MambaTSv3AutoCon_pems04_ftM_sl96_ll48_pl12_dm2_h179e4641a67e8f06/checkpoint.pth",
    "artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems08_MambaTSv3AutoCon_h12_s2131_l0p001_c1c86338c16d3_MambaTSv3AutoCon_pems08_ftM_sl96_ll48_pl12_dm2_h49466f04bd7735b8/checkpoint.pth",
)
TRACKED = (
    "models/MambaTS.py", "models/MambaDLinearHybrid.py", "layers/losses.py",
    "scripts/pretrain_mambatsv3_core_autocon.py", "scripts/finetune_mambatsv3_dlinear_hybrid.py",
    "scripts/run_gpu_guarded.py", "dataset/PEMS03/PEMS03.npz", "dataset/PEMS04/PEMS04.npz",
    "dataset/PEMS07/PEMS07.npz", "dataset/PEMS08/PEMS08.npz", *INITIAL,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def guard(python: Path, command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        str(python), str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")),
        "--", *command,
    ], cwd=ROOT, check=True)


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    event = json.loads(path.read_text(encoding="utf-8"))
    return event.get("status") == "completed" and event.get("trigger") is None


def pretrain_command(python: Path, output: Path, seed: int, arm: str) -> list[str]:
    command = [
        str(python), "scripts/pretrain_mambatsv3_core_autocon.py",
        "--source", SOURCES[0], "--source", SOURCES[1], "--source", SOURCES[2],
        "--initial-checkpoint", INITIAL[0], "--initial-checkpoint", INITIAL[1],
        "--output", str(output), "--seq-len", "96", "--patch-len", "24", "--patch-stride", "8",
        "--source-pred-len", "192", "--d-model", "24", "--d-ff", "48", "--mamba-d-state", "8",
        "--mamba-d-conv", "2", "--mamba-expand", "1", "--context-rank", "8", "--train-fraction", ".6",
        "--batch-size", "4", "--epochs", "3", "--max-windows", "64", "--relation-pair-policy", "contrastive_cluster",
        "--forecast-contrastive-pairing", "channel_coverage", "--min-channel-positive-support", ".5",
        "--reconstruction-weight", "0", "--predictive-weight", "1", "--prediction-target", "last_value_residual",
        "--pretrain-task", "next_patch", "--relation-view", "forecast_contrastive", "--mamba-lambda", ".1",
        "--relation-loss-fraction", ".1", "--learning-rate", ".0003", "--seed", str(seed),
        "--mamba-ts-mixer-type", "mamba1", "--mamba-headdim", "12",
    ]
    if arm == "channel_patch":
        command.extend(("--channel-patch-relation-fraction", ".05"))
    if arm not in {"forecast", "channel_patch"}:
        raise ValueError("arm must be forecast or channel_patch")
    return command


def finetune_command(python: Path, source: Path, output: Path, seed: int, few: float) -> list[str]:
    return [
        str(python), "scripts/finetune_mambatsv3_dlinear_hybrid.py", "--dataset", "dataset/PEMS03/PEMS03.npz",
        "--source-checkpoint", str(source), "--output", str(output), "--seq-len", "96", "--pred-len", "192",
        "--patch-len", "24", "--patch-stride", "8", "--d-model", "24", "--d-ff", "48", "--mamba-d-state", "8",
        "--mamba-d-conv", "2", "--mamba-expand", "1", "--context-rank", "16", "--channel-agnostic-scan",
        "--transfer-prediction-head", "--prediction-head-transfer-mode", "exact", "--linear-model", "NLinear",
        "--mamba-correction-output", "--few-shot-fraction", str(few), "--train-fraction", ".6", "--val-fraction", ".2",
        "--batch-size", "4", "--epochs", "5", "--max-train-windows", "64", "--freeze-mamba-epochs", "0",
        "--seed", str(seed), "--freeze-pretrained-core", "--mamba-ts-mixer-type", "mamba1", "--mamba-headdim", "12",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--machine-id", required=True,
                        help="operator-provided independent host identifier; recorded but not trusted automatically")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--few-shot-fraction", choices=[0.1, 0.25], type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true",
                        help="write the protocol/hash packet without starting any training")
    args = parser.parse_args(); python = args.python.resolve(); output = args.output.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    missing = [item for item in TRACKED if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError("missing protocol inputs: {}".format(", ".join(missing)))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "packet_only" if args.manifest_only else "registered_before_training",
        "independent_machine": "unverified_operator_claim", "machine_id": args.machine_id,
        "hostname": platform.node(), "python": str(python), "target": "PEMS03", "horizon": 192,
        "few_shot_fraction": args.few_shot_fraction, "seeds": list(SEEDS), "arms": ["forecast", "channel_patch"],
        "source_pool": ["PEMS04", "PEMS08", "PEMS07"], "initialization_sources": ["PEMS04", "PEMS08"],
        "target_excluded_from_source_and_initialization": True,
        "fixed_contract": {"mixer": "mamba1", "patch": "24/8", "d_model": 24, "d_ff": 48, "mamba_scan_count": 1, "target_core_frozen": True, "channel_patch_relation_fraction": .05},
        "thermal_guard": {"temperature_limit_c": 78, "fan_limit_percent": 85, "interval_sec": 5},
        "source_hashes": {item: digest(ROOT / item) for item in TRACKED},
        "independent_machine_acceptance": "requires an authorized reviewer to verify a distinct physical host and environment; this runner does not self-certify independence",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.manifest_only:
        return 0
    for seed in SEEDS:
        for arm in ("forecast", "channel_patch"):
            prefix = output / f"s{seed}_{arm}"; pre_output = prefix.with_name(prefix.name + "_pretrain.pt"); final_output = prefix.with_name(prefix.name + "_frozen_core.pt")
            pre_event = prefix.with_name(prefix.name + "_pretrain_event.json"); final_event = prefix.with_name(prefix.name + "_finetune_event.json")
            pretrain = pretrain_command(python, pre_output, seed, arm)
            finetune = finetune_command(python, pre_output, final_output, seed, args.few_shot_fraction)
            if "--evaluate-test" in pretrain + finetune:
                raise ValueError("independent replication must not read test rows")
            if not (args.resume and pre_output.is_file() and complete(pre_event)):
                guard(python, pretrain, prefix, "pretrain")
            if not (args.resume and final_output.is_file() and complete(final_event)):
                guard(python, finetune, prefix, "finetune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
