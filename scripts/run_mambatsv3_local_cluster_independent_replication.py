#!/usr/bin/env python3
"""Portable PEMS-BAY H192 local-cluster active/null replication packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008"
SEEDS = (6130, 6131, 6132)
SOURCES = (
    "PEMS07:dataset/PEMS07/PEMS07.npz:npz",
    "PEMS08:dataset/PEMS08/PEMS08.npz:npz",
    "PEMS04:dataset/PEMS04/PEMS04.npz:npz",
)
INITIAL = (
    "artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems03_MambaTSv3AutoCon_h12_s2131_l0p001_ca786fbe2493c_MambaTSv3AutoCon_pems03_ftM_sl96_ll48_pl12_dm2_h00dcc565bb3086f9/checkpoint.pth",
    "artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems04_MambaTSv3AutoCon_h12_s2131_l0p001_c22b7150b652d_MambaTSv3AutoCon_pems04_ftM_sl96_ll48_pl12_dm2_h179e4641a67e8f06/checkpoint.pth",
)
TRACKED = (
    "models/MambaTS.py", "models/MambaDLinearHybrid.py", "layers/losses.py",
    "data_provider/samplers.py", "scripts/pretrain_mambatsv3_core_autocon.py",
    "scripts/finetune_mambatsv3_dlinear_hybrid.py", "scripts/evaluate_lowdata_checkpoint_test.py",
    "scripts/run_gpu_guarded.py", "dataset/PEMS-BAY/PEMS-BAY.csv",
    "dataset/PEMS07/PEMS07.npz", "dataset/PEMS08/PEMS08.npz", "dataset/PEMS04/PEMS04.npz",
    *INITIAL,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete(path: Path) -> bool:
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guard(python: Path, command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        str(python), str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + f"_{stage}_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + f"_{stage}_event.json")),
        "--", *command,
    ], cwd=ROOT, check=True)


def pretrain_command(python: Path, output: Path, seed: int, active: bool) -> list[str]:
    return [
        str(python), "scripts/pretrain_mambatsv3_core_autocon.py",
        "--source", SOURCES[0], "--source", SOURCES[1], "--source", SOURCES[2],
        "--initial-checkpoint", INITIAL[0], "--initial-checkpoint", INITIAL[1],
        "--output", str(output), "--seq-len", "96", "--patch-len", "24", "--patch-stride", "24",
        "--source-pred-len", "192", "--d-model", "24", "--d-ff", "48", "--mamba-d-state", "8",
        "--mamba-d-conv", "2", "--mamba-expand", "1", "--context-rank", "8", "--train-fraction", ".6",
        "--batch-size", "4", "--epochs", "3", "--max-windows", "64", "--relation-pair-policy", "contrastive_cluster",
        "--forecast-contrastive-pairing", "channel_coverage", "--min-channel-positive-support", ".5",
        "--reconstruction-weight", "0", "--predictive-weight", "1", "--prediction-target", "last_value_residual",
        "--pretrain-task", "next_patch", "--relation-view", "forecast_contrastive",
        "--mamba-lambda", ".1" if active else "0", "--relation-loss-fraction", ".1" if active else "0",
        "--learning-rate", ".0003", "--seed", str(seed),
    ]


def finetune_command(python: Path, source: Path, output: Path, seed: int) -> list[str]:
    return [
        str(python), "scripts/finetune_mambatsv3_dlinear_hybrid.py", "--dataset", "dataset/PEMS-BAY/PEMS-BAY.csv",
        "--source-checkpoint", str(source), "--output", str(output), "--seq-len", "96", "--pred-len", "192",
        "--patch-len", "24", "--patch-stride", "24", "--d-model", "24", "--d-ff", "48",
        "--mamba-d-state", "8", "--mamba-d-conv", "2", "--mamba-expand", "1", "--context-rank", "16",
        "--channel-agnostic-scan", "--transfer-prediction-head", "--prediction-head-transfer-mode", "exact",
        "--linear-model", "NLinear", "--mamba-correction-output", "--few-shot-fraction", ".1",
        "--train-fraction", ".6", "--val-fraction", ".2", "--batch-size", "4", "--epochs", "5",
        "--max-train-windows", "64", "--freeze-mamba-epochs", "1", "--seed", str(seed),
    ]


def test_command(python: Path, checkpoint: Path, output: Path, seed: int) -> list[str]:
    return [
        str(python), "scripts/evaluate_lowdata_checkpoint_test.py", "--dataset", "dataset/PEMS-BAY/PEMS-BAY.csv",
        "--checkpoint", str(checkpoint), "--output", str(output), "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--open-test-after-validation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    python = args.python.resolve()
    output = args.output.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    missing = [item for item in TRACKED if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError("missing protocol inputs: {}".format(", ".join(missing)))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "packet_only" if args.manifest_only else "registered_before_training",
        "independent_machine": "unverified_operator_claim", "machine_id": args.machine_id,
        "python": str(python), "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": .1,
        "seeds": list(SEEDS), "arms": ["active", "null"], "source_pool": ["PEMS07", "PEMS08", "PEMS04"],
        "initialization_sources": ["PEMS03", "PEMS04"], "target_excluded_from_source_and_initialization": True,
        "fixed_contract": {"mixer": "mamba1", "patch": "24/24", "d_model": 24, "d_ff": 48, "mamba_scan_count": 1, "target_core_frozen": False, "relation_pair_policy": "contrastive_cluster", "forecast_contrastive_pairing": "channel_coverage", "min_channel_positive_support": .5, "relation_loss_fraction": .1},
        "thermal_guard": {"temperature_limit_c": 78, "fan_limit_percent": 85, "interval_sec": 5},
        "validation_gate_before_test": "all active source histories have coverage >= .8 and every active validation MSE is no larger than its paired null",
        "test_open_requested": bool(args.open_test_after_validation),
        "source_hashes": {item: digest(ROOT / item) for item in TRACKED},
        "independent_machine_acceptance": "an authorized reviewer must verify a distinct physical host and environment; this runner does not self-certify independence",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.manifest_only:
        return 0
    for seed in SEEDS:
        for arm in ("active", "null"):
            prefix = output / f"s{seed}_{arm}"
            pretrain = prefix.with_name(prefix.name + "_pretrain.pt")
            hybrid = prefix.with_name(prefix.name + "_hybrid.pt")
            if not (args.resume and pretrain.is_file() and complete(prefix.with_name(prefix.name + "_pretrain_event.json"))):
                guard(python, pretrain_command(python, pretrain, seed, arm == "active"), prefix, "pretrain")
            if not (args.resume and hybrid.is_file() and complete(prefix.with_name(prefix.name + "_finetune_event.json"))):
                guard(python, finetune_command(python, pretrain, hybrid, seed), prefix, "finetune")
    active_payloads = [load(output / f"s{seed}_active_pretrain.json") for seed in SEEDS]
    active_models = [load(output / f"s{seed}_active_hybrid.json") for seed in SEEDS]
    null_models = [load(output / f"s{seed}_null_hybrid.json") for seed in SEEDS]
    gate = all(min(float(row["global_relation_active_fraction"]) for row in payload["history"]) >= .8 for payload in active_payloads) and all(float(active["best_validation_mse"]) <= float(null["best_validation_mse"]) for active, null in zip(active_models, null_models))
    (output / "validation_gate.json").write_text(json.dumps({"passes": gate, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    if args.open_test_after_validation and gate:
        for seed in SEEDS:
            for arm in ("active", "null"):
                prefix = output / f"s{seed}_{arm}"
                guard(python, test_command(python, prefix.with_name(prefix.name + "_hybrid.pt"), prefix.with_name(prefix.name + "_test.json"), seed), prefix, "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
