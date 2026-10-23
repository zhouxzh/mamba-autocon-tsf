#!/usr/bin/env python3
"""One-seed registered screen for a shared channel-statistic residual gate."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006"
OUTPUT = ROOT / "artifacts/exploratory/electricity_d32_shared_statistic_gate_probe_20261007"
SEED = 6047


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace(command, flag, value):
    updated = list(command)
    updated[updated.index(flag) + 1] = value
    return updated


def main():
    source = REFERENCE / f"s{SEED}_channel_patch_pretrain.pt"
    event = REFERENCE / f"s{SEED}_channelwise_gate_event.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    command = list(load(event)["command"])
    command[0] = sys.executable
    command = [item for item in command if item != "--mamba-channelwise-residual-scale"]
    prefix = OUTPUT / f"s{SEED}_shared_statistic_gate"
    command = replace(command, "--source-checkpoint", str(source))
    command = replace(command, "--output", str(prefix.with_suffix(".pt")))
    command.extend([
        "--mamba-shared-channel-statistic-residual-gate",
        "--mamba-shared-channel-statistic-gate-rank", "8",
    ])
    if "--evaluate-test" in command:
        raise ValueError("shared-statistic gate screen cannot read test rows")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_training", "validation_only": True,
        "test_metrics_produced": False, "target": "Electricity", "horizon": 96,
        "few_shot_fraction": 0.25, "seed": SEED, "source_pretraining": str(source),
        "controls": {
            "global_gate": str(REFERENCE / f"s{SEED}_global_gate.pt"),
            "channelwise_gate": str(REFERENCE / f"s{SEED}_channelwise_gate.pt"),
        },
        "mechanism": "zero-initialized shared MLP over each channel's last value, local mean, local standard deviation, and endpoint trend",
        "architecture": {"mixer": "mamba1", "patch": "24/24", "d_model": 32, "d_ff": 64, "mamba_scan_count": 1, "gate_rank": 8},
        "screen_rule": "expand only if the shared gate beats the existing channelwise-gate validation MSE by at least 0.25 percent",
        "shared_parameter_contract": "no channel-indexed gate parameter; zero initialized output preserves the linear primary forecast initially",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
