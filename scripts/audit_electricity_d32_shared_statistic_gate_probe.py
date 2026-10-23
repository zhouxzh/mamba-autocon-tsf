#!/usr/bin/env python3
"""Read-only audit for the Electricity shared channel-statistic gate screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/electricity_d32_shared_statistic_gate_probe_20261007"
REFERENCE = ROOT / "artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006"
SEED = 6047


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    candidate = load(OUTPUT / f"s{SEED}_shared_statistic_gate.json")
    channelwise = load(REFERENCE / f"s{SEED}_channelwise_gate.json")
    global_gate = load(REFERENCE / f"s{SEED}_global_gate.json")
    if any(item.get("test_metrics_produced") for item in (candidate, channelwise, global_gate)):
        raise ValueError("test boundary")
    config = candidate["config"]
    if not config.get("mamba_shared_channel_statistic_residual_gate"):
        raise ValueError("shared statistic gate missing")
    if config.get("mamba_channelwise_residual_scale"):
        raise ValueError("screen must not contain channel-indexed gate")
    history = candidate["history"]
    if not all(item["shared_channel_statistic_residual_gate"] and item["statistic_gate_abs_max"] > 0 for item in history):
        raise ValueError("shared gate did not become active")
    candidate_mse = float(candidate["best_validation_mse"])
    channelwise_mse = float(channelwise["best_validation_mse"])
    global_mse = float(global_gate["best_validation_mse"])
    improvement = (candidate_mse / channelwise_mse - 1) * 100
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "shared_statistic_gate_mse": candidate_mse, "channelwise_gate_mse": channelwise_mse,
        "global_gate_mse": global_mse,
        "shared_vs_channelwise_percent": improvement,
        "shared_vs_global_percent": (candidate_mse / global_mse - 1) * 100,
        "screen_passes_expansion_margin": bool(improvement <= -0.25),
        "final_statistic_gate_abs_max": history[-1]["statistic_gate_abs_max"],
        "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
