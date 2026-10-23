#!/usr/bin/env python3
"""Read-only Weather calibrated Channel-Patch channelwise gate audit."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/weather_calibrated_channelwise_gate_probe_20261007"
REFERENCE = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
TIMEMIXER = ROOT / "artifacts/exploratory/official_timemixer_weather_channel_patch_calibration_20261007"
SEED = 6073


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    candidate = load(OUTPUT / f"s{SEED}_channelwise_gate.json"); scalar = load(REFERENCE / f"s{SEED}_channel_patch_calibration_frozen_core.json"); timemixer = load(TIMEMIXER / f"weather_h192_s{SEED}_official_timemixer.json")
    if any(item.get("test_metrics_produced") for item in (candidate, scalar, timemixer)): raise ValueError("test boundary")
    if not candidate["config"].get("mamba_channelwise_residual_scale"): raise ValueError("channelwise gate missing")
    if not all(row["residual_scale_shape"] == [1, 1, candidate["target_channels"]] for row in candidate["history"]): raise ValueError("gate shape mismatch")
    gate, scalar_mse, time = float(candidate["best_validation_mse"]), float(scalar["best_validation_mse"]), float(timemixer["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "channelwise_gate_mse": gate, "scalar_channel_patch_mse": scalar_mse, "official_timemixer_mse": time, "channelwise_vs_scalar_percent": (gate / scalar_mse - 1) * 100, "channelwise_vs_timemixer_percent": (gate / time - 1) * 100, "screen_passes_expansion_margin": bool(gate / scalar_mse - 1 <= -.0025 and gate < time), "final_gate_abs_max": candidate["history"][-1]["residual_scale_abs_max"], "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
