#!/usr/bin/env python3
"""Read-only audit for Weather target-prefix Channel-Patch calibration screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_target_calibration_probe_20261007"
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006"
SEED = 6070


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_frozen_core.json"); channel_patch = load(OUTPUT / f"s{SEED}_channel_patch_calibration_frozen_core.json"); source = load(OUTPUT / f"s{SEED}_channel_patch_calibration_pretrain.json"); event = load(OUTPUT / f"s{SEED}_channel_patch_calibration_pretrain_event.json"); uncalibrated = load(REFERENCE / "s5991_active_frozen_core.json")
    if any(item.get("test_metrics_produced") for item in (forecast, channel_patch, source, uncalibrated)): raise ValueError("test boundary")
    roles = {item["name"]: item for item in source["sources"]}
    if roles["Weather"]["role"] != "target_calibration" or roles["Weather"]["train_prefix_fraction"] != .06: raise ValueError("Weather prefix boundary")
    if roles["Weather"]["relation_loss_fraction"] != .1: raise ValueError("calibration relation schedule")
    if "--channel-patch-relation-fraction" not in event["command"]: raise ValueError("channel patch loss missing")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in source["history"]): raise ValueError("patch relation inactive")
    forecast_mse, patch_mse, uncalibrated_mse = float(forecast["best_validation_mse"]), float(channel_patch["best_validation_mse"]), float(uncalibrated["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "forecast_calibration_mse": forecast_mse, "channel_patch_calibration_mse": patch_mse, "uncalibrated_forecast_mse": uncalibrated_mse, "channel_patch_vs_forecast_calibration_percent": (patch_mse / forecast_mse - 1) * 100, "channel_patch_vs_uncalibrated_forecast_percent": (patch_mse / uncalibrated_mse - 1) * 100, "screen_passes_expansion_margin": bool(patch_mse / forecast_mse - 1 <= -.0025 and patch_mse < uncalibrated_mse), "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
