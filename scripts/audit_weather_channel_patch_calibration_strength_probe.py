#!/usr/bin/env python3
"""Read-only Weather strong target-calibration Channel-Patch screen audit."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_strength_probe_20261007"
PRIOR = ROOT / "artifacts/exploratory/weather_channel_patch_target_calibration_probe_20261007"
SEED = 6071


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_0p30_frozen_core.json"); patch = load(OUTPUT / f"s{SEED}_channel_patch_calibration_0p30_frozen_core.json"); pretrain = load(OUTPUT / f"s{SEED}_channel_patch_calibration_0p30_pretrain.json"); event = load(OUTPUT / f"s{SEED}_channel_patch_calibration_0p30_pretrain_event.json"); prior = load(PRIOR / "audit.json")
    if any(item.get("test_metrics_produced") for item in (forecast, patch, pretrain)): raise ValueError("test boundary")
    roles = {item["name"]: item for item in pretrain["sources"]}
    if roles["Weather"]["role"] != "target_calibration" or roles["Weather"]["train_prefix_fraction"] != .06 or roles["Weather"]["relation_loss_fraction"] != .3: raise ValueError("strong calibration prefix contract")
    if "--channel-patch-relation-fraction" not in event["command"]: raise ValueError("channel patch loss missing")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in pretrain["history"]): raise ValueError("patch relation inactive")
    forecast_mse, patch_mse = float(forecast["best_validation_mse"]), float(patch["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "forecast_calibration_0p30_mse": forecast_mse, "channel_patch_calibration_0p30_mse": patch_mse, "prior_channel_patch_calibration_0p10_mse": prior["channel_patch_calibration_mse"], "channel_patch_vs_forecast_calibration_percent": (patch_mse / forecast_mse - 1) * 100, "channel_patch_vs_prior_0p10_percent": (patch_mse / prior["channel_patch_calibration_mse"] - 1) * 100, "screen_passes_expansion_margin": bool(patch_mse / forecast_mse - 1 <= -.0025 and patch_mse < prior["channel_patch_calibration_mse"]), "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
