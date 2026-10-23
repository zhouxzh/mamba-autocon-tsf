#!/usr/bin/env python3
"""Read-only audit for the Traffic d32/24 Channel-Patch screen."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_probe_20261007"
SEED = 6090


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_frozen_core.json")
    channel_patch = load(OUTPUT / f"s{SEED}_channel_patch_calibration_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_channel_patch_calibration_pretrain.json")
    event = load(OUTPUT / f"s{SEED}_channel_patch_calibration_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (forecast, channel_patch, pretrain)):
        raise ValueError("test boundary")
    config = pretrain["config"]
    if config.get("channel_patch_relation_fraction") != 0.05:
        raise ValueError("source Channel-Patch fraction missing")
    if config.get("calibration_channel_patch_relation_fraction") != 0.05:
        raise ValueError("calibration Channel-Patch fraction missing")
    if "--channel-patch-relation-fraction" not in event["command"]:
        raise ValueError("Channel-Patch invocation missing")
    baseline = float(forecast["best_validation_mse"])
    candidate = float(channel_patch["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "forecast_calibration_mse": baseline, "channel_patch_calibration_mse": candidate,
        "channel_patch_vs_forecast_percent": (candidate / baseline - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(candidate / baseline - 1.0 <= -.0025),
        "single_scan": True, "target_core_frozen": True, "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
