#!/usr/bin/env python3
"""Read-only audit for transferable post-scan Traffic mixer."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_postscan_mixer_probe_rerun_20261007"
SEED = 6098


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_frozen_core.json")
    channel = load(OUTPUT / f"s{SEED}_channel_patch_frozen_core.json")
    candidate = load(OUTPUT / f"s{SEED}_postscan_channel_patch_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_postscan_channel_patch_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (forecast, channel, candidate, pretrain)):
        raise ValueError("test boundary")
    if not pretrain["config"].get("mamba_ts_shared_postscan_mixer"):
        raise ValueError("postscan mixer missing")
    if not any(key.startswith("ts_shared_postscan_mixer.") for key in pretrain["shared_keys"]):
        raise ValueError("mixer not transferred")
    values = {"forecast": float(forecast["best_validation_mse"]), "channel_patch": float(channel["best_validation_mse"]), "postscan_channel_patch": float(candidate["best_validation_mse"])}
    result = {"validation_only": True, "test_metrics_produced": False, "mse": values, "postscan_vs_forecast_percent": (values["postscan_channel_patch"] / values["forecast"] - 1) * 100, "postscan_vs_channel_patch_percent": (values["postscan_channel_patch"] / values["channel_patch"] - 1) * 100, "screen_passes_expansion_margin": bool(values["postscan_channel_patch"] / values["forecast"] - 1 <= -.0025 and values["postscan_channel_patch"] / values["channel_patch"] - 1 <= -.0025), "single_mamba_scan": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
