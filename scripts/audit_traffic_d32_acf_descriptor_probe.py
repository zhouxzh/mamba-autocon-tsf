#!/usr/bin/env python3
"""Read-only audit for ACF-conditioned Traffic patch representation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_probe_20261007"
SEED = 6095


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_frozen_core.json")
    channel_patch = load(OUTPUT / f"s{SEED}_channel_patch_frozen_core.json")
    descriptor = load(OUTPUT / f"s{SEED}_descriptor_channel_patch_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_descriptor_channel_patch_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (forecast, channel_patch, descriptor, pretrain)):
        raise ValueError("test boundary")
    if not pretrain["config"].get("mamba_ts_acf_channel_descriptor"):
        raise ValueError("descriptor pretraining flag missing")
    if pretrain["config"].get("channel_patch_relation_fraction") != .05:
        raise ValueError("Channel-Patch contract missing")
    if not all(source.get("acf_channel_descriptor") for source in pretrain["sources"]):
        raise ValueError("source descriptor provenance missing")
    values = {
        "forecast": float(forecast["best_validation_mse"]),
        "channel_patch": float(channel_patch["best_validation_mse"]),
        "descriptor_channel_patch": float(descriptor["best_validation_mse"]),
    }
    result = {
        "validation_only": True, "test_metrics_produced": False, "mse": values,
        "descriptor_vs_forecast_percent": (values["descriptor_channel_patch"] / values["forecast"] - 1.0) * 100.0,
        "descriptor_vs_channel_patch_percent": (values["descriptor_channel_patch"] / values["channel_patch"] - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(
            values["descriptor_channel_patch"] / values["forecast"] - 1.0 <= -.0025
            and values["descriptor_channel_patch"] / values["channel_patch"] - 1.0 <= -.0025
        ),
        "single_scan": True, "inference_descriptor_fixed_train_prefix": True,
        "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
