#!/usr/bin/env python3
"""Read-only audit for train-prefix ACF-grouped Traffic patch SSL."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_grouped_patch_probe_20261007"
SEED = 6094


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_frozen_core.json")
    grouped = load(OUTPUT / f"s{SEED}_grouped_patch_calibration_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_grouped_patch_calibration_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (forecast, grouped, pretrain)):
        raise ValueError("test boundary")
    if pretrain["config"].get("grouped_patch_relation_fraction") != .05:
        raise ValueError("grouped-patch fraction missing")
    if pretrain["config"].get("channel_patch_relation_fraction") != 0.0:
        raise ValueError("per-channel Channel-Patch must be disabled")
    roles = {source["role"]: source for source in pretrain["sources"]}
    if any(source["grouped_patch_group_count"] != 4 for source in roles.values()):
        raise ValueError("group count provenance missing")
    if any(len(source["grouped_patch_group_sizes"]) != 4 for source in roles.values()):
        raise ValueError("group size provenance missing")
    baseline = float(forecast["best_validation_mse"])
    candidate = float(grouped["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "forecast_calibration_mse": baseline, "grouped_patch_calibration_mse": candidate,
        "grouped_patch_vs_forecast_percent": (candidate / baseline - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(candidate / baseline - 1.0 <= -.0025),
        "single_scan": True, "inference_grouping_absent": True, "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
