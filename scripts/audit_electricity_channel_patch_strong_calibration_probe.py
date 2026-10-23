#!/usr/bin/env python3
"""Read-only audit for Electricity strong-calibration channel-patch gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_probe_20261006"
SEED = 6032


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    forecast = load(OUTPUT / f"s{SEED}_forecast_frozen_core.json")
    candidate = load(OUTPUT / f"s{SEED}_channel_patch_frozen_core.json")
    source = load(OUTPUT / f"s{SEED}_channel_patch_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (forecast, candidate, source)):
        raise ValueError("test boundary violated")
    if float(source["config"].get("channel_patch_relation_fraction", 0.0)) != .05:
        raise ValueError("channel-patch contract mismatch")
    active = [float(row.get("channel_patch_relation_active_fraction", 0.0)) for row in source["history"]]
    if not active or not all(value > 0 for value in active):
        raise ValueError("channel-patch relation had inactive epochs")
    left, right = float(candidate["best_validation_mse"]), float(forecast["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "seed": SEED,
              "forecast_mse": right, "channel_patch_mse": left,
              "channel_patch_vs_forecast_percent": (left / right - 1.0) * 100.0,
              "channel_patch_better": bool(left < right),
              "channel_patch_active_anchor_fractions": active, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
