#!/usr/bin/env python3
"""Read-only audit for independent Electricity channel-patch replication."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006"
SEEDS = (6033, 6034, 6035)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        forecast = load(OUTPUT / f"s{seed}_forecast_frozen_core.json")
        candidate = load(OUTPUT / f"s{seed}_channel_patch_frozen_core.json")
        source = load(OUTPUT / f"s{seed}_channel_patch_pretrain.json")
        if any(item.get("test_metrics_produced") for item in (forecast, candidate, source)): raise ValueError("test boundary violated")
        if float(source["config"].get("channel_patch_relation_fraction", 0.0)) != .05: raise ValueError("contract mismatch")
        if not all(float(item.get("channel_patch_relation_active_fraction", 0.0)) > 0 for item in source["history"]): raise ValueError("inactive channel patch anchors")
        rows.append({"seed": seed, "forecast": float(forecast["best_validation_mse"]), "channel_patch": float(candidate["best_validation_mse"])})
    forecast = np.asarray([row["forecast"] for row in rows]); candidate = np.asarray([row["channel_patch"] for row in rows]); ratios = candidate / forecast
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "channel_patch_vs_forecast": {"channel_patch_mean_mse": float(candidate.mean()), "forecast_mean_mse": float(forecast.mean()),
                                              "percent": float((candidate.mean() / forecast.mean() - 1.0) * 100.0), "wins": int((candidate < forecast).sum()),
                                              "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
                                              "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0)},
              "submission_ready": False}
    result["external_comparator_eligible"] = bool(result["channel_patch_vs_forecast"]["noninferior_at_one_se"] and result["channel_patch_vs_forecast"]["wins"] >= 2)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
