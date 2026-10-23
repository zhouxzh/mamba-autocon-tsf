#!/usr/bin/env python3
"""Read-only independent Weather calibrated Channel-Patch replication audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
SEEDS = (6073, 6074, 6075)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    rows = []
    for seed in SEEDS:
        forecast = load(OUTPUT / f"s{seed}_forecast_calibration_frozen_core.json"); patch = load(OUTPUT / f"s{seed}_channel_patch_calibration_frozen_core.json"); source = load(OUTPUT / f"s{seed}_channel_patch_calibration_pretrain.json")
        if any(item.get("test_metrics_produced") for item in (forecast, patch, source)): raise ValueError("test boundary")
        roles = {item["name"]: item for item in source["sources"]}
        if any(roles[name]["channel_patch_relation_fraction"] != .05 for name in roles): raise ValueError("role-specific patch schedule")
        if roles["Weather"]["role"] != "target_calibration" or roles["Weather"]["train_prefix_fraction"] != .06: raise ValueError("Weather calibration prefix")
        if not all(item["channel_patch_relation_active_fraction"] > 0 for item in source["history"]): raise ValueError("patch relation inactive")
        rows.append({"seed": seed, "forecast_calibration": float(forecast["best_validation_mse"]), "channel_patch_calibration": float(patch["best_validation_mse"])})
    forecast = np.asarray([row["forecast_calibration"] for row in rows]); patch = np.asarray([row["channel_patch_calibration"] for row in rows]); ratio = patch / forecast
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows, "channel_patch_vs_forecast_calibration": {"channel_patch_mean_mse": float(patch.mean()), "forecast_mean_mse": float(forecast.mean()), "percent": float((patch.mean() / forecast.mean() - 1) * 100), "wins": int((patch < forecast).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "promotion_eligible": False, "submission_ready": False}
    result["promotion_eligible"] = bool(result["channel_patch_vs_forecast_calibration"]["wins"] == 3 and result["channel_patch_vs_forecast_calibration"]["noninferior_at_one_se"])
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
