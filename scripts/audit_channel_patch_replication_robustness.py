#!/usr/bin/env python3
"""Read-only paired validation robustness audit for channel-patch ACF SSL."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/robustness_validation"
SEEDS = {"pems_bay": (6026, 6027, 6028), "metr_la": (6029, 6030, 6031)}
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(left: np.ndarray, right: np.ndarray) -> dict:
    ratios = left / right
    return {"left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()),
            "percent": float((left.mean() / right.mean() - 1.0) * 100.0), "wins": int((left < right).sum()),
            "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
            "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0)}


def main() -> int:
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target, seeds in SEEDS.items():
        conditions = {}
        for condition in CONDITIONS:
            rows = []
            for seed in seeds:
                values = {arm: load(RESULTS / f"{target}_s{seed}_{arm}.json") for arm in ("null", "forecast", "channel_patch", "official_patchtst")}
                if any(value.get("test_metrics_produced") for value in values.values()): raise ValueError("test boundary violated")
                configs = [values[arm]["conditions"][condition] for arm in values]
                if any((config["windows"], config["mask_fraction"], config["noise_scale"]) != (configs[0]["windows"], configs[0]["mask_fraction"], configs[0]["noise_scale"]) for config in configs[1:]):
                    raise ValueError("paired stress condition mismatch")
                rows.append({"seed": seed, **{arm: float(values[arm]["conditions"][condition]["mse"]) for arm in values}})
            values = {arm: np.asarray([row[arm] for row in rows]) for arm in ("null", "forecast", "channel_patch", "official_patchtst")}
            conditions[condition] = {"rows": rows, "channel_patch_vs_null": summary(values["channel_patch"], values["null"]),
                                     "channel_patch_vs_forecast": summary(values["channel_patch"], values["forecast"]),
                                     "channel_patch_vs_official_patchtst": summary(values["channel_patch"], values["official_patchtst"])}
        result["targets"][target] = {"conditions": conditions,
                                      "all_external_noninferior": all(row["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"] for row in conditions.values())}
    result["all_external_noninferior"] = all(item["all_external_noninferior"] for item in result["targets"].values())
    (RESULTS / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
