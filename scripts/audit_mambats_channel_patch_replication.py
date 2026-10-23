#!/usr/bin/env python3
"""Read-only audit for the independent channel-patch replication."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006"
SEEDS = {"pems_bay": (6026, 6027, 6028), "metr_la": (6029, 6030, 6031)}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(left: np.ndarray, right: np.ndarray) -> dict:
    ratios = left / right
    return {"left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()),
            "percent": float((left.mean() / right.mean() - 1.0) * 100.0),
            "wins": int((left < right).sum()),
            "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
            "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0)}


def main() -> int:
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target, seeds in SEEDS.items():
        rows = []
        for seed in seeds:
            records = {arm: load(OUTPUT / target / f"s{seed}_{arm}_frozen_core.json") for arm in ("null", "forecast", "channel_patch")}
            source = load(OUTPUT / target / f"s{seed}_channel_patch_pretrain.json")
            if any(record.get("test_metrics_produced") for record in (*records.values(), source)):
                raise ValueError("test boundary violated")
            if float(source["config"].get("channel_patch_relation_fraction", 0.0)) != .05:
                raise ValueError("channel patch fraction mismatch")
            if not any(float(row.get("channel_patch_relation_active_fraction", 0.0)) > 0 for row in source["history"]):
                raise ValueError("channel patch objective had no active anchors")
            rows.append({"seed": seed, **{arm: float(record["best_validation_mse"]) for arm, record in records.items()}})
        null = np.asarray([row["null"] for row in rows]); forecast = np.asarray([row["forecast"] for row in rows]); candidate = np.asarray([row["channel_patch"] for row in rows])
        result["targets"][target] = {"rows": rows, "forecast_vs_null": summary(forecast, null),
                                      "channel_patch_vs_null": summary(candidate, null),
                                      "channel_patch_vs_forecast": summary(candidate, forecast)}
    result["replication_eligible"] = bool(all(
        result["targets"][target]["channel_patch_vs_forecast"]["noninferior_at_one_se"]
        and result["targets"][target]["channel_patch_vs_forecast"]["wins"] >= 2
        for target in result["targets"]
    ))
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
