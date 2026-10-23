#!/usr/bin/env python3
"""Read-only PEMS03 external robustness audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007/robustness_validation"
SEEDS = (6055, 6056, 6057)
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary(left, right):
    ratio = left / right
    upper = ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))
    return {
        "left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()),
        "percent": float((left.mean() / right.mean() - 1) * 100), "wins": int((left < right).sum()),
        "ratio_upper_1se": float(upper), "noninferior_at_one_se": bool(upper <= 1),
    }


def main():
    plan = load(RESULTS / "plan.json")
    if not plan.get("validation_only") or not plan.get("target_excluded_from_source_and_initialization"):
        raise ValueError("invalid PEMS03 robustness plan")
    result = {"validation_only": True, "test_metrics_produced": False, "target_excluded_from_source_and_initialization": True, "conditions": {}, "submission_ready": False}
    for condition in CONDITIONS:
        rows = []
        for seed in SEEDS:
            records = {arm: load(RESULTS / f"s{seed}_{arm}.json") for arm in ("null", "forecast", "channel_patch", "official_patchtst")}
            if any(item.get("test_metrics_produced") for item in records.values()):
                raise ValueError("test boundary")
            selected = {arm: records[arm]["conditions"][condition] for arm in records}
            first = next(iter(selected.values()))
            if any((item["windows"], item["mask_fraction"], item["noise_scale"]) != (first["windows"], first["mask_fraction"], first["noise_scale"]) for item in selected.values()):
                raise ValueError("condition mismatch")
            rows.append({"seed": seed, **{arm: float(item["mse"]) for arm, item in selected.items()}})
        metrics = {arm: np.asarray([row[arm] for row in rows]) for arm in ("null", "forecast", "channel_patch", "official_patchtst")}
        result["conditions"][condition] = {
            "rows": rows,
            "channel_patch_vs_null": summary(metrics["channel_patch"], metrics["null"]),
            "channel_patch_vs_forecast": summary(metrics["channel_patch"], metrics["forecast"]),
            "channel_patch_vs_official_patchtst": summary(metrics["channel_patch"], metrics["official_patchtst"]),
        }
    result["all_external_noninferior"] = all(cell["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"] for cell in result["conditions"].values())
    (RESULTS / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
