#!/usr/bin/env python3
"""Read-only Weather audit for channel-patch SSL against locked controls."""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006"
REFERENCE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_weather_mamba1_10pct_independent_replication_20261006"
SEEDS = (5991, 5992, 5993, 5994, 5995)
def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def summary(left, right):
    ratio = left / right
    return {"left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()), "percent": float((left.mean() / right.mean() - 1) * 100), "wins": int((left < right).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}
def main() -> int:
    rows = []
    for seed in SEEDS:
        candidate = load(CANDIDATE / f"s{seed}_channel_patch_frozen_core.json"); forecast = load(REFERENCE / f"s{seed}_active_frozen_core.json"); null = load(REFERENCE / f"s{seed}_null_frozen_core.json"); patch = load(PATCH / f"weather_h192_s{seed}_official_patchtst.json")
        if any(item.get("test_metrics_produced") for item in (candidate, forecast, null, patch)): raise ValueError("test boundary violated")
        source = load(CANDIDATE / f"s{seed}_channel_patch_pretrain.json")
        if float(source["config"].get("channel_patch_relation_fraction", 0.0)) != .05: raise ValueError("channel patch contract mismatch")
        if not all(float(item.get("channel_patch_relation_active_fraction", 0.0)) > 0 for item in source["history"]): raise ValueError("inactive channel-patch anchors")
        rows.append({"seed": seed, "channel_patch": float(candidate["best_validation_mse"]), "forecast": float(forecast["best_validation_mse"]), "null": float(null["best_validation_mse"]), "official_patchtst": float(patch["best_validation_mse"])})
    values = {key: np.asarray([row[key] for row in rows]) for key in ("channel_patch", "forecast", "null", "official_patchtst")}
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "channel_patch_vs_forecast": summary(values["channel_patch"], values["forecast"]), "channel_patch_vs_null": summary(values["channel_patch"], values["null"]), "channel_patch_vs_official_patchtst": summary(values["channel_patch"], values["official_patchtst"]), "submission_ready": False}
    result["robustness_eligible"] = bool(result["channel_patch_vs_forecast"]["noninferior_at_one_se"] and result["channel_patch_vs_forecast"]["wins"] >= 3 and result["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"])
    (CANDIDATE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
