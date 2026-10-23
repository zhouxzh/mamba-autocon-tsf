#!/usr/bin/env python3
"""Read-only official PatchTST audit for Weather calibrated Channel-Patch."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_weather_channel_patch_calibration_20261007"
SEEDS = (6073, 6074, 6075)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    rows = []
    for seed in SEEDS:
        candidate = load(CANDIDATE / f"s{seed}_channel_patch_calibration_frozen_core.json"); patch = load(PATCH / f"weather_h192_s{seed}_official_patchtst.json"); event = load(PATCH / f"weather_h192_s{seed}_official_patchtst_event.json")
        if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"): raise ValueError("test boundary")
        if event.get("status") != "completed" or event.get("trigger") is not None or patch.get("official_commit") != "204c21efe0b39603ad6e2ca640ef5896646ab1a9": raise ValueError("PatchTST contract failure")
        rows.append({"seed": seed, "channel_patch_calibration": float(candidate["best_validation_mse"]), "official_patchtst": float(patch["best_validation_mse"])})
    candidate = np.asarray([row["channel_patch_calibration"] for row in rows]); patch = np.asarray([row["official_patchtst"] for row in rows]); ratio = candidate / patch
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows, "channel_patch_vs_official_patchtst": {"candidate_mean_mse": float(candidate.mean()), "official_mean_mse": float(patch.mean()), "percent": float((candidate.mean() / patch.mean() - 1) * 100), "wins": int((candidate < patch).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "promotion_eligible": False, "submission_ready": False}
    result["promotion_eligible"] = bool(result["channel_patch_vs_official_patchtst"]["wins"] == 3 and result["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"])
    (PATCH / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
