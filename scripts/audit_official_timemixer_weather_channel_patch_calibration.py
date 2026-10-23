#!/usr/bin/env python3
"""Read-only official TimeMixer audit for Weather calibrated Channel-Patch."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007"
OFFICIAL = ROOT / "artifacts/exploratory/official_timemixer_weather_channel_patch_calibration_20261007"
COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
SEEDS = (6073, 6074, 6075)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    rows = []
    for seed in SEEDS:
        candidate = load(CANDIDATE / f"s{seed}_channel_patch_calibration_frozen_core.json"); official = load(OFFICIAL / f"weather_h192_s{seed}_official_timemixer.json"); event = load(OFFICIAL / f"weather_h192_s{seed}_official_timemixer_event.json")
        if candidate.get("test_metrics_produced") or official.get("test_metrics_produced"): raise ValueError("test boundary")
        if event.get("status") != "completed" or event.get("trigger") is not None or official.get("official_commit") != COMMIT: raise ValueError("TimeMixer contract failure")
        rows.append({"seed": seed, "channel_patch_calibration": float(candidate["best_validation_mse"]), "official_timemixer": float(official["best_validation_mse"])})
    candidate = np.asarray([row["channel_patch_calibration"] for row in rows]); official = np.asarray([row["official_timemixer"] for row in rows]); ratio = candidate / official
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows, "channel_patch_vs_official_timemixer": {"candidate_mean_mse": float(candidate.mean()), "official_mean_mse": float(official.mean()), "percent": float((candidate.mean() / official.mean() - 1) * 100), "wins": int((candidate < official).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "submission_ready": False}
    (OFFICIAL / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
