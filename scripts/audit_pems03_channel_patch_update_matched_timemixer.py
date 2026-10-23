#!/usr/bin/env python3
"""Read-only update-matched PEMS03 Channel-Patch versus TimeMixer audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_update_matched_timemixer_20261007"
TIMEMIXER_BASE = ROOT / "artifacts/exploratory/official_timemixer_pems03_channel_patch_25pct_20261007"
TIMEMIXER_EXPANSION = ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007"
SEEDS = (6060, 6061, 6062, 6065, 6066)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def timemixer_path(seed):
    root = TIMEMIXER_BASE if seed in (6060, 6061, 6062) else TIMEMIXER_EXPANSION
    return root / f"pems03_h192_s{seed}_official_timemixer.json"


def main():
    plan = load(OUTPUT / "plan.json")
    if plan.get("candidate_updates") != plan.get("timemixer_updates") or plan.get("candidate_updates") != 160: raise ValueError("update-matching contract failure")
    rows = []
    for seed in SEEDS:
        candidate = load(OUTPUT / f"s{seed}_channel_patch_epoch10.json"); timemixer = load(timemixer_path(seed)); event = load(OUTPUT / f"s{seed}_channel_patch_epoch10_event.json")
        if candidate.get("test_metrics_produced") or timemixer.get("test_metrics_produced"): raise ValueError("test boundary")
        if int(candidate["config"]["epochs"]) != 10 or int(candidate["config"]["batch_size"]) != 4: raise ValueError("candidate budget mismatch")
        if int(timemixer["config"]["epochs"]) != 5 or int(timemixer["config"]["batch_size"]) != 2: raise ValueError("TimeMixer budget mismatch")
        if event.get("status") != "completed" or event.get("trigger") is not None: raise ValueError("GPU guard failure")
        rows.append({"seed": seed, "channel_patch_epoch10": float(candidate["best_validation_mse"]), "official_timemixer": float(timemixer["best_validation_mse"]), "candidate_best_epoch": int(min(candidate["history"], key=lambda item: item["validation_mse"])["epoch"])})
    candidate = np.asarray([row["channel_patch_epoch10"] for row in rows]); timemixer = np.asarray([row["official_timemixer"] for row in rows]); ratio = candidate / timemixer
    result = {"validation_only": True, "test_metrics_produced": False, "same_update_count": True, "rows": rows, "channel_patch_vs_official_timemixer": {"candidate_mean_mse": float(candidate.mean()), "official_mean_mse": float(timemixer.mean()), "percent": float((candidate.mean() / timemixer.mean() - 1) * 100), "wins": int((candidate < timemixer).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
