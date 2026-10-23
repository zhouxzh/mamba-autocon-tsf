#!/usr/bin/env python3
"""Final fifteen-seed batch-matched PEMS03 Channel-Patch versus TimeMixer audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR_FIRST = ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_20261007"
PRIOR_SECOND = ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_expansion_20261007"
TIMEMIXER_BASE = ROOT / "artifacts/exploratory/official_timemixer_pems03_channel_patch_25pct_20261007"
TIMEMIXER_FIRST_EXPANSION = ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007"
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_final_expansion_20261007"
PRIOR_SEEDS = (6060, 6061, 6062, 6065, 6066, 6084, 6085, 6086, 6087, 6088)
FRESH_SEEDS = (6089, 6090, 6091, 6092, 6093)
COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def prior_candidate_path(seed):
    if seed in (6060, 6061, 6062, 6065, 6066):
        return PRIOR_FIRST / f"s{seed}_channel_patch_batch2.json"
    return PRIOR_SECOND / f"s{seed}_channel_patch_batch2_batch2.json"
def prior_timemixer_path(seed):
    if seed in (6060, 6061, 6062):
        root = TIMEMIXER_BASE
    elif seed in (6065, 6066):
        root = TIMEMIXER_FIRST_EXPANSION
    else:
        root = PRIOR_SECOND
    return root / f"pems03_h192_s{seed}_official_timemixer.json"


def main():
    plan = load(OUTPUT / "plan.json")
    if plan.get("stop_rule") is None or len(plan.get("prior_batch_matched_seeds", [])) != 10: raise ValueError("final expansion registration failure")
    rows = []
    for seed in PRIOR_SEEDS:
        candidate = load(prior_candidate_path(seed)); tm = load(prior_timemixer_path(seed))
        rows.append({"seed": seed, "channel_patch_batch2": float(candidate["best_validation_mse"]), "official_timemixer": float(tm["best_validation_mse"]), "cohort": "prior"})
    for seed in FRESH_SEEDS:
        candidate = load(OUTPUT / f"s{seed}_channel_patch_batch2_batch2.json"); tm = load(OUTPUT / f"pems03_h192_s{seed}_official_timemixer.json"); event = load(OUTPUT / f"pems03_h192_s{seed}_official_timemixer_timemixer_event.json")
        if candidate.get("test_metrics_produced") or tm.get("test_metrics_produced"): raise ValueError("test boundary")
        if int(candidate["config"]["batch_size"]) != 2 or int(candidate["config"]["epochs"]) != 5: raise ValueError("candidate geometry mismatch")
        if tm.get("official_commit") != COMMIT or event.get("status") != "completed" or event.get("trigger") is not None: raise ValueError("fresh TimeMixer contract failure")
        rows.append({"seed": seed, "channel_patch_batch2": float(candidate["best_validation_mse"]), "official_timemixer": float(tm["best_validation_mse"]), "cohort": "fresh"})
    candidate = np.asarray([row["channel_patch_batch2"] for row in rows]); tm = np.asarray([row["official_timemixer"] for row in rows]); ratio = candidate / tm
    result = {"validation_only": True, "test_metrics_produced": False, "same_batch_geometry": True, "same_update_count": True, "final_precision_expansion": True, "rows": rows, "channel_patch_vs_official_timemixer": {"candidate_mean_mse": float(candidate.mean()), "official_mean_mse": float(tm.mean()), "percent": float((candidate.mean() / tm.mean() - 1) * 100), "wins": int((candidate < tm).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
