#!/usr/bin/env python3
"""Five-seed PEMS03/25% Channel-Patch versus official TimeMixer audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_CANDIDATE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
BASE_TIMEMIXER = ROOT / "artifacts/exploratory/official_timemixer_pems03_channel_patch_25pct_20261007"
EXPANSION = ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007"
COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
SEEDS = (6060, 6061, 6062, 6065, 6066)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def paths(seed):
    if seed in (6060, 6061, 6062):
        return BASE_CANDIDATE / f"s{seed}_channel_patch_frozen_core.json", BASE_TIMEMIXER / f"pems03_h192_s{seed}_official_timemixer.json", BASE_TIMEMIXER / f"pems03_h192_s{seed}_official_timemixer_event.json"
    return EXPANSION / f"s{seed}_channel_patch_frozen_core.json", EXPANSION / f"pems03_h192_s{seed}_official_timemixer.json", EXPANSION / f"pems03_h192_s{seed}_official_timemixer_timemixer_event.json"


def main():
    plan = load(EXPANSION / "plan.json")
    if plan.get("aggregation_rule") is None: raise ValueError("unregistered expansion")
    rows = []
    for seed in SEEDS:
        candidate_path, baseline_path, event_path = paths(seed); candidate, baseline, event = load(candidate_path), load(baseline_path), load(event_path)
        if candidate.get("test_metrics_produced") or baseline.get("test_metrics_produced"): raise ValueError("test boundary")
        if baseline.get("official_commit") != COMMIT or event.get("status") != "completed" or event.get("trigger") is not None: raise ValueError("baseline contract failure")
        rows.append({"seed": seed, "channel_patch": float(candidate["best_validation_mse"]), "official_timemixer": float(baseline["best_validation_mse"])})
    candidate = np.asarray([row["channel_patch"] for row in rows]); baseline = np.asarray([row["official_timemixer"] for row in rows]); ratio = candidate / baseline
    result = {"validation_only": True, "test_metrics_produced": False, "seeds": list(SEEDS), "rows": rows, "channel_patch_vs_official_timemixer": {"candidate_mean_mse": float(candidate.mean()), "official_mean_mse": float(baseline.mean()), "percent": float((candidate.mean() / baseline.mean() - 1) * 100), "wins": int((candidate < baseline).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "submission_ready": False}
    (EXPANSION / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
