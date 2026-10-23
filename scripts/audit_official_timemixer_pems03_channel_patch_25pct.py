#!/usr/bin/env python3
"""Read-only official TimeMixer PEMS03/25% comparison audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007"
OFFICIAL = ROOT / "artifacts/exploratory/official_timemixer_pems03_channel_patch_25pct_20261007"
COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
SEEDS = (6060, 6061, 6062)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    rows = []
    for seed in SEEDS:
        candidate = load(CANDIDATE / f"s{seed}_channel_patch_frozen_core.json"); official = load(OFFICIAL / f"pems03_h192_s{seed}_official_timemixer.json"); event = load(OFFICIAL / f"pems03_h192_s{seed}_official_timemixer_event.json")
        if candidate.get("test_metrics_produced") or official.get("test_metrics_produced"): raise ValueError("test boundary")
        if official.get("official_commit") != COMMIT or event.get("status") != "completed" or event.get("trigger") is not None: raise ValueError("official contract failure")
        if float(candidate["config"]["few_shot_fraction"]) != .25 or float(official["config"]["few_shot_fraction"]) != .25: raise ValueError("label mismatch")
        rows.append({"seed": seed, "channel_patch": float(candidate["best_validation_mse"]), "official_timemixer": float(official["best_validation_mse"])})
    candidate = np.asarray([row["channel_patch"] for row in rows]); official = np.asarray([row["official_timemixer"] for row in rows]); ratio = candidate / official
    result = {"validation_only": True, "test_metrics_produced": False, "target": "PEMS03", "horizon": 192, "few_shot_fraction": .25, "external_repository": "kwuking/TimeMixer", "official_commit": COMMIT, "rows": rows, "channel_patch_vs_official_timemixer": {"candidate_mean_mse": float(candidate.mean()), "official_mean_mse": float(official.mean()), "percent": float((candidate.mean() / official.mean() - 1) * 100), "wins": int((candidate < official).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "submission_ready": False}
    (OFFICIAL / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
