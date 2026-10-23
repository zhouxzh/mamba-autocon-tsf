#!/usr/bin/env python3
"""Read-only same-seed official PatchTST audit for Electricity channel-patch SSL."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_electricity_channel_patch_replication_20261006"
SEEDS = (6033, 6034, 6035)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        candidate = load(CANDIDATE / f"s{seed}_channel_patch_frozen_core.json")
        patch = load(PATCH / f"electricity_h96_s{seed}_official_patchtst.json")
        if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"): raise ValueError("test boundary violated")
        rows.append({"seed": seed, "channel_patch": float(candidate["best_validation_mse"]), "official_patchtst": float(patch["best_validation_mse"])})
    candidate = np.asarray([row["channel_patch"] for row in rows]); patch = np.asarray([row["official_patchtst"] for row in rows]); ratios = candidate / patch
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "channel_patch_vs_official_patchtst": {"candidate_mean_mse": float(candidate.mean()), "official_patchtst_mean_mse": float(patch.mean()),
                                                       "percent": float((candidate.mean() / patch.mean() - 1.0) * 100.0), "wins": int((candidate < patch).sum()),
                                                       "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
                                                       "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0)},
              "submission_ready": False}
    (CANDIDATE / "external_patchtst_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
