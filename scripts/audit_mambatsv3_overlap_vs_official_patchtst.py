#!/usr/bin/env python3
"""Audit same-seed overlap-patch MambaTSv3 versus official PatchTST validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_pems_overlap_patch_probe_20261003"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_overlap_pems25_validation_20261003"
SEEDS = (5860, 5861, 5862)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = _load(ACTIVE / "s{}_active_frozen_core.json".format(seed))
        baseline = _load(BASELINE / "pems_bay_h192_s{}_official_patchtst.json".format(seed))
        if active.get("test_metrics_produced") or baseline.get("test_metrics_produced"):
            raise ValueError("comparison opened test metrics")
        value, reference = float(active["best_validation_mse"]), float(baseline["best_validation_mse"])
        rows.append({"seed": seed, "overlap_active": value, "official_patchtst": reference,
                     "ratio": value / reference, "overlap_better": bool(value < reference)})
    values, baselines = np.asarray([row["overlap_active"] for row in rows]), np.asarray([row["official_patchtst"] for row in rows])
    ratios = values / baselines
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "mean_overlap_active": float(values.mean()), "mean_official_patchtst": float(baselines.mean()),
              "overlap_vs_official_percent": float((values.mean() / baselines.mean() - 1.0) * 100.0),
              "overlap_wins": int((values < baselines).sum()),
              "ratio_mean": float(ratios.mean()),
              "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
              "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
              "scope": "same-seed validation-only official-code comparison; no test evaluation"}
    (ACTIVE / "official_patchtst_validation_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
