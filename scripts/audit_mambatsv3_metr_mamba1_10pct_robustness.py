#!/usr/bin/env python3
"""Audit paired METR-LA 10%-label validation stresses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006/robustness_validation"
SEEDS = (5970, 5971, 5972)
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = {condition: [] for condition in CONDITIONS}
    for seed in SEEDS:
        active, null = (_load(RESULTS / f"s{seed}_active.json"), _load(RESULTS / f"s{seed}_null.json"))
        if active.get("test_metrics_produced") or null.get("test_metrics_produced"):
            raise ValueError("stress evaluation opened test metrics")
        for condition in CONDITIONS:
            left, right = active["conditions"][condition], null["conditions"][condition]
            if any(left[key] != right[key] for key in ("windows", "mask_fraction", "noise_scale")):
                raise ValueError(f"paired condition mismatch: {condition}")
            rows[condition].append((float(left["mse"]), float(right["mse"])))
    summary = {}
    for condition, values in rows.items():
        active = np.asarray([item[0] for item in values])
        null = np.asarray([item[1] for item in values])
        ratios = active / null
        upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
        summary[condition] = {"active_mean_mse": float(active.mean()), "null_mean_mse": float(null.mean()),
                              "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
                              "active_wins": int((active < null).sum()), "ratio_upper_1se": upper,
                              "noninferior_at_one_se": bool(upper <= 1.0)}
    result = {"validation_only": True, "test_metrics_produced": False, "summary": summary,
              "submission_ready": False, "scope": "METR-LA 10% independent validation stress audit"}
    (RESULTS / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
