#!/usr/bin/env python3
"""Audit frozen Mamba-1 replication stresses across PEMS-BAY, METR-LA, Weather."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006/robustness_validation"
SEEDS = (5930, 5931, 5932)
TARGETS = ("pems_bay", "metr_la", "weather")
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10",
              "missing_channels_30", "noise_0p05", "noise_0p10")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    summary = {}
    for target in TARGETS:
        cells = {condition: [] for condition in CONDITIONS}
        for seed in SEEDS:
            active = _load(RESULTS / target / f"s{seed}_active.json")
            null = _load(RESULTS / target / f"s{seed}_null.json")
            if active.get("test_metrics_produced") or null.get("test_metrics_produced"):
                raise ValueError(f"{target} opened test metrics")
            for condition in CONDITIONS:
                left, right = active["conditions"][condition], null["conditions"][condition]
                if any(left[key] != right[key] for key in ("windows", "mask_fraction", "noise_scale")):
                    raise ValueError(f"{target} paired stress mismatch")
                cells[condition].append((float(left["mse"]), float(right["mse"])))
        summary[target] = {}
        for condition, values in cells.items():
            active = np.asarray([item[0] for item in values])
            null = np.asarray([item[1] for item in values])
            ratios = active / null
            upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
            summary[target][condition] = {
                "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
                "active_wins": int((active < null).sum()), "ratio_upper_1se": upper,
                "noninferior_at_one_se": bool(upper <= 1.0),
            }
    result = {"validation_only": True, "test_metrics_produced": False,
              "mamba_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8},
              "summary": summary, "submission_ready": False,
              "scope": "independent frozen Mamba-1 replication stress audit"}
    (RESULTS / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
