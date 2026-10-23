#!/usr/bin/env python3
"""Audit validation-only robustness of the frozen-core PEMS-BAY matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEEDS = (5800, 5801, 5802)
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--prefix", default="s")
    args = parser.parse_args()
    plan = _load(args.root / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("robustness plan must be validation-only")
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seed values must be unique")
    result = {"validation_only": True, "test_metrics_produced": False, "conditions": {}}
    for condition in CONDITIONS:
        rows = []
        for seed in seeds:
            active = _load(args.root / "{}{}_active.json".format(args.prefix, seed))
            null = _load(args.root / "{}{}_null.json".format(args.prefix, seed))
            if active.get("test_metrics_produced") or null.get("test_metrics_produced"):
                raise ValueError("robustness evaluator opened test metrics")
            value = float(active["conditions"][condition]["mse"])
            reference = float(null["conditions"][condition]["mse"])
            rows.append({"seed": seed, "active": value, "exact_null": reference,
                         "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                         "active_better": bool(value < reference)})
        mean_active, mean_null = np.mean([row["active"] for row in rows]), np.mean([row["exact_null"] for row in rows])
        result["conditions"][condition] = {
            "rows": rows, "mean_active": float(mean_active), "mean_null": float(mean_null),
            "mean_active_vs_null_percent": float((mean_active / mean_null - 1.0) * 100.0),
            "all_seed_wins": bool(all(row["active_better"] for row in rows)),
        }
    result["all_conditions_all_seed_wins"] = bool(all(item["all_seed_wins"] for item in result["conditions"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
