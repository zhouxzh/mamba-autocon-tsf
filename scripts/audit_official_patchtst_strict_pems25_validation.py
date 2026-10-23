#!/usr/bin/env python3
"""Compare locked strict PEMS-BAY 25% active checkpoints with official PatchTST validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_strict_pems25_validation_20261003"
SEEDS = (5810, 5811, 5812)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(BASELINE / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("baseline plan must be validation-only")
    rows = []
    for seed in SEEDS:
        active = _load(ACTIVE / "s{}_active_frozen_core.json".format(seed))
        baseline = _load(BASELINE / "pems_bay_h192_s{}_official_patchtst.json".format(seed))
        if active.get("test_metrics_produced") or baseline.get("test_metrics_produced"):
            raise ValueError("validation comparison opened test metrics")
        value, reference = float(active["best_validation_mse"]), float(baseline["best_validation_mse"])
        rows.append({"seed": seed, "active": value, "official_patchtst": reference,
                     "active_vs_official_percent": (value / reference - 1.0) * 100.0,
                     "active_better": bool(value < reference)})
    active_values, baseline_values = np.asarray([row["active"] for row in rows]), np.asarray([row["official_patchtst"] for row in rows])
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "mean_active": float(active_values.mean()), "mean_official_patchtst": float(baseline_values.mean()),
              "active_vs_official_percent": float((active_values.mean() / baseline_values.mean() - 1.0) * 100.0),
              "active_wins": int(sum(row["active_better"] for row in rows)),
              "scope": "same-seed official-code validation comparator; a separately registered read-only test evaluation is required"}
    (BASELINE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
