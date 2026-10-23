#!/usr/bin/env python3
"""Compare staged strict transfer with same-seed official PatchTST validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_pems_staged_adaptation_probe_20261003"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_strict_pems25_validation_20261003"
SEEDS = (5810, 5811, 5812)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = _load(ACTIVE / "s{}_active_staged.json".format(seed))
        patchtst = _load(BASELINE / "pems_bay_h192_s{}_official_patchtst.json".format(seed))
        if active.get("test_metrics_produced") or patchtst.get("test_metrics_produced"):
            raise ValueError("validation comparison opened test metrics")
        value, reference = float(active["best_validation_mse"]), float(patchtst["best_validation_mse"])
        rows.append({"seed": seed, "staged_active": value, "official_patchtst": reference,
                     "ratio": value / reference, "staged_better": bool(value < reference)})
    active_values = np.asarray([row["staged_active"] for row in rows])
    baseline_values = np.asarray([row["official_patchtst"] for row in rows])
    ratios = active_values / baseline_values
    result = {
        "validation_only": True, "test_metrics_produced": False, "rows": rows,
        "mean_staged_active": float(active_values.mean()), "mean_official_patchtst": float(baseline_values.mean()),
        "staged_vs_official_percent": float((active_values.mean() / baseline_values.mean() - 1.0) * 100.0),
        "staged_wins": int((active_values < baseline_values).sum()),
        "staged_official_ratio_mean": float(ratios.mean()),
        "staged_official_ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        "scope": "same-seed validation-only official-code comparison; not a test or submission claim",
    }
    (ACTIVE / "official_patchtst_validation_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
