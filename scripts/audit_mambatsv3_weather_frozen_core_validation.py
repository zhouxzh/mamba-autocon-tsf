#!/usr/bin/env python3
"""Audit validation-only fresh Weather frozen-core transfer evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "artifacts/exploratory/mambatsv3_weather_frozen_core_validation_20261003"
SEEDS = (5830, 5831, 5832)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(RESULT_ROOT / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("matrix must be validation-only")
    rows = []
    for seed in SEEDS:
        active_pre, null_pre = _load(RESULT_ROOT / "s{}_active_pretrain.json".format(seed)), _load(RESULT_ROOT / "s{}_null_pretrain.json".format(seed))
        active, null = _load(RESULT_ROOT / "s{}_active_frozen_core.json".format(seed)), _load(RESULT_ROOT / "s{}_null_frozen_core.json".format(seed))
        if any(payload.get("test_metrics_produced") for payload in (active_pre, null_pre, active, null)):
            raise ValueError("matrix opened test metrics")
        for payload in (active, null):
            if not payload["config"].get("freeze_pretrained_core"):
                raise ValueError("frozen-core setting missing")
            if not all(row["mamba_core_frozen"] and row["mamba_forecast_head_trainable"] for row in payload["history"]):
                raise ValueError("frozen-core adaptation contract failed")
        value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
        rows.append({"seed": seed, "active": value, "exact_null": reference,
                     "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                     "active_better": bool(value < reference)})
    mean_active, mean_null = np.mean([row["active"] for row in rows]), np.mean([row["exact_null"] for row in rows])
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "mean_active": float(mean_active), "mean_null": float(mean_null),
              "mean_active_vs_null_percent": float((mean_active / mean_null - 1.0) * 100.0),
              "all_seed_wins": bool(all(row["active_better"] for row in rows))}
    (RESULT_ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
