#!/usr/bin/env python3
"""Audit the registered read-only strict-transfer post-validation test read."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_frozen_core_postvalidation_test import MATRICES, OUTPUT


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(OUTPUT / "plan.json")
    if plan.get("training_performed") or plan.get("selection_performed") or plan.get("checkpoint_mutation"):
        raise ValueError("test plan must be read-only")
    matrices = {}
    for name, spec in MATRICES.items():
        rows = []
        for seed in spec["seeds"]:
            active = _load(OUTPUT / "{}_s{}_active.json".format(name, seed))
            null = _load(OUTPUT / "{}_s{}_null.json".format(name, seed))
            if not active.get("test_metrics_produced") or not null.get("test_metrics_produced"):
                raise ValueError("missing test metric")
            if active.get("training_performed") or null.get("training_performed"):
                raise ValueError("test reader trained a checkpoint")
            value, reference = float(active["test_mse"]), float(null["test_mse"])
            rows.append({"seed": seed, "active_test_mse": value, "exact_null_test_mse": reference,
                         "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                         "active_better": bool(value < reference)})
        active_values = np.asarray([row["active_test_mse"] for row in rows])
        null_values = np.asarray([row["exact_null_test_mse"] for row in rows])
        matrices[name] = {
            "rows": rows, "mean_active_test_mse": float(active_values.mean()),
            "mean_null_test_mse": float(null_values.mean()),
            "active_vs_null_percent": float((active_values.mean() / null_values.mean() - 1.0) * 100.0),
            "active_wins": int(sum(row["active_better"] for row in rows)),
        }
    result = {"read_only": True, "test_metrics_produced": True,
              "scope": plan["scope"], "matrices": matrices}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
