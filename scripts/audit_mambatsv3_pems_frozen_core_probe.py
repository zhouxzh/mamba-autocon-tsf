#!/usr/bin/env python3
"""Audit the fresh validation-only PEMS-BAY frozen-core probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.root / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("probe must be validation-only")
    if float(plan.get("few_shot_fraction", 0.0)) not in {0.1, 0.25}:
        raise ValueError("plan has an invalid few-shot fraction")
    if not args.seed or len(set(args.seed)) != len(args.seed):
        raise ValueError("--seed values must be unique")
    rows = []
    for seed in args.seed:
        active_pre, null_pre = _load(args.root / "s{}_active_pretrain.json".format(seed)), _load(args.root / "s{}_null_pretrain.json".format(seed))
        active, null = _load(args.root / "s{}_active_frozen_core.json".format(seed)), _load(args.root / "s{}_null_frozen_core.json".format(seed))
        if any(payload.get("test_metrics_produced") for payload in (active_pre, null_pre, active, null)):
            raise ValueError("probe opened test metrics")
        for payload in (active, null):
            if not payload["config"].get("freeze_pretrained_core"):
                raise ValueError("frozen-core setting is absent")
            if not all(row["mamba_core_frozen"] and row["mamba_forecast_head_trainable"] for row in payload["history"]):
                raise ValueError("target adaptation did not preserve the frozen-core contract")
            if float(payload["config"]["few_shot_fraction"]) != float(plan["few_shot_fraction"]):
                raise ValueError("checkpoint few-shot fraction does not match plan")
        value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
        rows.append({"seed": seed, "active": value, "exact_null": reference,
                     "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                     "active_better": bool(value < reference)})
    mean_active, mean_null = np.mean([row["active"] for row in rows]), np.mean([row["exact_null"] for row in rows])
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "mean_active": float(mean_active), "mean_null": float(mean_null),
              "mean_active_vs_null_percent": float((mean_active / mean_null - 1.0) * 100.0),
              "all_seed_wins": bool(all(row["active_better"] for row in rows)),
              "next_step": "No test protocol is opened by this validation matrix."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
