#!/usr/bin/env python3
"""Audit validation-only overlapping-patch MambaTSv3 transfer probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_overlap_patch_probe_20261003"
SEED = 5860


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(OUTPUT / "plan.json")
    seeds = tuple(plan.get("seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("plan must include unique seeds")
    if not plan.get("validation_only"):
        raise ValueError("probe must stay validation-only")
    rows = []
    for seed in seeds:
        active_pre, null_pre = _load(OUTPUT / "s{}_active_pretrain.json".format(seed)), _load(OUTPUT / "s{}_null_pretrain.json".format(seed))
        active, null = _load(OUTPUT / "s{}_active_frozen_core.json".format(seed)), _load(OUTPUT / "s{}_null_frozen_core.json".format(seed))
        if any(item.get("test_metrics_produced") for item in (active_pre, null_pre, active, null)):
            raise ValueError("probe opened test metrics")
        for payload in (active_pre, null_pre, active, null):
            if int(payload["config"]["patch_len"]) != 24 or int(payload["config"]["patch_stride"]) != 8:
                raise ValueError("overlap patch geometry mismatch")
        for payload in (active, null):
            if not payload["config"].get("freeze_pretrained_core"):
                raise ValueError("target core must remain frozen")
        value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
        rows.append({"seed": seed, "active": value, "exact_null": reference,
                     "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                     "active_better": bool(value < reference)})
    active_mean, null_mean = np.mean([row["active"] for row in rows]), np.mean([row["exact_null"] for row in rows])
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "mean_active": float(active_mean), "mean_null": float(null_mean),
              "mean_active_vs_null_percent": float((active_mean / null_mean - 1.0) * 100.0),
              "all_seed_wins": bool(all(row["active_better"] for row in rows)),
              "next_step": "A separate same-seed external-baseline audit is required; no test is opened."}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
