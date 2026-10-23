#!/usr/bin/env python3
"""Audit a validation-only source-compatible active/null candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MATRICES = {"electricity": (5650, 5651, 5652), "traffic": (5660, 5661, 5662)}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.candidate_root / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("candidate plan must be validation-only")
    results = {}
    for target, seeds in MATRICES.items():
        rows = []
        for seed in seeds:
            active = _load(args.candidate_root / "{}_s{}_active_hybrid.json".format(target, seed))
            null = _load(args.candidate_root / "{}_s{}_null_hybrid.json".format(target, seed))
            active_pre = _load(args.candidate_root / "{}_s{}_active_pretrain.json".format(target, seed))
            null_pre = _load(args.candidate_root / "{}_s{}_null_pretrain.json".format(target, seed))
            for payload in (active, null, active_pre, null_pre):
                if payload.get("test_metrics_produced"):
                    raise ValueError("candidate opened test metrics")
            for payload in (active_pre, null_pre):
                if len(payload["sources"]) != 1 or payload["config"].get("max_windows") != 192:
                    raise ValueError("single-source matched-budget contract failed")
            value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
            rows.append({"seed": seed, "active": value, "exact_null": reference,
                         "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                         "active_better": value < reference})
        mean_active, mean_null = np.mean([row["active"] for row in rows]), np.mean([row["exact_null"] for row in rows])
        results[target] = {"rows": rows, "mean_active": float(mean_active), "mean_null": float(mean_null),
                           "mean_active_vs_null_percent": float((mean_active / mean_null - 1.0) * 100.0),
                           "all_seed_wins": bool(all(row["active_better"] for row in rows))}
    audit = {"validation_only": True, "test_metrics_produced": False,
             "candidate": plan["candidate"], "matrices": results,
             "gate_passed": bool(all(value["all_seed_wins"] for value in results.values())),
             "next_step": "No test protocol is opened by this audit."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
