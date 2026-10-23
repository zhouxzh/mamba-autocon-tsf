#!/usr/bin/env python3
"""Audit the validation-only strict-positive-label AutoCon candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_electricity_traffic_25_boundary_20261003"
MATRICES = {"electricity": (5550, 5551, 5552), "traffic": (5560, 5561, 5562)}


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
    matrices = {}
    for target, seeds in MATRICES.items():
        rows = []
        for seed in seeds:
            active = _load(args.candidate_root / "{}_s{}_coverage_labels_hybrid.json".format(target, seed))
            pretrain = _load(args.candidate_root / "{}_s{}_coverage_labels_pretrain.json".format(target, seed))
            null = _load(BASELINE_ROOT / "{}_s{}_null_hybrid.json".format(target, seed))
            if active["test_metrics_produced"] or pretrain["test_metrics_produced"]:
                raise ValueError("candidate opened test metrics")
            if pretrain["config"].get("forecast_contrastive_positive_lags") != "coverage_only":
                raise ValueError("candidate is missing strict positive-label coverage")
            value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
            rows.append({"seed": seed, "active": value, "exact_null": reference,
                         "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                         "active_better": value < reference})
        matrices[target] = {
            "rows": rows,
            "mean_active": float(np.mean([row["active"] for row in rows])),
            "mean_null": float(np.mean([row["exact_null"] for row in rows])),
            "all_seed_wins": bool(all(row["active_better"] for row in rows)),
        }
        matrices[target]["mean_active_vs_null_percent"] = (
            matrices[target]["mean_active"] / matrices[target]["mean_null"] - 1.0
        ) * 100.0
    result = {
        "validation_only": True,
        "test_metrics_produced": False,
        "candidate": "coverage_only InfoNCE channel-positive labels",
        "matrices": matrices,
        "gate_passed": bool(all(item["all_seed_wins"] for item in matrices.values())),
        "next_step": "No test protocol is opened by this audit. A separately registered confirmation is required even if the validation gate passes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
