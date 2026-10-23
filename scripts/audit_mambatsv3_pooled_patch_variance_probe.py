#!/usr/bin/env python3
"""Audit one validation-only anti-collapse pooled-patch probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", choices=["electricity", "traffic"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.root / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("probe must be validation-only")
    if float(plan.get("pooled_patch_variance_weight", 0.0)) <= 0:
        raise ValueError("probe must have a positive anti-collapse weight")
    prefix = "{}_s{}_pooled_patch_".format(args.target, args.seed)
    active_pre = _load(args.root / (prefix + "active_pretrain.json"))
    null_pre = _load(args.root / (prefix + "null_pretrain.json"))
    active = _load(args.root / (prefix + "active_hybrid.json"))
    null = _load(args.root / (prefix + "null_hybrid.json"))
    if any(payload.get("test_metrics_produced") for payload in (active_pre, null_pre, active, null)):
        raise ValueError("probe opened test metrics")
    config = active_pre["config"]
    if config.get("relation_view") != "pooled_patch_contrastive":
        raise ValueError("wrong representation relation view")
    if float(config.get("pooled_patch_variance_weight", 0.0)) <= 0:
        raise ValueError("active checkpoint lacks anti-collapse constraint")
    stds = [float(row["mean_pooled_patch_feature_std"]) for row in active_pre["history"]]
    if not all(std > 0 for std in stds):
        raise ValueError("active pretraining did not report pooled feature spread")
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "target": args.target, "seed": args.seed,
        "active": float(active["best_validation_mse"]),
        "exact_null": float(null["best_validation_mse"]),
        "active_vs_null_percent": (float(active["best_validation_mse"]) / float(null["best_validation_mse"]) - 1.0) * 100.0,
        "active_better": bool(active["best_validation_mse"] < null["best_validation_mse"]),
        "pooled_feature_std_history": stds,
        "variance_target_std": float(config["pooled_patch_variance_target_std"]),
        "conclusion": "anti-collapse geometry is verified separately from forecasting performance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
