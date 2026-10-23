#!/usr/bin/env python3
"""Summarize guarded, validation-only local-cluster robustness extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    "in_distribution", "late_validation_shift", "missing_channels_10",
    "missing_channels_30", "noise_0p05", "noise_0p10",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--stem-prefix", default=None,
                        help="optional file-stem prefix; defaults to --target")
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args()
    rows = []
    prefix = args.target if args.stem_prefix is None else args.stem_prefix
    for seed in args.seed:
        stem = "{}s{}".format(prefix + "_" if prefix else "", seed)
        active = load(args.root / (stem + "_active_stress.json"))
        null = load(args.root / (stem + "_null_stress.json"))
        for name, payload in (("active", active), ("null", null)):
            if payload.get("test_metrics_produced"):
                raise ValueError("{} stress evaluator opened test".format(name))
            event = load(args.root / (stem + "_{}_stress_event.json".format(name)))
            if event.get("status") != "completed" or event.get("trigger") is not None:
                raise ValueError("{} stress guard failed".format(name))
        rows.append({"seed": seed, "active": active["conditions"], "null": null["conditions"]})
    summary = {}
    for condition in CONDITIONS:
        active = np.asarray([row["active"][condition]["mse"] for row in rows])
        null = np.asarray([row["null"][condition]["mse"] for row in rows])
        summary[condition] = {
            "active_mean_mse": float(active.mean()),
            "null_mean_mse": float(null.mean()),
            "active_vs_null_percent": float((active.mean() / null.mean() - 1.0) * 100.0),
            "active_wins": int((active < null).sum()),
        }
    result = {"target": args.target, "rows": rows, "summary": summary, "audit_valid": True, "test_metrics_produced": False}
    (args.root / (args.target + "_robustness_audit.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
