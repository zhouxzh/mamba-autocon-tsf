#!/usr/bin/env python3
"""Verify DCRNN manuscript numbers against the strict local readouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    "metr_la": {"candidate_mse": 0.273251, "baseline_mse": 0.288547, "delta_pct": -5.287, "wins": "3/3"},
    "pems_bay": {"candidate_mse": 0.296305, "baseline_mse": 0.408946, "delta_pct": -27.444, "wins": "3/3"},
    "resource": {"candidate_ms": 0.3252, "dcrnn_ms": 9.8622, "candidate_params": 29430, "dcrnn_params": 14268},
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, digits: int = 6) -> bool:
    return round(float(actual), digits) == round(expected, digits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {row["dataset"]: row for row in rows(args.summary)}
    resource = {row["model"]: row for row in rows(args.resources)}
    checks = {}
    for dataset in ("metr_la", "pems_bay"):
        row, expected = summary.get(dataset), EXPECTED[dataset]
        checks[f"{dataset}_present"] = row is not None
        if row is not None:
            checks[f"{dataset}_candidate_mse"] = close(row["candidate_mse_mean"], expected["candidate_mse"])
            checks[f"{dataset}_dcrnn_mse"] = close(row["dcrnn_mse_mean"], expected["baseline_mse"])
            checks[f"{dataset}_delta"] = close(row["candidate_vs_dcrnn_pct"], expected["delta_pct"], 3)
            checks[f"{dataset}_wins"] = f"{row['candidate_wins']}/{row['n']}" == expected["wins"]
    candidate, dcrnn = resource.get("MambaTSv3AutoCon"), resource.get("DCRNN")
    checks["resource_models"] = candidate is not None and dcrnn is not None
    if candidate is not None and dcrnn is not None:
        checks["resource_candidate_ms"] = close(candidate["latency_ms_median"], EXPECTED["resource"]["candidate_ms"], 4)
        checks["resource_dcrnn_ms"] = close(dcrnn["latency_ms_median"], EXPECTED["resource"]["dcrnn_ms"], 4)
        checks["resource_candidate_params"] = int(candidate["parameter_trainable"]) == EXPECTED["resource"]["candidate_params"]
        checks["resource_dcrnn_params"] = int(dcrnn["parameter_trainable"]) == EXPECTED["resource"]["dcrnn_params"]
    manuscript = args.manuscript.read_text(encoding="utf-8")
    # The manuscript intentionally reports paired deltas and links the strict
    # readout for raw MSE values, avoiding duplicated dense baseline tables.
    for token in ("5.287%", "27.444%", "3/3", "0.3252", "9.8622"):
        checks[f"manuscript_{token}"] = token in manuscript
    report = {"checks": checks, "passed": all(checks.values()), "scope": "DCRNN validation-only numbers and METR-LA fixed-shape resource numbers only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
