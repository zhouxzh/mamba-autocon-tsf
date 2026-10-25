#!/usr/bin/env python3
"""Validate and summarize a completed ACE2 synthetic simulation phase."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ARMS = ("A_signed", "N_null", "U_unsigned", "S_shuffle", "R_random")


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input / "results.jsonl"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    statuses = Counter(row["status"] for row in rows)
    keys = [row["task_key"] for row in rows]
    duplicate_keys = len(keys) - len(set(keys))
    arm_groups: dict[tuple[object, ...], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        group_key = tuple(row[field] for field in ("dgp", "migration", "noise", "replicate", "horizon", "label_fraction", "optimization_seed"))
        arm_groups[group_key][row["arm"]] = row
    pair_rows = []
    for group_key, arms in sorted(arm_groups.items()):
        active = arms.get("A_signed")
        if not active or active.get("status") != "COMPLETED":
            continue
        for control in ARMS[1:]:
            candidate = arms.get(control)
            if candidate and candidate.get("status") == "COMPLETED":
                pair_rows.append({
                    "dgp": group_key[0], "migration": group_key[1], "noise": group_key[2],
                    "replicate": group_key[3], "horizon": group_key[4], "label_fraction": group_key[5],
                    "optimization_seed": group_key[6], "control": control,
                    "active_mse": active["mse"], "control_mse": candidate["mse"],
                    "delta_mse": float(active["mse"]) - float(candidate["mse"]),
                })
    by_condition: dict[tuple[str, str, str, int, float, str], list[float]] = defaultdict(list)
    for row in pair_rows:
        key = (row["dgp"], row["migration"], row["noise"], row["horizon"], row["label_fraction"], row["control"])
        by_condition[key].append(row["delta_mse"])
    summary_rows = []
    for key, deltas in sorted(by_condition.items()):
        summary_rows.append({
            "dgp": key[0], "migration": key[1], "noise": key[2], "horizon": key[3], "label_fraction": key[4], "control": key[5],
            "paired_n": len(deltas), "mean_delta_mse": mean(deltas), "median_delta_mse": statistics.median(deltas),
            "stdev_delta_mse": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        })
    valid = len(rows) == args.expected_tasks and duplicate_keys == 0 and statuses == Counter({"COMPLETED": args.expected_tasks})
    audit = {
        "valid": valid,
        "expected_tasks": args.expected_tasks,
        "observed_tasks": len(rows),
        "duplicate_task_keys": duplicate_keys,
        "statuses": dict(statuses),
        "five_arm_complete_groups": sum(all(arm in arms and arms[arm].get("status") == "COMPLETED" for arm in ARMS) for arms in arm_groups.values()),
        "observed_groups": len(arm_groups),
        "paired_contrasts": len(pair_rows),
        "scope": "synthetic ACE2 phase only; contrasts are descriptive pending the plan's locked statistical analysis.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]) if pair_rows else ["control"])
        writer.writeheader()
        writer.writerows(pair_rows)
    with (args.output / "condition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary_rows[0]) if summary_rows else ["control"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(audit, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
