#!/usr/bin/env python3
"""Audit the frozen three-seed local-cluster PEMS-BAY confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def event(path: Path):
    value = load(path)
    if value.get("status") != "completed" or value.get("trigger") is not None:
        raise ValueError("guard failure: {}".format(path))


def telemetry(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("guard_trigger") for row in rows):
        raise ValueError("invalid telemetry: {}".format(path))
    return {
        "max_temperature_c": max(float(row["temperature.gpu"]) for row in rows),
        "max_fan_percent": max(float(row["fan.speed"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_confirmation_20261003"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--expected-source", action="append")
    parser.add_argument("--development-seed-excluded", type=int)
    args = parser.parse_args()
    root = args.root
    seeds = tuple(args.seed or (5510, 5511, 5512))
    expected_sources = args.expected_source or ["PEMS07", "PEMS08", "PEMS04"]
    rows = []
    telemetry_rows = {}
    inference = load(root / "inference.json")
    for seed in seeds:
        active_pre = load(root / "s{}_active_pretrain.json".format(seed))
        null_pre = load(root / "s{}_null_pretrain.json".format(seed))
        active_train = load(root / "s{}_active_hybrid.json".format(seed))
        null_train = load(root / "s{}_null_hybrid.json".format(seed))
        active_test = load(root / "s{}_active_test.json".format(seed))
        null_test = load(root / "s{}_null_test.json".format(seed))
        for name, payload in (("active_pretrain", active_pre), ("null_pretrain", null_pre), ("active_hybrid", active_train), ("null_hybrid", null_train)):
            if payload.get("test_metrics_produced"):
                raise ValueError("validation artifact opened test: {} {}".format(seed, name))
        if active_pre["config"]["relation_pair_policy"] != "contrastive_cluster":
            raise ValueError("missing local cluster geometry")
        if active_pre["config"]["forecast_contrastive_pairing"] != "channel_coverage":
            raise ValueError("missing channel coverage geometry")
        if [source["name"] for source in active_pre["sources"]] != expected_sources:
            raise ValueError("unexpected active source pool")
        if [source["name"] for source in null_pre["sources"]] != expected_sources:
            raise ValueError("unexpected null source pool")
        ignored = {"output", "mamba_lambda", "relation_loss_fraction"}
        active_config = {
            key: value for key, value in active_pre["config"].items()
            if key not in ignored
        }
        null_config = {
            key: value for key, value in null_pre["config"].items()
            if key not in ignored
        }
        if active_config != null_config:
            raise ValueError("active/null pretraining changed more than the relation objective")
        if min(float(item["global_relation_active_fraction"]) for item in active_pre["history"]) < .8:
            raise ValueError("active source coverage below gate")
        if float(active_train["best_validation_mse"]) > float(null_train["best_validation_mse"]):
            raise ValueError("validation gate failed before test read")
        for name, payload in (("active_test", active_test), ("null_test", null_test)):
            if payload.get("test_metrics_produced") is not True or payload.get("training_performed") is not False or payload.get("selection_done_before_test") is not True:
                raise ValueError("invalid held-out reader: {} {}".format(seed, name))
        for name in ("active_pretrain", "null_pretrain", "active_finetune", "null_finetune", "active_test", "null_test"):
            event(root / "s{}_{}_event.json".format(seed, name))
            telemetry_rows["s{}_{}".format(seed, name)] = telemetry(root / "s{}_{}_telemetry.csv".format(seed, name))
        active = float(active_test["test_mse"])
        null = float(null_test["test_mse"])
        rows.append({
            "seed": seed,
            "active_validation": float(active_train["best_validation_mse"]),
            "null_validation": float(null_train["best_validation_mse"]),
            "active_test": active,
            "null_test": null,
            "test_relative_percent": (active / null - 1.0) * 100.0,
            "active_win": int(active < null),
        })
    if (
        inference.get("mamba_scan_count") != 1
        or inference.get("relation_loss_included") is not False
        or inference.get("autocon_representation_returned") is not False
    ):
        raise ValueError("inference is not the required prediction-only single-scan path")
    event(root / "inference_event.json")
    telemetry_rows["inference"] = telemetry(root / "inference_telemetry.csv")
    active = np.asarray([row["active_test"] for row in rows])
    null = np.asarray([row["null_test"] for row in rows])
    delta = active - null
    mean_delta = float(delta.mean())
    ci95 = float(1.96 * delta.std(ddof=1) / math.sqrt(len(delta)))
    result = {
        "audit_valid": True,
        "confirmation_only": True,
        "development_seed_excluded": args.development_seed_excluded,
        "rows": rows,
        "summary": {
            "active_mean_test_mse": float(active.mean()),
            "null_mean_test_mse": float(null.mean()),
            "active_vs_null_percent": float((active.mean() / null.mean() - 1.0) * 100.0),
            "active_wins": int((active < null).sum()),
            "paired_delta_mean": mean_delta,
            "paired_delta_ci95": [mean_delta - ci95, mean_delta + ci95],
        },
        "telemetry": telemetry_rows,
        "inference": inference,
        "test_metrics_produced": True,
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
