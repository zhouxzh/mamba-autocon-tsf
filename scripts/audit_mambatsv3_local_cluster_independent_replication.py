#!/usr/bin/env python3
"""Audit an independently reviewed PEMS-BAY H192 local-cluster replication."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np


SEEDS = (6130, 6131, 6132)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def telemetry(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("guard_trigger") for row in rows):
        raise ValueError("invalid telemetry: {}".format(path))
    return {"max_temperature_c": max(float(row["temperature.gpu"]) for row in rows), "max_fan_percent": max(float(row["fan.speed"]) for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--verified-independent-machine-id", required=True)
    parser.add_argument("--host-evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    manifest, evidence = load(root / "manifest.json"), load(args.host_evidence)
    remote = evidence.get("remote", {})
    if manifest.get("machine_id") != args.verified_independent_machine_id or evidence.get("machine_id") != args.verified_independent_machine_id:
        raise ValueError("reviewer machine identifier mismatch")
    if not remote.get("cuda_available") or not remote.get("gpu"):
        raise ValueError("remote CUDA evidence missing")
    if remote.get("hostname") == platform.node():
        raise ValueError("remote host is not distinct from auditor host")
    if not manifest.get("target_excluded_from_source_and_initialization") or manifest.get("test_open_requested") is not True:
        raise ValueError("protocol boundary mismatch")
    if len(manifest.get("source_hashes", {})) != 14 or any(len(value) != 64 for value in manifest["source_hashes"].values()):
        raise ValueError("source hash manifest is incomplete")
    gate = load(root / "validation_gate.json")
    if gate.get("passes") is not True or gate.get("test_metrics_produced"):
        raise ValueError("test opened without a successful validation gate")
    rows, records = [], {}
    for seed in SEEDS:
        active_pre, null_pre = load(root / f"s{seed}_active_pretrain.json"), load(root / f"s{seed}_null_pretrain.json")
        active, null = load(root / f"s{seed}_active_hybrid.json"), load(root / f"s{seed}_null_hybrid.json")
        active_test, null_test = load(root / f"s{seed}_active_test.json"), load(root / f"s{seed}_null_test.json")
        if any(item.get("test_metrics_produced") for item in (active_pre, null_pre, active, null)):
            raise ValueError("training artifact opened test")
        if active_pre["config"]["relation_pair_policy"] != "contrastive_cluster" or active_pre["config"]["forecast_contrastive_pairing"] != "channel_coverage":
            raise ValueError("local-cluster contract mismatch")
        if min(float(item["global_relation_active_fraction"]) for item in active_pre["history"]) < .8:
            raise ValueError("active coverage below gate")
        if float(active["best_validation_mse"]) > float(null["best_validation_mse"]):
            raise ValueError("validation gate failure")
        for item in (active_test, null_test):
            if item.get("test_metrics_produced") is not True or item.get("training_performed") is not False or item.get("selection_done_before_test") is not True:
                raise ValueError("invalid held-out reader")
        for arm in ("active", "null"):
            for stage in ("pretrain", "finetune", "test"):
                event = load(root / f"s{seed}_{arm}_{stage}_event.json")
                if event.get("status") != "completed" or event.get("trigger") is not None:
                    raise ValueError("guard failure")
                records[f"s{seed}_{arm}_{stage}"] = telemetry(root / f"s{seed}_{arm}_{stage}_telemetry.csv")
        rows.append({"seed": seed, "active_validation_mse": float(active["best_validation_mse"]), "null_validation_mse": float(null["best_validation_mse"]), "active_test_mse": float(active_test["test_mse"]), "null_test_mse": float(null_test["test_mse"])})
    active = np.asarray([row["active_test_mse"] for row in rows])
    null = np.asarray([row["null_test_mse"] for row in rows])
    ratios = active / null
    result = {"independent_machine": True, "verified_machine_id": args.verified_independent_machine_id, "host_evidence_verified": {"hostname": remote["hostname"], "gpu": remote["gpu"], "torch": remote["torch"], "mamba_ssm": remote["mamba_ssm"], "different_from_audit_host": True}, "test_metrics_produced": True, "rows": rows, "active_vs_null": {"active_mean_test_mse": float(active.mean()), "null_mean_test_mse": float(null.mean()), "percent": float((active.mean() / null.mean() - 1) * 100), "wins": int((active < null).sum()), "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))), "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1)}, "telemetry": records, "audit_valid": True, "submission_ready": False}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
