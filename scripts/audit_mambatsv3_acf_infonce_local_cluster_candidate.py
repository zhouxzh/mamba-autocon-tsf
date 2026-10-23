#!/usr/bin/env python3
"""Audit the validation-gated local-contrastive PEMS-BAY candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _event(path: Path):
    value = _load(path)
    if value.get("status") != "completed" or value.get("trigger") is not None:
        raise ValueError("GPU guard failed: {}".format(path))


def _telemetry(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("missing telemetry samples: {}".format(path))
    if any(row.get("guard_trigger") for row in rows):
        raise ValueError("telemetry contains a guard trigger: {}".format(path))
    return {
        "max_temperature_c": max(float(row["temperature.gpu"]) for row in rows),
        "max_fan_percent": max(float(row["fan.speed"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_s5503_20261003"),
    )
    args = parser.parse_args()
    root = args.root
    active_pretrain = _load(root / "active_pretrain.json")
    null_pretrain = _load(root / "null_pretrain.json")
    active_train = _load(root / "active_hybrid.json")
    null_train = _load(root / "null_hybrid.json")
    active_test = _load(root / "active_test.json")
    null_test = _load(root / "null_test.json")
    inference = _load(root / "active_inference.json")

    for name, payload in (("active", active_pretrain), ("null", null_pretrain)):
        config = payload["config"]
        if config["relation_view"] != "forecast_contrastive":
            raise ValueError("{} arm is not forecast InfoNCE".format(name))
        if config["forecast_contrastive_pairing"] != "channel_coverage":
            raise ValueError("{} arm lacks channel coverage pairing".format(name))
        if config["relation_pair_policy"] != "contrastive_cluster":
            raise ValueError("{} arm lacks local contrastive clusters".format(name))
        if payload.get("test_metrics_produced"):
            raise ValueError("{} pretraining opened the test partition".format(name))
        if [source["name"] for source in payload["sources"]] != ["PEMS07", "PEMS08", "PEMS04"]:
            raise ValueError("unexpected source pool in {} arm".format(name))
        if any(source["forecast_pair_lags"] != [288, 576] for source in payload["sources"]):
            raise ValueError("unexpected pair lags in {} arm".format(name))
    if active_pretrain["config"]["mamba_lambda"] != 0.1 or active_pretrain["config"]["relation_loss_fraction"] != 0.1:
        raise ValueError("active objective configuration mismatch")
    if null_pretrain["config"]["mamba_lambda"] != 0.0 or null_pretrain["config"]["relation_loss_fraction"] != 0.0:
        raise ValueError("exact-null objective configuration mismatch")
    active_coverage = [float(item["global_relation_active_fraction"]) for item in active_pretrain["history"]]
    if len(active_coverage) != 3 or min(active_coverage) < 0.8:
        raise ValueError("active InfoNCE coverage is below the registered threshold")

    for name, payload in (("active", active_train), ("null", null_train)):
        if payload.get("test_metrics_produced"):
            raise ValueError("{} validation training opened test".format(name))
        if payload["config"].get("prediction_head_transfer_mode") != "exact":
            raise ValueError("{} arm lacks exact prediction-head transfer".format(name))
        if payload["config"].get("mamba_correction_output") is not True:
            raise ValueError("{} arm lacks residual correction path".format(name))
    active_validation = float(active_train["best_validation_mse"])
    null_validation = float(null_train["best_validation_mse"])
    if active_validation > null_validation:
        raise ValueError("test was opened despite failing the registered validation gate")
    for name, payload in (("active", active_test), ("null", null_test)):
        if payload.get("test_metrics_produced") is not True or payload.get("training_performed") is not False:
            raise ValueError("invalid {} held-out reader".format(name))
        if payload.get("selection_done_before_test") is not True:
            raise ValueError("{} test read before validation selection".format(name))
    if (
        inference.get("mamba_scan_count") != 1
        or inference.get("relation_loss_included") is not False
        or inference.get("autocon_representation_returned") is not False
    ):
        raise ValueError("candidate inference path is not prediction-only one-scan")

    telemetry = {}
    for name in ("active_pretrain", "null_pretrain", "active_finetune", "null_finetune", "active_test", "null_test"):
        _event(root / (name + "_event.json"))
        telemetry[name] = _telemetry(root / (name + "_telemetry.csv"))
    _event(root / "inference_event.json")
    telemetry["inference"] = _telemetry(root / "inference_telemetry.csv")

    active_mse = float(active_test["test_mse"])
    null_mse = float(null_test["test_mse"])
    result = {
        "root": str(root),
        "candidate_only": True,
        "test_metrics_produced": True,
        "active_coverage_by_epoch": active_coverage,
        "validation": {
            "active_mse": active_validation,
            "null_mse": null_validation,
            "active_vs_null_percent": (active_validation / null_validation - 1.0) * 100.0,
        },
        "test": {
            "active_mse": active_mse,
            "null_mse": null_mse,
            "active_vs_null_percent": (active_mse / null_mse - 1.0) * 100.0,
            "active_wins": int(active_mse < null_mse),
        },
        "telemetry": telemetry,
        "inference": inference,
        "audit_valid": True,
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
