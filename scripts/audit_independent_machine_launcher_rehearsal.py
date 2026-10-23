#!/usr/bin/env python3
"""Audit a same-machine execution rehearsal of the independent-machine launcher."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("artifacts/reproducibility/independent_machine_replication_20261003")


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def telemetry(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("guard_trigger") for row in rows):
        raise ValueError("invalid telemetry: {}".format(path))
    return {"max_temperature_c": max(float(row["temperature.gpu"]) for row in rows), "max_fan_percent": max(float(row["fan.speed"]) for row in rows)}


def main() -> int:
    active_train, null_train = load(ROOT / "active_hybrid.json"), load(ROOT / "null_hybrid.json")
    active_test, null_test = load(ROOT / "active_test.json"), load(ROOT / "null_test.json")
    if active_train["best_validation_mse"] > null_train["best_validation_mse"]:
        raise ValueError("test opened despite failed validation gate")
    for name, payload in (("active_test", active_test), ("null_test", null_test)):
        if payload.get("test_metrics_produced") is not True or payload.get("training_performed") is not False or payload.get("selection_done_before_test") is not True:
            raise ValueError("invalid test reader: {}".format(name))
    records = {}
    for name in ("active_pretrain", "null_pretrain", "active_finetune", "null_finetune", "active_test", "null_test"):
        event = load(ROOT / (name + "_event.json"))
        if event.get("status") != "completed" or event.get("trigger") is not None:
            raise ValueError("GPU guard failure: {}".format(name))
        records[name] = telemetry(ROOT / (name + "_telemetry.csv"))
    active, null = float(active_test["test_mse"]), float(null_test["test_mse"])
    result = {"launcher_rehearsal": True, "independent_machine": False, "active_test_mse": active, "null_test_mse": null, "active_vs_null_percent": (active / null - 1.0) * 100.0, "active_wins": int(active < null), "telemetry": records, "audit_valid": True, "scope": "same-machine rehearsal of portable launcher; excluded from primary effect estimate and independent-machine gate"}
    (ROOT / "rehearsal_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
