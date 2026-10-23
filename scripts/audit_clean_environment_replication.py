#!/usr/bin/env python3
"""Audit the same-machine independent-interpreter PEMS-BAY replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_clean_environment_replication_20261003")
MANIFEST = ROOT / "manifest.json"
HASH_PATHS = {
    "scripts/pretrain_mambatsv3_core_autocon.py": Path("scripts/pretrain_mambatsv3_core_autocon.py"),
    "scripts/finetune_mambatsv3_dlinear_hybrid.py": Path("scripts/finetune_mambatsv3_dlinear_hybrid.py"),
    "models/MambaTS.py": Path("models/MambaTS.py"),
    "models/MambaDLinearHybrid.py": Path("models/MambaDLinearHybrid.py"),
    "layers/losses.py": Path("layers/losses.py"),
    "data_provider/samplers.py": Path("data_provider/samplers.py"),
    "dataset/PEMS-BAY/PEMS-BAY.csv": Path("dataset/PEMS-BAY/PEMS-BAY.csv"),
    "dataset/PEMS07/PEMS07.npz": Path("dataset/PEMS07/PEMS07.npz"),
    "dataset/PEMS08/PEMS08.npz": Path("dataset/PEMS08/PEMS08.npz"),
    "dataset/PEMS04/PEMS04.npz": Path("dataset/PEMS04/PEMS04.npz"),
}


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard(name: str):
    event = load(ROOT / (name + "_event.json"))
    if event.get("status") != "completed" or event.get("trigger") is not None:
        raise ValueError("GPU guard failure: {}".format(name))


def main() -> int:
    manifest = load(MANIFEST)
    if manifest["label"] != "same-machine independent-interpreter replication":
        raise ValueError("unexpected replication label")
    for key, path in HASH_PATHS.items():
        if sha256(path) != manifest["sha256"][key]:
            raise ValueError("source/data hash mismatch: {}".format(key))
    active_pre = load(ROOT / "active_pretrain.json")
    null_pre = load(ROOT / "null_pretrain.json")
    active_train = load(ROOT / "active_hybrid.json")
    null_train = load(ROOT / "null_hybrid.json")
    active_test = load(ROOT / "active_test.json")
    null_test = load(ROOT / "null_test.json")
    for name, payload in (("active_pretrain", active_pre), ("null_pretrain", null_pre), ("active_hybrid", active_train), ("null_hybrid", null_train)):
        if payload.get("test_metrics_produced"):
            raise ValueError("{} opened test before gate".format(name))
    config = active_pre["config"]
    if config["relation_pair_policy"] != "contrastive_cluster" or config["forecast_contrastive_pairing"] != "channel_coverage":
        raise ValueError("replication geometry mismatch")
    if min(item["global_relation_active_fraction"] for item in active_pre["history"]) < .8:
        raise ValueError("active coverage below registered threshold")
    if active_train["best_validation_mse"] > null_train["best_validation_mse"]:
        raise ValueError("test opened despite failed validation gate")
    for name, payload in (("active_test", active_test), ("null_test", null_test)):
        if payload.get("test_metrics_produced") is not True or payload.get("training_performed") is not False or payload.get("selection_done_before_test") is not True:
            raise ValueError("invalid test reader: {}".format(name))
    for name in ("active_pretrain", "null_pretrain", "active_finetune", "null_finetune", "active_test", "null_test"):
        guard(name)
    active, null = float(active_test["test_mse"]), float(null_test["test_mse"])
    result = {
        "same_machine_independent_interpreter": True,
        "independent_machine": False,
        "active_test_mse": active,
        "null_test_mse": null,
        "active_vs_null_percent": (active / null - 1.0) * 100.0,
        "active_wins": int(active < null),
        "test_metrics_produced": True,
        "audit_valid": True,
        "scope": "replication support only; excluded from primary three-seed effect estimate",
    }
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
