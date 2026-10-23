#!/usr/bin/env python3
"""Audit a clean-install replication that uses locked local binary artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_clean_install_replication_20261003")
MANIFEST = ROOT / "manifest.json"
WHEEL_HASHES = {
    "mamba_ssm_wheel": "e5efa9866d44ab54572d105e5a6b2c6cd0ec8aea623373c3d3c1310638079d41",
    "causal_conv1d_wheel": "435cfb57c204eec017e051fdf51b3d3f22edfcb624f394f18b3f1c46b67f4e28",
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
    if manifest.get("label") != "clean-install plus locked-local-binary-artifacts replication":
        raise ValueError("unexpected replication label")
    if manifest.get("independent_machine") is not False or manifest.get("from_source_binary_build") is not False:
        raise ValueError("replication scope incorrectly stated")
    artifacts = manifest["binary_artifacts"]
    for key, expected in WHEEL_HASHES.items():
        wheel = Path(artifacts[key])
        if sha256(wheel) != expected or artifacts[key.replace("_wheel", "_sha256")] != expected:
            raise ValueError("binary artifact hash mismatch: {}".format(key))
    python = Path(manifest["interpreter"])
    output = subprocess.check_output(
        [str(python), "-c", "import torch,mamba_ssm,causal_conv1d; print(torch.__version__, int(torch.cuda.is_available()), mamba_ssm.__version__)"],
        text=True,
    ).strip().split()
    if output != [manifest["torch"], "1", manifest["mamba_ssm"]]:
        raise ValueError("clean environment runtime mismatch")
    active_pre, null_pre = load(ROOT / "active_pretrain.json"), load(ROOT / "null_pretrain.json")
    active_train, null_train = load(ROOT / "active_hybrid.json"), load(ROOT / "null_hybrid.json")
    active_test, null_test = load(ROOT / "active_test.json"), load(ROOT / "null_test.json")
    for name, payload in (("active_pretrain", active_pre), ("null_pretrain", null_pre), ("active_hybrid", active_train), ("null_hybrid", null_train)):
        if payload.get("test_metrics_produced"):
            raise ValueError("{} opened test before gate".format(name))
    if min(item["global_relation_active_fraction"] for item in active_pre["history"]) < .8:
        raise ValueError("active coverage below threshold")
    if active_train["best_validation_mse"] > null_train["best_validation_mse"]:
        raise ValueError("test opened despite failed validation gate")
    for name, payload in (("active_test", active_test), ("null_test", null_test)):
        if payload.get("test_metrics_produced") is not True or payload.get("training_performed") is not False or payload.get("selection_done_before_test") is not True:
            raise ValueError("invalid test reader: {}".format(name))
    for name in ("active_pretrain", "null_pretrain", "active_finetune", "null_finetune", "active_test", "null_test"):
        guard(name)
    active, null = float(active_test["test_mse"]), float(null_test["test_mse"])
    result = {
        "clean_install": True,
        "same_machine": True,
        "independent_machine": False,
        "from_source_binary_build": False,
        "active_test_mse": active,
        "null_test_mse": null,
        "active_vs_null_percent": (active / null - 1.0) * 100.0,
        "active_wins": int(active < null),
        "test_metrics_produced": True,
        "audit_valid": True,
        "scope": "clean-install support replication with locked local binary artifacts; excluded from primary three-seed effect estimate",
    }
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
