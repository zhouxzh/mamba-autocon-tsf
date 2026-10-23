#!/usr/bin/env python3
"""Verify the failed PEMS03 external gate did not open the test partition."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems03_external_confirmation_20261003")
SEEDS = (5570, 5571, 5572)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = load(ROOT / "s{}_active_hybrid.json".format(seed))
        null = load(ROOT / "s{}_null_hybrid.json".format(seed))
        for name, payload in (("active", active), ("null", null)):
            if payload.get("test_metrics_produced"):
                raise ValueError("{} validation checkpoint opened test".format(name))
            event = load(ROOT / "s{}_{}_finetune_event.json".format(seed, name))
            if event.get("status") != "completed" or event.get("trigger") is not None:
                raise ValueError("{} guard failure".format(name))
        rows.append({"seed": seed, "active_validation": active["best_validation_mse"], "null_validation": null["best_validation_mse"]})
    passed = all(row["active_validation"] <= row["null_validation"] for row in rows)
    if passed:
        raise ValueError("PEMS03 all-seed gate unexpectedly passed")
    tests = list(ROOT.glob("*_test.json"))
    if tests:
        raise ValueError("failed gate must not have test artifacts")
    result = {"rows": rows, "all_seed_gate_passed": False, "test_metrics_produced": False, "audit_valid": True}
    (ROOT / "gate_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
