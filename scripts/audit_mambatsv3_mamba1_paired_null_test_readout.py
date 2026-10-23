#!/usr/bin/env python3
"""Audit held-out active/null contrast after the frozen primary release."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
NULL_ROOT = RELEASE / "paired_null_readout"
SEEDS = (5930, 5931, 5932)
TARGETS = ("pems_bay", "metr_la", "weather")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    manifest = _load(NULL_ROOT / "manifest_before_null_test.json")
    if manifest.get("status") != "paired_controls_frozen_before_readout" or manifest.get("selection_after_primary_test"):
        raise ValueError("paired null manifest lifecycle mismatch")
    rows = []
    for target in TARGETS:
        expected = {Path(item["checkpoint"]).name: item for item in manifest["null_checkpoints"][target]}
        for seed in SEEDS:
            active = _load(RELEASE / target / f"s{seed}_active_test.json")
            null = _load(NULL_ROOT / target / f"s{seed}_null_test.json")
            if not (active.get("test_metrics_produced") and null.get("test_metrics_produced")):
                raise ValueError(f"missing test result: {target}/{seed}")
            if active.get("training_performed") or null.get("training_performed"):
                raise ValueError(f"reader trained: {target}/{seed}")
            checkpoint = Path(null["checkpoint"])
            if _sha256(checkpoint) != expected[checkpoint.name]["sha256"]:
                raise ValueError(f"null checkpoint changed: {target}/{seed}")
            event = _load(NULL_ROOT / target / f"s{seed}_null_event.json")
            if event.get("status") != "completed" or event.get("trigger") is not None:
                raise ValueError(f"null GPU guard failed: {target}/{seed}")
            candidate, control = float(active["test_mse"]), float(null["test_mse"])
            rows.append({"target": target, "seed": seed, "active_test_mse": candidate,
                         "null_test_mse": control, "ratio": candidate / control,
                         "active_better": candidate < control})
    summary = {}
    for target in TARGETS:
        subset = [row for row in rows if row["target"] == target]
        active = np.asarray([row["active_test_mse"] for row in subset])
        null = np.asarray([row["null_test_mse"] for row in subset])
        ratios = active / null
        upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
        summary[target] = {"active_mean_test_mse": float(active.mean()), "null_mean_test_mse": float(null.mean()),
                           "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
                           "active_wins": int((active < null).sum()), "ratio_upper_1se": upper,
                           "noninferior_at_one_se": bool(upper <= 1.0)}
    result = {"rows": rows, "summary": summary, "audit_valid": True,
              "scope": "post-primary-release matched null controls; no architecture selection after test",
              "submission_ready": False}
    (NULL_ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
