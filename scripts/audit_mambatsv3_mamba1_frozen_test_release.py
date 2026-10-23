#!/usr/bin/env python3
"""Audit frozen Mamba-1 held-out readers against official PatchTST readers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
SEEDS = (5930, 5931, 5932)
TARGETS = ("pems_bay", "metr_la", "weather")
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _guard(path: Path) -> None:
    payload = _load(path)
    if payload.get("status") != "completed" or payload.get("trigger") is not None or payload.get("returncode") != 0:
        raise ValueError(f"GPU guard failure: {path}")


def main() -> int:
    manifest = _load(RELEASE / "manifest_before_test.json")
    if manifest.get("status") != "frozen_before_test" or not manifest.get("validation_only_selection_complete"):
        raise ValueError("manifest was not frozen before test")
    if tuple(manifest.get("seeds", ())) != SEEDS:
        raise ValueError("seed manifest mismatch")
    rows = []
    for target in TARGETS:
        candidate_manifest = {Path(item["checkpoint"]).name: item for item in manifest["candidate"][target]}
        baseline_manifest = {Path(item["checkpoint"]).name: item for item in manifest["official_patchtst"][target]}
        for seed in SEEDS:
            active_path = RELEASE / target / f"s{seed}_active_test.json"
            patch_path = RELEASE / target / f"s{seed}_official_patchtst_test.json"
            active, patch = _load(active_path), _load(patch_path)
            if not (active.get("test_metrics_produced") and patch.get("test_metrics_produced")):
                raise ValueError(f"missing test metric: {target}/{seed}")
            if active.get("training_performed") or patch.get("training_performed"):
                raise ValueError(f"test reader trained: {target}/{seed}")
            if not (active.get("selection_done_before_test") and patch.get("selection_done_before_test")):
                raise ValueError(f"selection lifecycle mismatch: {target}/{seed}")
            active_checkpoint = Path(active["checkpoint"])
            patch_checkpoint = Path(patch["checkpoint"])
            if _sha256(active_checkpoint) != candidate_manifest[active_checkpoint.name]["sha256"]:
                raise ValueError(f"candidate checkpoint changed: {target}/{seed}")
            if _sha256(patch_checkpoint) != baseline_manifest[patch_checkpoint.name]["sha256"]:
                raise ValueError(f"baseline checkpoint changed: {target}/{seed}")
            if patch.get("official_commit") != COMMIT:
                raise ValueError(f"official commit mismatch: {target}/{seed}")
            _guard(RELEASE / target / f"s{seed}_active_event.json")
            _guard(RELEASE / target / f"s{seed}_official_patchtst_event.json")
            rows.append({"target": target, "seed": seed, "active_test_mse": float(active["test_mse"]),
                         "official_patchtst_test_mse": float(patch["test_mse"]),
                         "ratio": float(active["test_mse"]) / float(patch["test_mse"]),
                         "active_better": bool(float(active["test_mse"]) < float(patch["test_mse"])),
                         "test_windows": int(active["test_windows"])})
    summary = {}
    for target in TARGETS:
        subset = [row for row in rows if row["target"] == target]
        active = np.asarray([row["active_test_mse"] for row in subset])
        baseline = np.asarray([row["official_patchtst_test_mse"] for row in subset])
        ratios = active / baseline
        summary[target] = {
            "active_mean_test_mse": float(active.mean()),
            "official_patchtst_mean_test_mse": float(baseline.mean()),
            "active_vs_patchtst_percent": float((active.mean() / baseline.mean() - 1) * 100),
            "active_wins": int((active < baseline).sum()),
            "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
            "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
        }
    result = {
        "manifest_before_test": str(RELEASE / "manifest_before_test.json"),
        "historical_public_test_access_caveat": bool(manifest.get("historical_public_test_access_caveat")),
        "rows": rows, "summary": summary, "audit_valid": True,
        "scope": "frozen-checkpoint read-only chronological test release; official PatchTST external-code comparison",
        "submission_ready": False,
        "submission_blocker": "historical prior public-test access and missing broader external baselines/replication",
    }
    (RELEASE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
