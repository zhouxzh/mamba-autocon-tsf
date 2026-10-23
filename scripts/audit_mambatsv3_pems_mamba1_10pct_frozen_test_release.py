#!/usr/bin/env python3
"""Audit the PEMS-BAY H192 10% frozen active/PatchTST test release."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_pems10_frozen_test_release_20261006"
SEEDS = (5960, 5961, 5962)
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = _load(RELEASE / "manifest_before_test.json")
    if manifest.get("status") != "frozen_before_test" or tuple(manifest.get("seeds", ())) != SEEDS:
        raise ValueError("test manifest lifecycle mismatch")
    candidates = {item["seed"]: item for item in manifest["candidate"]}
    baselines = {item["seed"]: item for item in manifest["official_patchtst"]}
    rows = []
    for seed in SEEDS:
        active = _load(RELEASE / f"s{seed}_active_test.json")
        patch = _load(RELEASE / f"s{seed}_official_patchtst_test.json")
        if not (active.get("test_metrics_produced") and patch.get("test_metrics_produced")):
            raise ValueError(f"missing test readout: {seed}")
        if active.get("training_performed") or patch.get("training_performed"):
            raise ValueError(f"test reader trained: {seed}")
        if not (active.get("selection_done_before_test") and patch.get("selection_done_before_test")):
            raise ValueError(f"selection lifecycle mismatch: {seed}")
        if _sha256(Path(active["checkpoint"])) != candidates[seed]["sha256"]:
            raise ValueError(f"candidate changed: {seed}")
        if _sha256(Path(patch["checkpoint"])) != baselines[seed]["sha256"] or patch.get("official_commit") != COMMIT:
            raise ValueError(f"PatchTST changed: {seed}")
        for event_name in (f"s{seed}_active_event.json", f"s{seed}_official_patchtst_event.json"):
            event = _load(RELEASE / event_name)
            if event.get("status") != "completed" or event.get("trigger") is not None or event.get("returncode") != 0:
                raise ValueError(f"guard failure: {event_name}")
        candidate, baseline = float(active["test_mse"]), float(patch["test_mse"])
        rows.append({"seed": seed, "active_test_mse": candidate, "official_patchtst_test_mse": baseline,
                     "ratio": candidate / baseline, "active_better": candidate < baseline})
    active = np.asarray([row["active_test_mse"] for row in rows])
    patch = np.asarray([row["official_patchtst_test_mse"] for row in rows])
    ratios = active / patch
    result = {
        "audit_valid": True, "rows": rows,
        "active_mean_test_mse": float(active.mean()), "official_patchtst_mean_test_mse": float(patch.mean()),
        "active_vs_patchtst_percent": float((active.mean() / patch.mean() - 1) * 100),
        "active_wins": int((active < patch).sum()),
        "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
        "historical_public_test_access_caveat": bool(manifest.get("historical_public_test_access_caveat")),
        "scope": "frozen PEMS-BAY 10% active/PatchTST read-only chronological test release",
        "submission_ready": False,
    }
    (RELEASE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
