#!/usr/bin/env python3
"""Audit frozen Mamba-1 release against post-release iTransformer extension."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
EXTENSION = RELEASE / "official_itransformer_extension"
SEEDS = (5930, 5931, 5932)
TARGETS = ("pems_bay", "metr_la", "weather")
COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    manifest = _load(EXTENSION / "manifest_before_test.json")
    if manifest.get("status") != "frozen_before_test" or not manifest.get("selection_after_candidate_release"):
        raise ValueError("iTransformer extension manifest lifecycle mismatch")
    rows = []
    for target in TARGETS:
        expected = {Path(item["checkpoint"]).name: item for item in manifest["baselines"][target]}
        for seed in SEEDS:
            active = _load(RELEASE / target / f"s{seed}_active_test.json")
            itransformer = _load(EXTENSION / target / f"s{seed}_official_itransformer_test.json")
            if not (active.get("test_metrics_produced") and itransformer.get("test_metrics_produced")):
                raise ValueError(f"missing test readout: {target}/{seed}")
            if active.get("training_performed") or itransformer.get("training_performed"):
                raise ValueError(f"readout trained a model: {target}/{seed}")
            if not (active.get("selection_done_before_test") and itransformer.get("selection_done_before_test")):
                raise ValueError(f"selection lifecycle mismatch: {target}/{seed}")
            checkpoint = Path(itransformer["checkpoint"])
            if _sha256(checkpoint) != expected[checkpoint.name]["sha256"]:
                raise ValueError(f"baseline checkpoint changed: {target}/{seed}")
            if itransformer.get("official_commit") != COMMIT:
                raise ValueError(f"official commit mismatch: {target}/{seed}")
            for key, expected_hash in expected[checkpoint.name]["source_hashes"].items():
                if itransformer.get(key) != expected_hash:
                    raise ValueError(f"source hash mismatch {key}: {target}/{seed}")
            event = _load(EXTENSION / target / f"s{seed}_official_itransformer_event.json")
            if event.get("status") != "completed" or event.get("trigger") is not None or event.get("returncode") != 0:
                raise ValueError(f"GPU guard failed: {target}/{seed}")
            candidate, baseline = float(active["test_mse"]), float(itransformer["test_mse"])
            rows.append({"target": target, "seed": seed, "active_test_mse": candidate,
                         "official_itransformer_test_mse": baseline, "ratio": candidate / baseline,
                         "active_better": candidate < baseline})
    summary = {}
    for target in TARGETS:
        subset = [row for row in rows if row["target"] == target]
        active = np.asarray([row["active_test_mse"] for row in subset])
        baseline = np.asarray([row["official_itransformer_test_mse"] for row in subset])
        ratios = active / baseline
        summary[target] = {
            "active_mean_test_mse": float(active.mean()),
            "official_itransformer_mean_test_mse": float(baseline.mean()),
            "active_vs_itransformer_percent": float((active.mean() / baseline.mean() - 1) * 100),
            "active_wins": int((active < baseline).sum()),
            "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
            "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
        }
    result = {"audit_valid": True, "rows": rows, "summary": summary,
              "scope": "post-release frozen iTransformer external-baseline extension", "submission_ready": False}
    (EXTENSION / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
