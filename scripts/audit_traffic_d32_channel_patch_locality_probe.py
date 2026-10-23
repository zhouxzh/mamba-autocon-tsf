#!/usr/bin/env python3
"""Read-only audit for source-only Traffic Channel-Patch SSL."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_locality_probe_20261007"
SEED = 6092


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    both = load(OUTPUT / f"s{SEED}_source_and_target_frozen_core.json")
    source_only = load(OUTPUT / f"s{SEED}_source_only_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_source_only_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (both, source_only, pretrain)):
        raise ValueError("test boundary")
    roles = {source["role"]: source for source in pretrain["sources"]}
    if roles["source"]["channel_patch_relation_fraction"] != .05:
        raise ValueError("source Channel-Patch missing")
    if roles["target_calibration"]["channel_patch_relation_fraction"] != 0.0:
        raise ValueError("target Channel-Patch is not disabled")
    baseline = float(both["best_validation_mse"])
    candidate = float(source_only["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "source_and_target_mse": baseline, "source_only_mse": candidate,
        "source_only_vs_source_and_target_percent": (candidate / baseline - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(candidate / baseline - 1.0 <= -.0025),
        "single_scan": True, "target_core_frozen": True, "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
