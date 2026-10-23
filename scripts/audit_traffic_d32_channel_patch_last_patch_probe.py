#!/usr/bin/env python3
"""Read-only audit for Traffic final-patch Channel-Patch similarity."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_last_patch_probe_20261007"
SEED = 6093


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    trajectory = load(OUTPUT / f"s{SEED}_trajectory_frozen_core.json")
    last_patch = load(OUTPUT / f"s{SEED}_last_patch_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_last_patch_pretrain.json")
    event = load(OUTPUT / f"s{SEED}_last_patch_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (trajectory, last_patch, pretrain)):
        raise ValueError("test boundary")
    if pretrain["config"].get("channel_patch_similarity") != "last_patch":
        raise ValueError("last-patch mode missing")
    if "--channel-patch-similarity" not in event["command"]:
        raise ValueError("last-patch invocation missing")
    baseline = float(trajectory["best_validation_mse"])
    candidate = float(last_patch["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "trajectory_mse": baseline, "last_patch_mse": candidate,
        "last_patch_vs_trajectory_percent": (candidate / baseline - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(candidate / baseline - 1.0 <= -.0025),
        "single_scan": True, "inference_similarity_absent": True, "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
