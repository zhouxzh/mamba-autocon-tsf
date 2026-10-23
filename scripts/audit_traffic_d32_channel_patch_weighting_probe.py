#!/usr/bin/env python3
"""Read-only audit for Traffic ACF-strength Channel-Patch aggregation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_channel_patch_weighting_probe_20261007"
SEED = 6091


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    uniform = load(OUTPUT / f"s{SEED}_uniform_frozen_core.json")
    strength = load(OUTPUT / f"s{SEED}_acf_strength_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_acf_strength_pretrain.json")
    event = load(OUTPUT / f"s{SEED}_acf_strength_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (uniform, strength, pretrain)):
        raise ValueError("test boundary")
    if pretrain["config"].get("channel_patch_channel_weighting") != "acf_strength":
        raise ValueError("ACF-strength mode missing")
    if "--channel-patch-channel-weighting" not in event["command"]:
        raise ValueError("ACF-strength invocation missing")
    summaries = [source for source in pretrain["sources"] if source["channel_patch_channel_weighting"] == "acf_strength"]
    if not summaries or any(source["channel_patch_channel_weight_max"] is None for source in summaries):
        raise ValueError("channel-weight provenance missing")
    baseline = float(uniform["best_validation_mse"])
    candidate = float(strength["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "uniform_channel_patch_mse": baseline, "acf_strength_channel_patch_mse": candidate,
        "acf_strength_vs_uniform_percent": (candidate / baseline - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(candidate / baseline - 1.0 <= -.0025),
        "single_scan": True, "inference_weighting_absent": True, "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
