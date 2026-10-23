#!/usr/bin/env python3
"""Read-only audit for PEMS03 channel-patch stability weighting screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_stability_probe_20261007"
SEED = 6063


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    plan = load(OUTPUT / "plan.json")
    plain = load(OUTPUT / f"s{SEED}_plain_channel_patch_frozen_core.json")
    stability = load(OUTPUT / f"s{SEED}_stability_channel_patch_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_stability_channel_patch_pretrain.json")
    event = load(OUTPUT / f"s{SEED}_stability_channel_patch_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (plain, stability, pretrain)): raise ValueError("test boundary")
    command = event["command"]
    if "--channel-patch-stability-weighting" not in command: raise ValueError("stability flag missing")
    if any("pems03" in item.lower() for index, item in enumerate(command[:-1]) if command[index] in {"--source", "--initial-checkpoint"}): raise ValueError("target leakage")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in pretrain["history"]): raise ValueError("patch relation inactive")
    if any(source["stability_weight_mean"] is None for source in pretrain["sources"]): raise ValueError("stability weights absent")
    plain_mse, stability_mse = float(plain["best_validation_mse"]), float(stability["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "plain_channel_patch_mse": plain_mse, "stability_channel_patch_mse": stability_mse, "stability_vs_plain_percent": (stability_mse / plain_mse - 1) * 100, "screen_passes_expansion_margin": bool(stability_mse / plain_mse - 1 <= -.0025), "target_excluded_from_source_and_initialization": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
