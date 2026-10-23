#!/usr/bin/env python3
"""Read-only audit for PEMS03 channel-patch ACF-confidence screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_confidence_probe_20261007"
SEED = 6064


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    plain = load(OUTPUT / f"s{SEED}_plain_channel_patch_frozen_core.json"); confidence = load(OUTPUT / f"s{SEED}_confidence_channel_patch_frozen_core.json"); pretrain = load(OUTPUT / f"s{SEED}_confidence_channel_patch_pretrain.json"); event = load(OUTPUT / f"s{SEED}_confidence_channel_patch_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (plain, confidence, pretrain)): raise ValueError("test boundary")
    command = event["command"]
    if "--channel-patch-acf-confidence-weighting" not in command: raise ValueError("confidence flag missing")
    if any("pems03" in item.lower() for index, item in enumerate(command[:-1]) if command[index] in {"--source", "--initial-checkpoint"}): raise ValueError("target leakage")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in pretrain["history"]): raise ValueError("patch relation inactive")
    plain_mse, confidence_mse = float(plain["best_validation_mse"]), float(confidence["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "plain_channel_patch_mse": plain_mse, "confidence_channel_patch_mse": confidence_mse, "confidence_vs_plain_percent": (confidence_mse / plain_mse - 1) * 100, "screen_passes_expansion_margin": bool(confidence_mse / plain_mse - 1 <= -.0025), "target_excluded_from_source_and_initialization": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
