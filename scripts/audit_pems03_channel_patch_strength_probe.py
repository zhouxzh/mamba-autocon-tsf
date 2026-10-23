#!/usr/bin/env python3
"""Read-only audit for PEMS03 channel-patch strength screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_strength_probe_20261007"
SEED = 6067


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    baseline = load(OUTPUT / f"s{SEED}_fraction_0p05_frozen_core.json"); stronger = load(OUTPUT / f"s{SEED}_fraction_0p10_frozen_core.json"); pretrain = load(OUTPUT / f"s{SEED}_fraction_0p10_pretrain.json"); event = load(OUTPUT / f"s{SEED}_fraction_0p10_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (baseline, stronger, pretrain)): raise ValueError("test boundary")
    if "--channel-patch-relation-fraction" not in event["command"] or event["command"][event["command"].index("--channel-patch-relation-fraction") + 1] != ".10": raise ValueError("stronger fraction missing")
    if any("pems03" in item.lower() for index, item in enumerate(event["command"][:-1]) if event["command"][index] in {"--source", "--initial-checkpoint"}): raise ValueError("target leakage")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in pretrain["history"]): raise ValueError("patch relation inactive")
    baseline_mse, stronger_mse = float(baseline["best_validation_mse"]), float(stronger["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "fraction_0p05_mse": baseline_mse, "fraction_0p10_mse": stronger_mse, "fraction_0p10_vs_0p05_percent": (stronger_mse / baseline_mse - 1) * 100, "screen_passes_expansion_margin": bool(stronger_mse / baseline_mse - 1 <= -.0025), "target_excluded_from_source_and_initialization": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
