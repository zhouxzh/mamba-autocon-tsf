#!/usr/bin/env python3
"""Read-only audit for Weather Channel-Patch locality screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/weather_channel_patch_calibration_locality_probe_20261007"
SEED = 6072


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    records = {arm: load(OUTPUT / f"s{SEED}_{arm}_frozen_core.json") for arm in ("forecast_calibration", "source_and_target_channel_patch", "target_only_channel_patch")}
    target_pretrain = load(OUTPUT / f"s{SEED}_target_only_channel_patch_pretrain.json")
    source_pretrain = load(OUTPUT / f"s{SEED}_source_and_target_channel_patch_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (*records.values(), target_pretrain, source_pretrain)): raise ValueError("test boundary")
    target_roles = {item["name"]: item for item in target_pretrain["sources"]}; source_roles = {item["name"]: item for item in source_pretrain["sources"]}
    if target_roles["Weather"]["channel_patch_relation_fraction"] != .05 or any(target_roles[name]["channel_patch_relation_fraction"] != 0.0 for name in ("METR-LA", "PEMS-BAY", "Electricity")): raise ValueError("target-only locality contract")
    if any(source_roles[name]["channel_patch_relation_fraction"] != .05 for name in source_roles): raise ValueError("source-and-target locality contract")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in target_pretrain["history"]): raise ValueError("target-only patch relation inactive")
    values = {arm: float(item["best_validation_mse"]) for arm, item in records.items()}
    result = {"validation_only": True, "test_metrics_produced": False, **values, "target_only_vs_forecast_percent": (values["target_only_channel_patch"] / values["forecast_calibration"] - 1) * 100, "target_only_vs_source_and_target_percent": (values["target_only_channel_patch"] / values["source_and_target_channel_patch"] - 1) * 100, "screen_passes_expansion_margin": bool(values["target_only_channel_patch"] / values["forecast_calibration"] - 1 <= -.0025 and values["target_only_channel_patch"] / values["source_and_target_channel_patch"] - 1 <= -.0025), "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
