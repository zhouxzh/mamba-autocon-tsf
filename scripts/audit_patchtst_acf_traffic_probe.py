#!/usr/bin/env python3
"""Read-only audit for PatchTST ACF-pretraining Traffic gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/patchtst_acf_traffic_probe_20261007"
SEED = 6097


def load(path):
    return json.loads(path.read_text())


def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_head.json")
    candidate = load(OUTPUT / f"s{SEED}_channel_patch_head.json")
    pretrain = load(OUTPUT / f"s{SEED}_channel_patch_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (forecast, candidate, pretrain)):
        raise ValueError("test boundary")
    config = pretrain["config"]
    if config["channel_patch_relation_fraction"] != .05 or config["calibration_channel_patch_relation_fraction"] != .05:
        raise ValueError("Channel-Patch contract")
    values = {"forecast": float(forecast["best_validation_mse"]), "channel_patch": float(candidate["best_validation_mse"])}
    result = {"validation_only": True, "test_metrics_produced": False, "mse": values, "channel_patch_vs_forecast_percent": (values["channel_patch"] / values["forecast"] - 1) * 100, "screen_passes_expansion_margin": bool(values["channel_patch"] / values["forecast"] - 1 <= -.0025), "encoder_frozen": all(row["encoder_frozen"] for row in candidate["history"]), "transformer_encoder_passes": 1, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
