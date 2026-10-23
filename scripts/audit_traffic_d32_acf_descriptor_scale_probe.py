#!/usr/bin/env python3
"""Read-only audit for target-tuned ACF descriptor gate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_scale_probe_20261007"
SEED = 6096
def load(path): return json.loads(path.read_text(encoding="utf-8"))
def main():
    forecast = load(OUTPUT / f"s{SEED}_forecast_calibration_frozen_core.json")
    frozen = load(OUTPUT / f"s{SEED}_descriptor_frozen_frozen_core.json")
    tuned = load(OUTPUT / f"s{SEED}_descriptor_target_scale_frozen_core.json")
    if any(item.get("test_metrics_produced") for item in (forecast, frozen, tuned)): raise ValueError("test boundary")
    history = tuned["history"]
    if not all(row["acf_descriptor_scale_trainable"] for row in history): raise ValueError("target descriptor scale was not trainable")
    if not all(row["mamba_core_frozen"] for row in history): raise ValueError("shared core was unfrozen")
    values = {"forecast": float(forecast["best_validation_mse"]), "descriptor_frozen": float(frozen["best_validation_mse"]), "descriptor_target_scale": float(tuned["best_validation_mse"])}
    result = {"validation_only": True, "test_metrics_produced": False, "mse": values, "target_scale_vs_forecast_percent": (values["descriptor_target_scale"] / values["forecast"] - 1) * 100, "target_scale_vs_frozen_descriptor_percent": (values["descriptor_target_scale"] / values["descriptor_frozen"] - 1) * 100, "target_scale_initial": history[0]["acf_descriptor_scale"], "target_scale_final": history[-1]["acf_descriptor_scale"], "screen_passes_expansion_margin": bool(values["descriptor_target_scale"] / values["forecast"] - 1 <= -.0025 and values["descriptor_target_scale"] / values["descriptor_frozen"] - 1 <= -.0025), "single_scan": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))
if __name__ == "__main__": main()
