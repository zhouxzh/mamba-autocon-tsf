#!/usr/bin/env python3
"""Audit the frozen PEMS-BAY H192/H336 ACF-InfoNCE held-out release."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003")
SOURCE_SELECTION = Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_selection_20261003/pems_bay_sources.json")
HORIZONS = {"h192": (192, (5410, 5411, 5412)), "h336": (336, (5413, 5414, 5415))}
ACTIVE_ROOTS = {
    "h192": Path("artifacts/exploratory/mambatsv3_acf_infonce_h192_screen_20261003"),
    "h336": Path("artifacts/exploratory/mambatsv3_acf_infonce_h336_screen_20261003"),
}
CONTROL_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_horizon_primary_validation_20261003")
BASELINE_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_horizon_strong_baselines_20261003")
STRESS_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_horizon_stress_validation_20261003")
EXPECTED_SOURCES = ["PEMS07", "PEMS08", "PEMS04"]
STRESSES = (
    "in_distribution", "late_validation_shift", "missing_channels_10",
    "missing_channels_30", "noise_0p05", "noise_0p10",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def event(path: Path):
    value = load(path)
    if value.get("status") != "completed" or value.get("trigger") is not None:
        raise ValueError("guard failure: {}".format(path))


def main() -> int:
    selection = load(SOURCE_SELECTION)
    if selection.get("main_validation_accessed") or selection.get("test_metrics_produced"):
        raise ValueError("source selection crossed its boundary")
    if [item["name"] for item in selection.get("selected", [])] != EXPECTED_SOURCES:
        raise ValueError("unexpected PEMS-BAY source selection")
    rows = []
    for label, (horizon, seeds) in HORIZONS.items():
        active_root = ACTIVE_ROOTS[label]
        for seed in seeds:
            stem = "pems_bay_{}_s{}".format(label, seed)
            active_train = load(active_root / (stem + "_active_hybrid.json"))
            null_train = load(active_root / (stem + "_null_hybrid.json"))
            control_train = load(CONTROL_ROOT / (stem + "_control.json"))
            patch_train = load(BASELINE_ROOT / (stem + "_patchtst.json"))
            itrans_train = load(BASELINE_ROOT / (stem + "_itransformer.json"))
            for name, payload in (("active", active_train), ("null", null_train), ("nlinear", control_train), ("patchtst", patch_train), ("itransformer", itrans_train)):
                if payload.get("test_metrics_produced"):
                    raise ValueError("validation artifact opened test: {} {}".format(stem, name))
            if active_train.get("config", {}).get("mamba_correction_output") is not True:
                raise ValueError("candidate residual correction disabled: {}".format(stem))
            if active_train.get("transfer", {}).get("prediction_head_transfer_mode") != "exact":
                raise ValueError("candidate exact head transfer missing: {}".format(stem))
            active_pretrain = load(active_root / (stem + "_active_pretrain.json"))
            if active_pretrain.get("config", {}).get("relation_view") != "forecast_contrastive":
                raise ValueError("candidate is not forecast contrastive: {}".format(stem))
            if active_pretrain.get("config", {}).get("source_pred_len") != horizon:
                raise ValueError("source horizon mismatch: {}".format(stem))
            if active_pretrain.get("test_metrics_produced"):
                raise ValueError("pretraining opened test: {}".format(stem))
            for name in ("active_pretrain", "null_pretrain", "active_hybrid", "null_hybrid"):
                event(active_root / (stem + "_" + name + "_event.json"))
            metrics = {}
            for name in ("active", "nlinear", "patchtst", "itransformer"):
                output = load(ROOT / (stem + "_" + name + "_test.json"))
                if output.get("test_metrics_produced") is not True or output.get("training_performed") is not False:
                    raise ValueError("invalid test reader payload: {} {}".format(stem, name))
                event(ROOT / (stem + "_" + name + "_test_event.json"))
                metrics[name] = float(output["test_mse"])
            stress = {}
            for arm in ("null", "active"):
                payload = load(STRESS_ROOT / (stem + "_" + arm + "_stress.json"))
                if payload.get("test_metrics_produced"):
                    raise ValueError("stress reader opened test: {} {}".format(stem, arm))
                event(STRESS_ROOT / (stem + "_" + arm + "_stress_event.json"))
                stress[arm] = payload["conditions"]
            for condition in STRESSES:
                if condition not in stress["null"] or condition not in stress["active"]:
                    raise ValueError("missing stress condition: {} {}".format(stem, condition))
            rows.append({"horizon": horizon, "seed": seed, **metrics, "stress": stress})
    summary = {}
    for label, (horizon, _) in HORIZONS.items():
        subset = [row for row in rows if row["horizon"] == horizon]
        active = np.asarray([row["active"] for row in subset])
        entry = {"active_mean_mse": float(active.mean())}
        for baseline in ("nlinear", "patchtst", "itransformer"):
            values = np.asarray([row[baseline] for row in subset])
            ratios = active / values
            entry[baseline + "_mean_mse"] = float(values.mean())
            entry["active_vs_" + baseline + "_percent"] = float((active.mean() / values.mean() - 1) * 100)
            entry["active_wins_vs_" + baseline] = int((active < values).sum())
            entry["active_" + baseline + "_ratio_upper_1se"] = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
        entry["stress"] = {}
        for condition in STRESSES:
            null = np.asarray([row["stress"]["null"][condition]["mse"] for row in subset])
            active_stress = np.asarray([row["stress"]["active"][condition]["mse"] for row in subset])
            entry["stress"][condition] = {
                "null_mean_mse": float(null.mean()),
                "active_mean_mse": float(active_stress.mean()),
                "active_vs_null_percent": float((active_stress.mean() / null.mean() - 1) * 100),
                "active_wins": int((active_stress < null).sum()),
            }
        summary["H{}".format(horizon)] = entry
    result = {"release_root": str(ROOT), "source_selection": str(SOURCE_SELECTION), "rows": rows, "summary": summary, "audit_valid": True, "test_metrics_produced": True}
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# PEMS-BAY ACF-InfoNCE H192/H336 Held-Out Audit", "", "All frozen validation checkpoints, test readers, exact-head transfers, and GPU guards passed.", "", "| Horizon | Active | NLinear | PatchTST | iTransformer | vs NLinear | vs PatchTST | vs iTransformer |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for horizon, value in summary.items():
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:+.3f}% | {:+.3f}% | {:+.3f}% |".format(horizon, value["active_mean_mse"], value["nlinear_mean_mse"], value["patchtst_mean_mse"], value["itransformer_mean_mse"], value["active_vs_nlinear_percent"], value["active_vs_patchtst_percent"], value["active_vs_itransformer_percent"]))
    lines += ["", "## Closed Validation Stress", "", "| Horizon | Condition | Active vs null | Wins |", "| --- | --- | ---: | ---: |"]
    for horizon, value in summary.items():
        for condition, stress in value["stress"].items():
            lines.append("| {} | {} | {:+.3f}% | {}/3 |".format(horizon, condition, stress["active_vs_null_percent"], stress["active_wins"]))
    (ROOT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
