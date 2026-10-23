#!/usr/bin/env python3
"""Strict reader for the METR-LA H192 ACF-InfoNCE expansion release."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003")
ACTIVE_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_metr_la_h192_screen_20261003")
BASELINE_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_metr_la_h192_baselines_validation_20261003")
TIMESNET_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_metr_la_h192_timesnet_validation_20261003")
STRESS_ROOT = Path("artifacts/exploratory/mambatsv3_acf_infonce_metr_la_h192_stress_validation_20261003")
SOURCE_SELECTION = Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_selection_20261003/metr_la_sources.json")
SEEDS = (5440, 5441, 5442)
EXPECTED_SOURCES = ["PEMS-BAY", "PEMS07", "PEMS08"]
STRESSES = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guard(path):
    payload = load(path)
    if payload.get("status") != "completed" or payload.get("trigger") is not None:
        raise ValueError("GPU guard failure: {}".format(path))


def main() -> int:
    selector = load(SOURCE_SELECTION)
    if selector.get("main_validation_accessed") or selector.get("test_metrics_produced"):
        raise ValueError("source selection boundary violation")
    if [item["name"] for item in selector.get("selected", [])] != EXPECTED_SOURCES:
        raise ValueError("source selection mismatch")
    rows = []
    for seed in SEEDS:
        stem = "metr_la_h192_s{}".format(seed)
        active_train = load(ACTIVE_ROOT / (stem + "_active_hybrid.json"))
        null_train = load(ACTIVE_ROOT / (stem + "_null_hybrid.json"))
        pretrain = load(ACTIVE_ROOT / (stem + "_active_pretrain.json"))
        validation = {
            "nlinear": load(BASELINE_ROOT / (stem + "_nlinear.json")),
            "patchtst": load(BASELINE_ROOT / (stem + "_patchtst.json")),
            "itransformer": load(BASELINE_ROOT / (stem + "_itransformer.json")),
            "timesnet": load(TIMESNET_ROOT / (stem + "_timesnet.json")),
        }
        for name, payload in {"active": active_train, "null": null_train, "pretrain": pretrain, **validation}.items():
            if payload.get("test_metrics_produced"):
                raise ValueError("validation artifact opened test: {} {}".format(stem, name))
        if active_train.get("config", {}).get("mamba_correction_output") is not True:
            raise ValueError("residual correction disabled: {}".format(stem))
        if active_train.get("transfer", {}).get("prediction_head_transfer_mode") != "exact":
            raise ValueError("exact head missing: {}".format(stem))
        if pretrain.get("config", {}).get("relation_view") != "forecast_contrastive" or pretrain.get("config", {}).get("source_pred_len") != 192:
            raise ValueError("unexpected pretraining contract: {}".format(stem))
        for name in ("active_pretrain", "null_pretrain", "active_hybrid", "null_hybrid"):
            guard(ACTIVE_ROOT / (stem + "_" + name + "_event.json"))
        metrics = {}
        for name in ("active", "nlinear", "patchtst", "itransformer", "timesnet"):
            test = load(ROOT / (stem + "_" + name + "_test.json"))
            if test.get("test_metrics_produced") is not True or test.get("training_performed") is not False:
                raise ValueError("invalid held-out payload: {} {}".format(stem, name))
            guard(ROOT / (stem + "_" + name + "_test_event.json"))
            metrics[name] = float(test["test_mse"])
        stress = {}
        for arm in ("null", "active"):
            payload = load(STRESS_ROOT / (stem + "_" + arm + "_stress.json"))
            if payload.get("test_metrics_produced"):
                raise ValueError("stress reader opened test")
            guard(STRESS_ROOT / (stem + "_" + arm + "_stress_event.json"))
            stress[arm] = payload["conditions"]
        rows.append({"seed": seed, **metrics, "stress": stress})
    summary = {"active_mean_mse": float(np.mean([row["active"] for row in rows]))}
    active = np.asarray([row["active"] for row in rows])
    for baseline in ("nlinear", "patchtst", "itransformer", "timesnet"):
        values = np.asarray([row[baseline] for row in rows])
        ratio = active / values
        summary[baseline + "_mean_mse"] = float(values.mean())
        summary["active_vs_" + baseline + "_percent"] = float((active.mean() / values.mean() - 1) * 100)
        summary["active_wins_vs_" + baseline] = int((active < values).sum())
        summary["active_" + baseline + "_ratio_upper_1se"] = float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)))
    summary["stress"] = {}
    for condition in STRESSES:
        null = np.asarray([row["stress"]["null"][condition]["mse"] for row in rows])
        active_stress = np.asarray([row["stress"]["active"][condition]["mse"] for row in rows])
        summary["stress"][condition] = {"null_mean_mse": float(null.mean()), "active_mean_mse": float(active_stress.mean()), "active_vs_null_percent": float((active_stress.mean() / null.mean() - 1) * 100), "active_wins": int((active_stress < null).sum())}
    result = {"release_root": str(ROOT), "source_selection": str(SOURCE_SELECTION), "rows": rows, "summary": summary, "audit_valid": True, "test_metrics_produced": True}
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# METR-LA H192 ACF-InfoNCE Held-Out Audit", "", "All selection, exact-head, test-reader, stress, and GPU-guard contracts passed.", "", "| Active | NLinear | PatchTST | iTransformer | TimesNet | vs NLinear | vs PatchTST | vs iTransformer | vs TimesNet |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", "| {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:+.3f}% | {:+.3f}% | {:+.3f}% | {:+.3f}% |".format(summary["active_mean_mse"], summary["nlinear_mean_mse"], summary["patchtst_mean_mse"], summary["itransformer_mean_mse"], summary["timesnet_mean_mse"], summary["active_vs_nlinear_percent"], summary["active_vs_patchtst_percent"], summary["active_vs_itransformer_percent"], summary["active_vs_timesnet_percent"]), "", "## Closed Validation Stress", "", "| Condition | Active vs null | Wins |", "| --- | ---: | ---: |"]
    for condition, value in summary["stress"].items():
        lines.append("| {} | {:+.3f}% | {}/3 |".format(condition, value["active_vs_null_percent"], value["active_wins"]))
    (ROOT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
