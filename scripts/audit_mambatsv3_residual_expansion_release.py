#!/usr/bin/env python3
"""Audit the frozen METR-LA/PEMS-BAY residual-pretraining release."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_confirmatory_20261003")
VALIDATION_ROOT = Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_validation_20261003")
SELECTION_ROOT = Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_selection_20261003")
COORDINATES = {"metr_la": (5301, 5303, 5305), "pems_bay": (5302, 5304, 5306)}
DATASETS = {"metr_la": "dataset/METR-LA/METR-LA.csv", "pems_bay": "dataset/PEMS-BAY/PEMS-BAY.csv"}
SOURCES = {
    "metr_la": ["PEMS-BAY", "PEMS07", "PEMS08"],
    "pems_bay": ["PEMS07", "PEMS08", "PEMS04"],
}
PRIMARY = {"metr_la": "NLinear", "pems_bay": "NLinear"}
STRESSES = (
    "in_distribution", "late_validation_shift", "missing_channels_10",
    "missing_channels_30", "noise_0p05", "noise_0p10",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_event(path: Path) -> None:
    event = load(path)
    if event.get("status") != "completed" or event.get("trigger") is not None:
        raise ValueError("guard/event failure: {}".format(path))


def main() -> int:
    rows = []
    for domain, seeds in COORDINATES.items():
        selection = load(SELECTION_ROOT / (domain + "_sources.json"))
        if selection.get("main_validation_accessed") or selection.get("test_metrics_produced"):
            raise ValueError("source selection boundary violation: {}".format(domain))
        if [item["name"] for item in selection.get("selected", [])] != SOURCES[domain]:
            raise ValueError("source selection mismatch: {}".format(domain))
        primary_selection = load(SELECTION_ROOT / (domain + "_primary.json"))
        if primary_selection.get("main_validation_accessed") or primary_selection.get("test_metrics_produced"):
            raise ValueError("primary selection boundary violation: {}".format(domain))
        if primary_selection.get("selected_model") != PRIMARY[domain] or primary_selection.get("selected_sampler") != "uniform":
            raise ValueError("primary selection mismatch: {}".format(domain))
        for seed in seeds:
            stem = "{}_s{}".format(domain, seed)
            selector = primary_selection
            pretrain = load(VALIDATION_ROOT / (stem + "_pretrain.json"))
            control = load(ROOT / (stem + "_control.json"))
            hybrid = load(ROOT / (stem + "_hybrid.json"))
            itransformer = load(ROOT / (stem + "_itransformer.json"))
            patchtst = load(ROOT / (stem + "_patchtst.json"))
            if pretrain.get("test_metrics_produced"):
                raise ValueError("pretraining opened test metrics: {}".format(stem))
            if pretrain.get("config", {}).get("prediction_target") != "last_value_residual":
                raise ValueError("unexpected pretraining target: {}".format(stem))
            if pretrain.get("config", {}).get("relation_view") != "forecast_affine":
                raise ValueError("unexpected relation view: {}".format(stem))
            for name, payload in (("control", control), ("hybrid", hybrid), ("iTransformer", itransformer), ("PatchTST", patchtst)):
                if payload.get("test_metrics_produced") is not True or payload.get("test_mse") is None:
                    raise ValueError("missing held-out metric: {} {}".format(stem, name))
            if selector.get("selected_model") != PRIMARY[domain]:
                raise ValueError("unexpected primary model: {}".format(stem))
            if hybrid.get("config", {}).get("mamba_correction_output") is not True:
                raise ValueError("residual output disabled: {}".format(stem))
            if hybrid.get("transfer", {}).get("prediction_head_transfer_mode") != "exact":
                raise ValueError("exact head transfer missing: {}".format(stem))
            check_event(VALIDATION_ROOT / (stem + "_pretrain_event.json"))
            for name in ("control", "hybrid", "itransformer", "patchtst"):
                check_event(ROOT / (stem + "_" + name + "_event.json"))
            stress = {}
            control_stress = load(ROOT / (stem + "_control_stress.json"))
            hybrid_stress = load(ROOT / (stem + "_hybrid_stress.json"))
            for arm, payload in (("control", control_stress), ("hybrid", hybrid_stress)):
                if payload.get("test_metrics_produced"):
                    raise ValueError("stress opened test metrics: {} {}".format(stem, arm))
                check_event(ROOT / (stem + "_" + arm + "_stress_event.json"))
            for condition in STRESSES:
                if condition not in control_stress.get("conditions", {}) or condition not in hybrid_stress.get("conditions", {}):
                    raise ValueError("missing stress condition: {} {}".format(stem, condition))
                stress[condition] = {
                    "control_mse": control_stress["conditions"][condition]["mse"],
                    "hybrid_mse": hybrid_stress["conditions"][condition]["mse"],
                }
            rows.append({
                "domain": domain, "seed": seed, "primary": selector["selected_model"],
                "dataset": DATASETS[domain], "source_selection": SOURCES[domain],
                "control_test_mse": control["test_mse"], "hybrid_test_mse": hybrid["test_mse"],
                "itransformer_test_mse": itransformer["test_mse"], "patchtst_test_mse": patchtst["test_mse"],
                "control_validation_mse": control["best_validation_mse"],
                "hybrid_validation_mse": hybrid["best_validation_mse"], "stress": stress,
            })

    summary = {}
    for domain in COORDINATES:
        subset = [row for row in rows if row["domain"] == domain]
        arrays = {name: np.asarray([row[name + "_test_mse"] for row in subset]) for name in ("control", "hybrid", "itransformer", "patchtst")}
        ratio = arrays["hybrid"] / arrays["control"]
        upper = float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)))
        item = {
            "control_mean_mse": float(arrays["control"].mean()), "hybrid_mean_mse": float(arrays["hybrid"].mean()),
            "itransformer_mean_mse": float(arrays["itransformer"].mean()), "patchtst_mean_mse": float(arrays["patchtst"].mean()),
            "hybrid_vs_control_percent": float((arrays["hybrid"].mean() / arrays["control"].mean() - 1) * 100),
            "hybrid_vs_itransformer_percent": float((arrays["hybrid"].mean() / arrays["itransformer"].mean() - 1) * 100),
            "hybrid_vs_patchtst_percent": float((arrays["hybrid"].mean() / arrays["patchtst"].mean() - 1) * 100),
            "paired_wins_vs_control": int((arrays["hybrid"] < arrays["control"]).sum()),
            "paired_wins_vs_itransformer": int((arrays["hybrid"] < arrays["itransformer"]).sum()),
            "paired_wins_vs_patchtst": int((arrays["hybrid"] < arrays["patchtst"]).sum()),
            "paired_ratio_upper_1se": upper, "noninferior_1p02": bool(upper < 1.02),
        }
        item["stress"] = {}
        for condition in STRESSES:
            c = np.asarray([row["stress"][condition]["control_mse"] for row in subset])
            h = np.asarray([row["stress"][condition]["hybrid_mse"] for row in subset])
            item["stress"][condition] = {"control_mean_mse": float(c.mean()), "hybrid_mean_mse": float(h.mean()), "hybrid_vs_control_percent": float((h.mean() / c.mean() - 1) * 100), "paired_wins": int((h < c).sum())}
        summary[domain] = item

    output = {"release_root": str(ROOT), "validation_root": str(VALIDATION_ROOT), "rows": rows, "summary": summary, "audit_valid": True, "test_metrics_produced": True}
    (ROOT / "audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# Residual-Target Expansion Confirmatory Audit", "", "All six METR-LA/PEMS-BAY coordinates passed few-shot source selection, primary-model selection, exact-head transfer, held-out metrics, and GPU-guard checks.", "", "| Target | Control | Hybrid | vs control | Wins | vs iTransformer | vs PatchTST |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for domain, item in summary.items():
        lines.append("| {} | {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:+.3f}% | {:+.3f}% |".format(domain, item["control_mean_mse"], item["hybrid_mean_mse"], item["hybrid_vs_control_percent"], item["paired_wins_vs_control"], item["hybrid_vs_itransformer_percent"], item["hybrid_vs_patchtst_percent"]))
    lines += ["", "## Closed Validation Stress", "", "| Target | Condition | vs selected primary | Wins |", "| --- | --- | ---: | ---: |"]
    for domain, item in summary.items():
        for condition, value in item["stress"].items():
            lines.append("| {} | {} | {:+.3f}% | {}/3 |".format(domain, condition, value["hybrid_vs_control_percent"], value["paired_wins"]))
    (ROOT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
