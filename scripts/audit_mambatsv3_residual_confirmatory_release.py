#!/usr/bin/env python3
"""Strict reader for the frozen residual-target low-data held-out release."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/paper_final/mambatsv3_residual_autocon_confirmatory_20261003")
COORDINATES = {"traffic": (5201, 5202, 5203), "weather": (5204, 5205, 5206)}
PRIMARY = {"traffic": "DLinear", "weather": "NLinear"}
STRESSES = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")
SOURCE_SELECTION = {
    "traffic": Path("artifacts/exploratory/mambatsv3_fewshot_source_selection_20261003/traffic.json"),
    "weather": Path("artifacts/exploratory/mambatsv3_fewshot_source_selection_20261003/weather.json"),
}
EXPECTED_SOURCES = {
    "traffic": ["Electricity", "METR-LA", "PEMS-BAY"],
    "weather": ["METR-LA", "PEMS-BAY", "Electricity"],
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    rows = []
    for domain, seeds in COORDINATES.items():
        source_selection = load(SOURCE_SELECTION[domain])
        if source_selection.get("main_validation_accessed") or source_selection.get("test_metrics_produced"):
            raise ValueError("source-selection boundary violation: {}".format(domain))
        if [item["name"] for item in source_selection.get("selected", [])] != EXPECTED_SOURCES[domain]:
            raise ValueError("source-selection mismatch: {}".format(domain))
        for seed in seeds:
            stem = "{}_s{}".format(domain, seed)
            selector = load(ROOT / (stem + "_selector.json"))
            control = load(ROOT / (stem + "_control.json"))
            hybrid = load(ROOT / (stem + "_hybrid.json"))
            itransformer = load(ROOT / (stem + "_itransformer.json"))
            patchtst = load(ROOT / (stem + "_patchtst.json"))
            for name, payload in (("control", control), ("hybrid", hybrid), ("iTransformer", itransformer), ("PatchTST", patchtst)):
                if payload.get("test_metrics_produced") is not True or payload.get("test_mse") is None:
                    raise ValueError("missing held-out metric: {} {}".format(stem, name))
            if selector.get("main_validation_accessed") or selector.get("test_metrics_produced"):
                raise ValueError("selection boundary violation: {}".format(stem))
            if selector.get("selected_model") != PRIMARY[domain]:
                raise ValueError("unexpected selected primary: {}".format(stem))
            if hybrid.get("config", {}).get("mamba_correction_output") is not True:
                raise ValueError("candidate residual path disabled: {}".format(stem))
            if hybrid.get("transfer", {}).get("prediction_head_transfer_mode") != "exact":
                raise ValueError("candidate head transfer is not exact: {}".format(stem))
            for name in ("selector", "pretrain", "control", "hybrid", "itransformer", "patchtst"):
                event = load(ROOT / (stem + "_" + name + "_event.json"))
                if event.get("status") != "completed" or event.get("trigger") is not None:
                    raise ValueError("guard/event failure: {} {}".format(stem, name))
            stress = {}
            control_stress = load(ROOT / (stem + "_control_stress.json"))
            hybrid_stress = load(ROOT / (stem + "_hybrid_stress.json"))
            for arm, payload in (("control", control_stress), ("hybrid", hybrid_stress)):
                event = load(ROOT / (stem + "_" + arm + "_stress_event.json"))
                if event.get("status") != "completed" or event.get("trigger") is not None:
                    raise ValueError("stress guard failure: {} {}".format(stem, arm))
                if payload.get("test_metrics_produced") or not payload.get("hybrid"):
                    raise ValueError("stress boundary violation: {} {}".format(stem, arm))
            for condition in STRESSES:
                if condition not in control_stress["conditions"] or condition not in hybrid_stress["conditions"]:
                    raise ValueError("missing stress condition: {} {}".format(stem, condition))
                stress[condition] = {"control_mse": control_stress["conditions"][condition]["mse"], "hybrid_mse": hybrid_stress["conditions"][condition]["mse"]}
            rows.append({"domain": domain, "seed": seed, "primary": selector["selected_model"], "source_selection_artifact": str(SOURCE_SELECTION[domain]), "control_test_mse": control["test_mse"], "hybrid_test_mse": hybrid["test_mse"], "itransformer_test_mse": itransformer["test_mse"], "patchtst_test_mse": patchtst["test_mse"], "control_validation_mse": control["best_validation_mse"], "hybrid_validation_mse": hybrid["best_validation_mse"], "stress": stress})
    summary = {}
    for domain in COORDINATES:
        subset = [row for row in rows if row["domain"] == domain]
        control = np.asarray([row["control_test_mse"] for row in subset])
        hybrid = np.asarray([row["hybrid_test_mse"] for row in subset])
        itransformer = np.asarray([row["itransformer_test_mse"] for row in subset])
        patchtst = np.asarray([row["patchtst_test_mse"] for row in subset])
        ratios = hybrid / control
        upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
        summary[domain] = {"control_mean_mse": float(control.mean()), "hybrid_mean_mse": float(hybrid.mean()), "itransformer_mean_mse": float(itransformer.mean()), "patchtst_mean_mse": float(patchtst.mean()), "hybrid_vs_control_percent": float((hybrid.mean() / control.mean() - 1.0) * 100.0), "hybrid_vs_itransformer_percent": float((hybrid.mean() / itransformer.mean() - 1.0) * 100.0), "hybrid_vs_patchtst_percent": float((hybrid.mean() / patchtst.mean() - 1.0) * 100.0), "paired_wins_vs_control": int((hybrid < control).sum()), "paired_wins_vs_patchtst": int((hybrid < patchtst).sum()), "paired_ratio_upper_1se": upper, "noninferior_1p02": bool(upper < 1.02), "all_hybrid_beats_itransformer": bool((hybrid < itransformer).all())}
        stress_summary = {}
        for condition in STRESSES:
            stress_control = np.asarray([row["stress"][condition]["control_mse"] for row in subset])
            stress_hybrid = np.asarray([row["stress"][condition]["hybrid_mse"] for row in subset])
            stress_summary[condition] = {"control_mean_mse": float(stress_control.mean()), "hybrid_mean_mse": float(stress_hybrid.mean()), "hybrid_vs_control_percent": float((stress_hybrid.mean() / stress_control.mean() - 1.0) * 100.0), "paired_wins": int((stress_hybrid < stress_control).sum())}
        summary[domain]["stress"] = stress_summary
    output = {"release_root": str(ROOT), "source_selection_artifacts": {key: str(value) for key, value in SOURCE_SELECTION.items()}, "rows": rows, "summary": summary, "audit_valid": True, "test_metrics_produced": True}
    (ROOT / "audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# Residual-Target Confirmatory Held-Out Audit", "", "All six frozen coordinates passed target-primary selection, pre-existing few-shot source-selection, exact-head, test-metric, and GPU-guard checks.", "", "| Target | Control test MSE | Hybrid test MSE | Hybrid vs control | Wins | Upper ratio | iTransformer | PatchTST |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for domain, value in summary.items():
        lines.append("| {} | {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:.4f} | {:.6f} | {:.6f} |".format(domain, value["control_mean_mse"], value["hybrid_mean_mse"], value["hybrid_vs_control_percent"], value["paired_wins_vs_control"], value["paired_ratio_upper_1se"], value["itransformer_mean_mse"], value["patchtst_mean_mse"]))
    lines += ["", "## Closed Validation Stress", "", "| Target | Condition | Hybrid vs selected primary | Wins |", "| --- | --- | ---: | ---: |"]
    for domain, value in summary.items():
        for condition, stress in value["stress"].items():
            lines.append("| {} | {} | {:+.3f}% | {}/3 |".format(domain, condition, stress["hybrid_vs_control_percent"], stress["paired_wins"]))
    (ROOT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
