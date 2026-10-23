#!/usr/bin/env python3
"""Relate train-prefix source compatibility to frozen active/null transfer gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGETS = {
    "PEMS-BAY": {
        "selection": Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_selection_20261003/pems_bay_sources.json"),
        "result": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_confirmation_20261003/audit.json"),
        "fraction": 0.10,
    },
    "METR-LA": {
        "selection": Path("artifacts/paper_final/mambatsv3_residual_autocon_expansion_selection_20261003/metr_la_sources.json"),
        "result": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_metr_confirmation_20261003/audit.json"),
        "fraction": 0.10,
    },
    "Weather": {
        "selection": Path("artifacts/exploratory/mambatsv3_fewshot_source_selection_20261003/weather.json"),
        "result": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_weather_confirmation_20261003/audit.json"),
        "fraction": 0.10,
    },
    "Electricity": {
        "selection": Path("artifacts/paper_final/mambatsv3_acf_infonce_electricity_selection_20261003/sources.json"),
        "result": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_electricity_traffic_25_boundary_20261003"),
        "fraction": 0.25,
    },
    "Traffic": {
        "selection": Path("artifacts/exploratory/mambatsv3_fewshot_source_selection_20261003/traffic.json"),
        "result": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_electricity_traffic_25_boundary_20261003"),
        "fraction": 0.25,
    },
}


def _load_result(target: str, root: Path):
    if root.is_file():
        payload = json.loads(root.read_text(encoding="utf-8"))
        return float(payload["summary"]["active_vs_null_percent"])
    seeds = {
        "Electricity": (5550, 5551, 5552),
        "Traffic": (5560, 5561, 5562),
    }[target]
    active, null = [], []
    for seed in seeds:
        active.append(json.loads((root / f"{target.lower()}_s{seed}_active_hybrid.json").read_text())["best_validation_mse"])
        null.append(json.loads((root / f"{target.lower()}_s{seed}_null_hybrid.json").read_text())["best_validation_mse"])
    return float((np.mean(active) / np.mean(null) - 1.0) * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for target, spec in TARGETS.items():
        selection = json.loads(spec["selection"].read_text(encoding="utf-8"))
        selected = selection["selected"]
        similarity = np.asarray([float(item["similarity"]) for item in selected], dtype=float)
        gain = _load_result(target, spec["result"])
        rows.append({
            "target": target,
            "few_shot_fraction": spec["fraction"],
            "selected_sources": [item["name"] for item in selected],
            "mean_selected_source_similarity": float(similarity.mean()),
            "min_selected_source_similarity": float(similarity.min()),
            "active_vs_null_percent": gain,
            "active_better": bool(gain < 0),
        })
    compatibility = np.asarray([row["mean_selected_source_similarity"] for row in rows])
    gains = np.asarray([row["active_vs_null_percent"] for row in rows])
    result = {
        "rows": rows,
        "exploratory": True,
        "interpretation": "Five target-level points only; compatibility/gain association is descriptive and requires preregistered future domains.",
        "pearson_r": float(np.corrcoef(compatibility, gains)[0, 1]),
        "test_metrics_produced": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ACF Transfer Compatibility Analysis", "",
        "Exploratory descriptive analysis over five target-level points; no test metrics are produced here.", "",
        "| Target | Labels | Mean selected-source similarity | Active vs null |", "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {} | {:.0%} | {:.3f} | {:+.3f}% |".format(row["target"], row["few_shot_fraction"], row["mean_selected_source_similarity"], row["active_vs_null_percent"]))
    lines += ["", "Pearson r (descriptive only): {:.3f}".format(result["pearson_r"]), "", "The source-compatibility score is computed from target train-prefix source selection artifacts; it is not a post-hoc test-set feature."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
