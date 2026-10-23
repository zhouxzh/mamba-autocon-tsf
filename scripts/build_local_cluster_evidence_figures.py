#!/usr/bin/env python3
"""Render provenance-bound publication figures from the local-cluster evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUNDLE = Path("artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/bundle.json")
OUTPUT = Path("artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/figures")


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def _failed_gate_percent(rows):
    return [
        (float(row["active_validation"]) / float(row["null_validation"]) - 1.0) * 100.0
        for row in rows
    ]


def main() -> int:
    bundle = load(BUNDLE)
    confirmations = bundle["confirmations"]
    failures = bundle["failed_gates"]
    robustness = bundle["robustness_extensions"]
    inference = bundle["inference_boundary"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    confirm_labels = ["{}\nH{} {}".format(row["target"], row["horizon"], "{:.0%}".format(row["few_shot_fraction"])) for row in confirmations]
    confirm_values = [float(row["active_vs_null_percent"]) for row in confirmations]
    failure_labels, failure_values = [], []
    for key, payload in failures.items():
        values = _failed_gate_percent(payload["rows"])
        failure_labels.append(key.replace("_", "\n"))
        failure_values.append(float(np.mean(values)))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.75), layout="constrained")
    positive = "#0072B2"
    boundary = "#D55E00"
    neutral = "#4D4D4D"

    ax = axes[0]
    labels = confirm_labels + failure_labels
    values = confirm_values + failure_values
    bars = ax.bar(np.arange(len(values)), values, color=[positive] * len(confirm_values) + [boundary] * len(failure_values), edgecolor="black", linewidth=.6)
    for bar in bars[len(confirm_values):]: bar.set_hatch("//")
    ax.axhline(0, color=neutral, linewidth=.8)
    ax.set_ylim(min(values) - .08, .08)
    ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Active vs exact-null MSE (%)")
    ax.set_title("A. Held-out confirmations and failed gates", fontsize=10)
    ax.text(.01, .03, "Negative is better", transform=ax.transAxes, va="bottom", fontsize=8)

    ax = axes[1]
    robust_labels, centers, lowers, uppers = [], [], [], []
    for key, payload in robustness.items():
        effect = [float(item["active_vs_null_percent"]) for item in payload["summary"].values()]
        robust_labels.append(key.replace("_", "\n"))
        centers.append(float(np.mean(effect)))
        lowers.append(float(np.mean(effect) - min(effect)))
        uppers.append(float(max(effect) - np.mean(effect)))
    positions = np.arange(len(robust_labels))
    ax.errorbar(positions, centers, yerr=np.asarray([lowers, uppers]), fmt="o", color=positive, markeredgecolor="black", capsize=4, linewidth=1.2)
    ax.axhline(0, color=neutral, linewidth=.8)
    ax.set_ylim(min(np.asarray(centers) - np.asarray(lowers)) - .08, .08)
    ax.set_xticks(positions, robust_labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Active vs exact-null MSE (%)")
    ax.set_title("B. Validation robustness range", fontsize=10)
    ax.text(.01, .03, "Mean and min/max across six conditions", transform=ax.transAxes, va="bottom", fontsize=8)

    ax = axes[2]
    active = inference["active"]
    official = inference["official_patchtst"]
    ax.scatter(active["peak_allocated_mb"], active["latency_ms_median"], color=positive, marker="o", s=55, edgecolor="black", label="Active single-scan")
    ax.scatter(official["peak_allocated_mb"], official["latency_ms_median"], color=boundary, marker="s", s=55, edgecolor="black", label="Official PatchTST")
    ax.annotate("Active single-scan", (active["peak_allocated_mb"], active["latency_ms_median"]), xytext=(8, -12), textcoords="offset points", fontsize=8)
    ax.annotate("Official PatchTST", (official["peak_allocated_mb"], official["latency_ms_median"]), xytext=(-80, 8), textcoords="offset points", fontsize=8)
    ax.set_xlim(20.5, 27.3)
    ax.set_ylim(.248, .323)
    ax.set_xlabel("Peak allocated memory (MB)")
    ax.set_ylabel("Median forward latency (ms)")
    ax.set_title("C. H336 inference boundary", fontsize=10)
    ax.text(.02, .04, "Batch 4; device-resident", transform=ax.transAxes, va="bottom", fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / ("local_cluster_evidence." + ext), dpi=300, facecolor="white")
    plt.close(fig)
    manifest = {
        "figure": "local_cluster_evidence",
        "source_bundle": str(BUNDLE),
        "source_audits": {
            "confirmations": [row["path"] for row in confirmations],
            "failed_gates": {key: value["path"] for key, value in failures.items()},
            "robustness": {key: value["path"] for key, value in robustness.items()},
            "inference_boundary": "artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_h336_extension_20261003/inference_boundary.json",
        },
        "transformations": ["direct audited percentages", "robustness panel shows mean and min/max across six closed validation conditions"],
        "uncertainty": "Panel B is a condition range, not a confidence interval.",
        "test_metrics_produced": False,
    }
    (OUTPUT / "local_cluster_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
