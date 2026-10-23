#!/usr/bin/env python3
"""Read the frozen Weather MambaTS v3 test release without model selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats

PLAN_PHASES = {
    "active": ("weather_selection_confirmatory_active_test", 15, "MambaTSv3AutoCon"),
    "baseline": ("weather_confirmatory_strong_baselines_test", 60, None),
}
BASELINES = ("PatchTST", "TimesNet", "iTransformer", "Mamba2")
HORIZONS = (96, 336, 720)
SEEDS = (2031, 2032, 2033, 2034, 2035)


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase_rows(root: Path, phase: str, expected: int, required_model: str | None):
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError("invalid phase validation: {}".format(root))
    provenance = load(root / "provenance.json")
    launch = provenance.get("launch_authorization", {})
    if launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError("missing enforced launch binding: {}".format(root))
    source = provenance.get("source_tree_sha256")
    rows = []
    seen = set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["job_id"] in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest job")
        seen.add(record["job_id"])
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source mismatch: {}".format(record["job_id"]))
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        stream = load(result / "stream_metrics.json")
        model = str(config["model"])
        if required_model and model != required_model:
            raise ValueError("model mismatch: {}".format(model))
        if config.get("skip_test") is True or stream.get("aggregation") != "streaming_window_weighted":
            raise ValueError("test contract mismatch: {}".format(record["job_id"]))
        mse = float(stream["scaled"]["mse"])
        mae = float(stream["scaled"]["mae"])
        if not (math.isfinite(mse) and mse > 0 and math.isfinite(mae) and mae >= 0):
            raise ValueError("invalid stream metric: {}".format(record["job_id"]))
        rows.append({"phase": phase, "model": model, "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "scaled_mse": mse, "scaled_mae": mae, "original_mse": float(stream["original"]["mse"]), "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"])})
    if len(rows) != expected:
        raise ValueError("manifest coverage mismatch")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label, (phase, count, model) in PLAN_PHASES.items():
        rows.extend(phase_rows(args.artifact_root / phase, phase, count, model))
    index = {(r["model"], r["horizon"], r["seed"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate model/horizon/seed coordinate")
    summary = []
    for horizon in HORIZONS:
        candidate = [index[("MambaTSv3AutoCon", horizon, seed)] for seed in SEEDS]
        for baseline in BASELINES:
            reference = [index[(baseline, horizon, seed)] for seed in SEEDS]
            log_ratios = [math.log(c["scaled_mse"] / r["scaled_mse"]) for c, r in zip(candidate, reference)]
            mean_log = statistics.fmean(log_ratios)
            se = statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
            upper_ratio = math.exp(mean_log + stats.t.ppf(0.95, len(log_ratios) - 1) * se)
            ratios = [c["scaled_mse"] / r["scaled_mse"] for c, r in zip(candidate, reference)]
            summary.append({
                "horizon": horizon, "baseline": baseline,
                "candidate_mse_mean": statistics.fmean(x["scaled_mse"] for x in candidate),
                "baseline_mse_mean": statistics.fmean(x["scaled_mse"] for x in reference),
                "mean_mse_change_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
                "candidate_wins": sum(value < 1.0 for value in ratios),
                "n": len(ratios), "upper_ratio_95_one_sided": upper_ratio,
                "noninferior_ratio_1p02": upper_ratio <= 1.02,
                "candidate_elapsed_mean_sec": statistics.fmean(x["elapsed_sec"] for x in candidate),
                "baseline_elapsed_mean_sec": statistics.fmean(x["elapsed_sec"] for x in reference),
                "candidate_elapsed_faster": statistics.fmean(x["elapsed_sec"] for x in candidate) < statistics.fmean(x["elapsed_sec"] for x in reference),
            })
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / "test_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "paired_test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    lines = ["# Frozen Weather MambaTS v3 Test Release", "", "Five fresh paired seeds; one-sided 95% upper MSE-ratio bound uses the predeclared 1.02 non-inferiority threshold.", "", "| H | Baseline | Candidate MSE | Baseline MSE | Change | Wins | Upper ratio | Non-inferior | Candidate elapsed | Baseline elapsed |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for item in summary:
        lines.append("| {horizon} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {mean_mse_change_pct:+.3f}% | {candidate_wins}/{n} | {upper_ratio_95_one_sided:.4f} | {noninferior_ratio_1p02} | {candidate_elapsed_mean_sec:.2f}s | {baseline_elapsed_mean_sec:.2f}s |".format(**item))
    lines.extend(["", "Wall time includes one full job under the shared protocol and is not a hardware-independent training-throughput claim. The result is Weather-only and does not test the independent AutoCon effect, whose lambda-zero control was intentionally retained in the prior validation screen."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": True})


if __name__ == "__main__":
    main()
