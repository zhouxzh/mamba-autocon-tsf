#!/usr/bin/env python3
"""Strict reader for the frozen ETTh2/ETTm2 linear-blend transfer screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DATASETS = ("ETTh2", "ETTm2")
SEEDS = (3201, 3202, 3203)
BASELINES = ("RLinear", "PatchTST")
CANDIDATE = "MambaTSv3AutoCon"


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_arm(root: Path, kind: str, expected_jobs: int, models: set[str]):
    phase = load(root / "phase_validation.json")
    if phase.get("valid") is not True or phase.get("expected_jobs") != expected_jobs:
        raise ValueError("phase validation failed: {}".format(root))
    source = load(root / "provenance.json").get("source_tree_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError("missing source snapshot")
    rows = []
    seen = set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = record.get("job_id")
        if job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source mismatch: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        model = str(config.get("model"))
        if model not in models or config.get("skip_test") is not True:
            raise ValueError("model/test contract mismatch: {}".format(job_id))
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact found: {}".format(job_id))
        if kind == "blend":
            if config.get("mamba_ts_repr_linear_expert") is not True or config.get("mamba_ts_repr_linear_expert_mode") != "blend":
                raise ValueError("blend contract mismatch: {}".format(job_id))
            if float(config.get("mamba_ts_repr_linear_blend_bias")) != -4.0:
                raise ValueError("blend bias mismatch: {}".format(job_id))
        elif config.get("mamba_ts_repr_linear_expert") is not False:
            raise ValueError("non-blend arm received expert: {}".format(job_id))
        command = record["command"]
        checkpoint_root = Path(command[command.index("--checkpoints") + 1]) / result.name
        values = [float(value) for value in load(checkpoint_root / "loss_logs.json").get("val_loss", [])]
        if not values:
            raise ValueError("missing validation loss: {}".format(job_id))
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "model": model, "arm": kind, "best_val_mse": min(values), "elapsed_sec": float(record["elapsed_sec"]),
        })
    expected = {(dataset, 96, seed, model) for dataset in DATASETS for seed in SEEDS for model in models}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if len(rows) != expected_jobs or found != expected:
        raise ValueError("coordinate mismatch: {}".format(root))
    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate, source = read_arm(args.artifact_root / "blend_h96", "blend", 6, {CANDIDATE})
    control, control_source = read_arm(args.artifact_root / "control_h96", "control", 6, {CANDIDATE})
    baselines, baseline_source = read_arm(args.artifact_root / "baselines_h96", "baseline", 12, set(BASELINES))
    if source != control_source or source != baseline_source:
        raise ValueError("arms have different source snapshots")
    rows = candidate + control + baselines
    index = {(row["dataset"], row["seed"], row["model"], row["arm"]): row for row in rows}
    summary = []
    paired = []
    for dataset in DATASETS:
        blend_values = [index[(dataset, seed, CANDIDATE, "blend")]["best_val_mse"] for seed in SEEDS]
        control_values = [index[(dataset, seed, CANDIDATE, "control")]["best_val_mse"] for seed in SEEDS]
        baseline_means = {model: statistics.fmean(index[(dataset, seed, model, "baseline")]["best_val_mse"] for seed in SEEDS) for model in BASELINES}
        strongest = min(baseline_means, key=baseline_means.get)
        for model in BASELINES:
            deltas = [index[(dataset, seed, CANDIDATE, "blend")]["best_val_mse"] - index[(dataset, seed, model, "baseline")]["best_val_mse"] for seed in SEEDS]
            paired.append({
                "dataset": dataset, "baseline": model, "candidate_mean_val_mse": statistics.fmean(blend_values),
                "baseline_mean_val_mse": baseline_means[model], "mean_delta_mse": statistics.fmean(deltas),
                "relative_delta_pct": 100.0 * statistics.fmean(deltas) / baseline_means[model], "candidate_wins": sum(delta < 0 for delta in deltas), "n": len(SEEDS),
            })
        blend_mean = statistics.fmean(blend_values); control_mean = statistics.fmean(control_values); strongest_mean = baseline_means[strongest]
        summary.append({
            "dataset": dataset, "blend_mean": blend_mean, "control_mean": control_mean,
            "blend_vs_control_pct": 100.0 * (blend_mean / control_mean - 1.0),
            "control_wins": sum(b < c for b, c in zip(blend_values, control_values)), "strongest_baseline": strongest,
            "strongest_mean": strongest_mean, "blend_vs_strongest_pct": 100.0 * (blend_mean / strongest_mean - 1.0),
            "baseline_means": baseline_means,
        })
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    for name, values in (("validation_rows.csv", rows), ("paired_summary.csv", paired), ("summary.csv", summary)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ["# Current-source ETTh2/ETTm2 AutoCon linear-blend transfer", "", "Validation-only strict readout. No test metric or prediction artifact was read or produced.", "", "| Dataset | Blend MSE | Exact v3 MSE | Blend vs v3 | Wins | Strongest baseline | Blend vs strongest | Gate |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |"]
    for item in summary:
        paired_gate = item["blend_vs_control_pct"] < 0.0 and item["control_wins"] >= 2
        external_gate = item["blend_vs_strongest_pct"] <= 2.0
        gate = paired_gate and external_gate
        lines.append("| {dataset} | {blend_mean:.6f} | {control_mean:.6f} | {blend_vs_control_pct:+.2f}% | {control_wins}/3 | {strongest_baseline} | {blend_vs_strongest_pct:+.2f}% | {gate} |".format(gate="PASS" if gate else "FAIL", **item))
        lines.append("|  | baselines: " + ", ".join("{}={:.6f}".format(k, v) for k, v in item["baseline_means"].items()) + " |  |  |  |  |  |  |")
    lines.extend(["", "Negative deltas favor the blend. All rows share source snapshot `{}`.".format(source), "This transfer screen is validation-only and does not establish causal AutoCon attribution or authorize held-out testing."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "summary_rows": len(summary), "source_tree_sha256": source, "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
