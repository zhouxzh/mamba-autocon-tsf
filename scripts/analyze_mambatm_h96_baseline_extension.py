#!/usr/bin/env python3
"""Strict validation-only reader for the TimeMachine Mamba baseline extension."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = (
    "DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer",
    "Mamba2AutoCon", "MambaTSv3AutoCon", "MambaTM1AutoCon", "MambaTM2AutoCon",
)
DATASETS = ("ETTh1", "weather", "exchange_rate", "national_illness")
SEEDS = (2021, 2022, 2023)
HORIZON = 96
AUTOCON_MODELS = frozenset(model for model in MODELS if model.endswith("AutoCon"))


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rows_for_phase(root: Path):
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != 108:
        raise ValueError("invalid phase validation: {}".format(root))
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    rows = []
    seen = set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source snapshot mismatch: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        if config.get("skip_test") is not True or any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test boundary violated: {}".format(job_id))
        model = str(config.get("model"))
        if model not in MODELS:
            raise ValueError("unexpected model: {}".format(model))
        if model in AUTOCON_MODELS:
            if config.get("AutoCon") is not True or float(config.get("AutoCon_lambda", -1.0)) != 0.001:
                raise ValueError("AutoCon contract mismatch: {}".format(job_id))
        elif config.get("AutoCon"):
            raise ValueError("plain baseline unexpectedly enables AutoCon: {}".format(job_id))
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        loss = load(checkpoint / "loss_logs.json")
        val_loss = [float(value) for value in loss.get("val_loss", [])]
        if not val_loss or not all(math.isfinite(value) for value in val_loss):
            raise ValueError("missing/non-finite validation loss: {}".format(job_id))
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]), "model": model,
            "best_val_mse": min(val_loss), "elapsed_sec": float(record["elapsed_sec"]),
            "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"]),
        })
    expected = {(dataset, HORIZON, seed, model) for dataset in DATASETS for seed in SEEDS for model in MODELS}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if len(rows) != 108 or found != expected:
        raise ValueError("coordinate coverage mismatch")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = rows_for_phase(args.phase_root)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    summaries = []
    comparisons = []
    for dataset in DATASETS:
        means = {model: statistics.fmean(row["best_val_mse"] for row in grouped[(dataset, model)]) for model in MODELS}
        winner = min(means, key=means.get)
        for model in MODELS:
            model_rows = {row["seed"]: row for row in grouped[(dataset, model)]}
            summaries.append({
                "dataset": dataset, "horizon": HORIZON, "model": model,
                "best_val_mse_mean": means[model],
                "best_model": winner,
                "elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in model_rows.values()),
                "epochs_ran_mean": statistics.fmean(row["epochs_ran"] for row in model_rows.values()),
            })
        for candidate in ("MambaTM1AutoCon", "MambaTM2AutoCon", "MambaTSv3AutoCon"):
            candidate_rows = {row["seed"]: row for row in grouped[(dataset, candidate)]}
            for baseline in MODELS:
                baseline_rows = {row["seed"]: row for row in grouped[(dataset, baseline)]}
                deltas = [100.0 * (candidate_rows[seed]["best_val_mse"] - baseline_rows[seed]["best_val_mse"]) / baseline_rows[seed]["best_val_mse"] for seed in SEEDS]
                comparisons.append({
                    "dataset": dataset, "horizon": HORIZON, "candidate": candidate,
                    "baseline": baseline, "candidate_mse_mean": means[candidate],
                    "baseline_mse_mean": means[baseline],
                    "mean_mse_change_pct": statistics.fmean(deltas),
                    "candidate_wins": sum(delta < 0.0 for delta in deltas), "n": len(SEEDS),
                })
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in (("validation_rows.csv", rows), ("model_summary.csv", summaries), ("paired_comparisons.csv", comparisons)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = [
        "# TimeMachine-Style Mamba H96 Baseline Extension",
        "",
        "Validation-only, three paired seeds and no test metric artifacts. The best model is descriptive within this fixed screen, not a model-selection result.",
        "",
        "| Dataset | Model | Mean validation MSE | Best model | Mean job wall time |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for item in summaries:
        lines.append("| {dataset} | {model} | {best_val_mse_mean:.6f} | {best_model} | {elapsed_sec_mean:.2f}s |".format(**item))
    lines.extend(["", "For all Mamba candidate-to-baseline paired contrasts, see `paired_comparisons.csv`. No test claim, cross-dataset non-inferiority claim, or hardware-independent speed claim follows from this screen."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "summary_rows": len(summaries), "comparison_rows": len(comparisons), "test_metrics_produced": False})


if __name__ == "__main__":
    main()
