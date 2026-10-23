#!/usr/bin/env python3
"""Strict validation-only reader for the Solar MambaTSv3 baseline matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASET = "solar"
CANDIDATE = "MambaTSv3AutoCon"
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")
SEEDS = (3271, 3272, 3273)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, expected_models: tuple[str, ...], expected_jobs: int) -> tuple[list[dict], str]:
    validation = load_json(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load_json(root / "provenance.json")
    source_hash = provenance.get("source_tree_sha256")
    if not source_hash:
        raise ValueError(f"missing source hash: {root}")

    rows: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        coordinate = (int(record.get("seed")), str(record.get("model")))
        if coordinate in seen or coordinate[0] not in SEEDS or coordinate[1] not in expected_models:
            raise ValueError(f"unexpected/duplicate coordinate: {record.get('job_id')}")
        if record.get("status") != "ok" or record.get("returncode") != 0 or record.get("dataset") != DATASET:
            raise ValueError(f"failed or invalid record: {record.get('job_id')}")
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError(f"source drift: {record.get('job_id')}")
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file() or any((result_dir / item).exists() for item in TEST_ARTIFACTS):
            raise ValueError(f"test artifact or incomplete run: {result_dir}")
        config = load_json(result_dir / "config.json")
        if (config.get("model"), int(config.get("seed")), config.get("data"), int(config.get("pred_len"))) != (
            coordinate[1], coordinate[0], DATASET, 96
        ) or config.get("skip_test") is not True:
            raise ValueError(f"config boundary mismatch: {result_dir}")
        if coordinate[1] == CANDIDATE:
            if not (config.get("AutoCon") is True and config.get("mamba_ts_variant") == "v3" and config.get("mamba_ts_repr_forecast_fusion") is True):
                raise ValueError("candidate mechanism contract mismatch")
        elif config.get("AutoCon") is True or config.get("mamba_ts_repr_forecast_fusion") is True:
            raise ValueError("baseline received candidate-only mechanism")
        command = record["command"]
        checkpoint_root = Path(command[command.index("--checkpoints") + 1]) / result_dir.name
        losses = [float(value) for value in load_json(checkpoint_root / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"missing/non-finite validation loss: {result_dir}")
        seen.add(coordinate)
        rows.append({
            "dataset": DATASET, "horizon": 96, "seed": coordinate[0], "model": coordinate[1],
            "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]),
            "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"]),
        })
    expected = {(seed, model) for seed in SEEDS for model in expected_models}
    if seen != expected or len(rows) != expected_jobs:
        raise ValueError(f"coverage mismatch: {root}")
    return rows, str(source_hash)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    candidate_rows, candidate_hash = read_phase(root / "candidate", (CANDIDATE,), 3)
    baseline_rows, baseline_hash = read_phase(root / "baselines", BASELINES, 15)
    if candidate_hash != baseline_hash:
        raise ValueError("candidate and baseline source snapshots differ")
    rows = candidate_rows + baseline_rows
    means = {}
    for model in (CANDIDATE,) + BASELINES:
        subset = [row for row in rows if row["model"] == model]
        means[model] = {
            "mean_val_mse": statistics.fmean(row["best_val_mse"] for row in subset),
            "std_val_mse": statistics.stdev(row["best_val_mse"] for row in subset),
            "mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in subset),
            "n": len(subset),
        }
    candidate_by_seed = {row["seed"]: row["best_val_mse"] for row in candidate_rows}
    paired = []
    for baseline in BASELINES:
        baseline_by_seed = {row["seed"]: row["best_val_mse"] for row in baseline_rows if row["model"] == baseline}
        deltas = [candidate_by_seed[seed] - baseline_by_seed[seed] for seed in SEEDS]
        paired.append({
            "dataset": DATASET, "horizon": 96, "baseline": baseline,
            "candidate_mean_val_mse": means[CANDIDATE]["mean_val_mse"],
            "baseline_mean_val_mse": means[baseline]["mean_val_mse"],
            "mean_delta_mse": statistics.fmean(deltas),
            "relative_delta_pct": 100.0 * statistics.fmean(deltas) / means[baseline]["mean_val_mse"],
            "candidate_wins": sum(delta < 0.0 for delta in deltas), "n": len(deltas),
        })
    summary = [
        {"dataset": DATASET, "horizon": 96, "model": model, **means[model]}
        for model in (CANDIDATE,) + BASELINES
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "validation_rows.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_summary.csv", paired)
    lines = [
        "# Solar MambaTSv3 AutoCon Strong-Baseline Validation",
        "",
        "Validation-only paired screen. No held-out test metric, prediction, or test artifact was read or produced.",
        "",
        "| Model | Mean validation MSE | SD | Mean job sec | Seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append("| {model} | {mean_val_mse:.6f} | {std_val_mse:.6f} | {mean_elapsed_sec:.3f} | {n} |".format(**row))
    lines.extend(["", "## Paired Candidate Deltas", "", "Negative delta favors MambaTSv3AutoCon.", "", "| Baseline | Candidate MSE | Baseline MSE | Delta | Relative delta | Candidate wins |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in paired:
        lines.append("| {baseline} | {candidate_mean_val_mse:.6f} | {baseline_mean_val_mse:.6f} | {mean_delta_mse:.6f} | {relative_delta_pct:.2f}% | {candidate_wins}/{n} |".format(**row))
    lines.extend(["", f"All rows share source snapshot `{candidate_hash}`. This is a Solar H96 boundary result, not a held-out or causal claim."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_rows": len(paired), "source_tree_sha256": candidate_hash, "test_metrics_produced": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
