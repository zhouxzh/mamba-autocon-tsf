#!/usr/bin/env python3
"""Read the frozen PEMS d24 MambaTS v3 test release without model selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats


SELECTION_PHASE = "selection_pems_d24_candidate_validation"
CANDIDATE_PHASE = "confirmatory_pems_d24_candidate_test"
BASELINE_PHASE = "confirmatory_pems_strong_baselines_test"
CANDIDATE_MODEL = "MambaTSv3AutoCon"
DATASETS = ("pems03", "pems04", "pems07", "pems08")
SEEDS = (2131, 2132, 2133, 2134, 2135)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _phase_rows(
    root: Path,
    phase: str,
    expected: int,
    required_model: str | None,
    test_producing: bool,
) -> list[dict]:
    validation = _load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError("invalid phase validation: {}".format(root))
    provenance = _load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError("missing source or enforced launch binding: {}".format(root))
    rows, seen = [], set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest job in {}".format(root))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source mismatch: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        config = _load(result / "config.json")
        model = str(config.get("model"))
        if required_model and model != required_model:
            raise ValueError("model mismatch: {}".format(job_id))
        row = {
            "phase": phase, "dataset": str(config["data"]), "model": model,
            "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"]),
        }
        if test_producing:
            stream = _load(result / "stream_metrics.json")
            if config.get("skip_test") is True or stream.get("aggregation") != "streaming_window_weighted":
                raise ValueError("test contract mismatch: {}".format(job_id))
            mse, mae = float(stream["scaled"]["mse"]), float(stream["scaled"]["mae"])
            if not (math.isfinite(mse) and mse > 0 and math.isfinite(mae) and mae >= 0):
                raise ValueError("invalid streaming metric: {}".format(job_id))
            row.update({
                "scaled_mse": mse, "scaled_mae": mae,
                "original_mse": float(stream["original"]["mse"]),
                "original_mae": float(stream["original"]["mae"]),
            })
        elif config.get("skip_test") is not True or not (result / "training_complete.json").is_file():
            raise ValueError("selection contract mismatch: {}".format(job_id))
        rows.append(row)
    if len(rows) != expected:
        raise ValueError("manifest coverage mismatch: {} expected {}, got {}".format(root, expected, len(rows)))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.plan)
    if plan.get("status") != "launch_authorized":
        raise ValueError("plan is not launch authorized")
    phases = {phase["name"]: phase for phase in plan["phases"]}
    counts = plan["registered_job_counts"]
    required = (SELECTION_PHASE, CANDIDATE_PHASE, BASELINE_PHASE)
    if any(name not in phases or name not in counts for name in required):
        raise ValueError("plan does not contain the frozen PEMS release phases")
    root = args.artifact_root.resolve()
    selection = _phase_rows(root / "selection", SELECTION_PHASE, int(counts[SELECTION_PHASE]), CANDIDATE_MODEL, False)
    if any(row["horizon"] != 12 for row in selection):
        raise ValueError("selection horizon mismatch")
    candidate = _phase_rows(root / "candidate", CANDIDATE_PHASE, int(counts[CANDIDATE_PHASE]), CANDIDATE_MODEL, True)
    baselines = _phase_rows(root / "baselines", BASELINE_PHASE, int(counts[BASELINE_PHASE]), None, True)
    rows = candidate + baselines
    index = {(row["dataset"], row["model"], row["horizon"], row["seed"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate test coordinate")
    baseline_models = tuple(phases[BASELINE_PHASE]["models"])
    summary = []
    for dataset in DATASETS:
        for baseline in baseline_models:
            candidate_rows = [index[(dataset, CANDIDATE_MODEL, 12, seed)] for seed in SEEDS]
            baseline_rows = [index[(dataset, baseline, 12, seed)] for seed in SEEDS]
            ratios = [left["scaled_mse"] / right["scaled_mse"] for left, right in zip(candidate_rows, baseline_rows)]
            log_ratios = [math.log(value) for value in ratios]
            upper_ratio = math.exp(
                statistics.fmean(log_ratios)
                + stats.t.ppf(0.95, len(log_ratios) - 1)
                * statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
            )
            summary.append({
                "dataset": dataset, "baseline": baseline, "candidate_mse_mean": statistics.fmean(row["scaled_mse"] for row in candidate_rows),
                "baseline_mse_mean": statistics.fmean(row["scaled_mse"] for row in baseline_rows),
                "mean_mse_change_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
                "candidate_wins": sum(value < 1.0 for value in ratios), "n": len(ratios),
                "upper_ratio_95_one_sided": upper_ratio, "noninferior_ratio_1p02": upper_ratio <= 1.02,
                "candidate_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in candidate_rows),
                "baseline_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in baseline_rows),
            })
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "selection_rows.csv", selection)
    _write_csv(out / "test_rows.csv", rows)
    _write_csv(out / "paired_test_summary.csv", summary)
    lines = [
        "# Frozen PEMS d24 MambaTS v3 Test Release", "",
        "Five fresh paired seeds; one-sided 95% upper MSE-ratio bounds use the registered 1.02 non-inferiority threshold.", "",
        "| Dataset | Baseline | Candidate MSE | Baseline MSE | Change | Wins | Upper ratio | Non-inferior |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary:
        lines.append("| {dataset} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {mean_mse_change_pct:+.3f}% | {candidate_wins}/{n} | {upper_ratio_95_one_sided:.4f} | {noninferior_ratio_1p02} |".format(**item))
    lines.extend(["", "Wall time is a full-job implementation measurement, not hardware-independent throughput. DLinear and NLinear remain descriptive speed boundaries; no universal efficiency claim follows from this PEMS-H12-only release."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"selection_rows": len(selection), "test_rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
