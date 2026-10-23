#!/usr/bin/env python3
"""Read the validated held-out MambaTSv3 representation-fusion release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats


SELECTION_PHASE = "selection_fusion_candidate_validation"
CANDIDATE_PHASE = "confirmatory_fusion_candidate_test"
BASELINE_PHASE = "confirmatory_same_source_baselines_test"
CANDIDATE_MODEL = "MambaTSv3AutoCon"
DATASETS = ("metr_la", "pems_bay")
SEEDS = (2921, 2922, 2923, 2924, 2925)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase_rows(root: Path, phase: str, expected: int, required_model: str | None,
                test_producing: bool, plan_sha256: str) -> list[dict]:
    validation = _load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = _load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError(f"missing source or enforced launch binding: {root}")
    if launch.get("plan_sha256") != plan_sha256:
        raise ValueError(f"plan hash mismatch: {root}")
    rows, seen = [], set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = record.get("job_id")
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest job: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source mismatch: {job_id}")
        result = Path(record["expected_artifact"]).parent
        config = _load(result / "config.json")
        model = str(config.get("model"))
        if required_model and model != required_model:
            raise ValueError(f"model mismatch: {job_id}")
        row = {"phase": phase, "dataset": str(config["data"]), "model": model,
               "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
               "elapsed_sec": float(record["elapsed_sec"]), "epochs_ran": int(record["epochs_ran"])}
        if test_producing:
            stream = _load(result / "stream_metrics.json")
            if config.get("skip_test") is True or config.get("stream_metrics") is not True:
                raise ValueError(f"test contract mismatch: {job_id}")
            if stream.get("aggregation") != "streaming_window_weighted":
                raise ValueError(f"stream aggregation mismatch: {job_id}")
            scaled, original = stream["scaled"], stream["original"]
            row.update({"scaled_mse": float(scaled["mse"]), "scaled_mae": float(scaled["mae"]),
                        "original_mse": float(original["mse"]), "original_mae": float(original["mae"])})
        elif config.get("skip_test") is not True or not (result / "training_complete.json").is_file():
            raise ValueError(f"selection contract mismatch: {job_id}")
        rows.append(row)
    if len(rows) != expected:
        raise ValueError(f"manifest coverage mismatch: expected {expected}, got {len(rows)}")
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
    plan_sha256 = _sha256(args.plan)
    phases = {phase["name"]: phase for phase in plan["phases"]}
    counts = plan["registered_job_counts"]
    root = args.artifact_root.resolve()
    selection = _phase_rows(root / "selection", SELECTION_PHASE, int(counts[SELECTION_PHASE]), CANDIDATE_MODEL, False, plan_sha256)
    candidate = _phase_rows(root / "candidate", CANDIDATE_PHASE, int(counts[CANDIDATE_PHASE]), CANDIDATE_MODEL, True, plan_sha256)
    baselines = _phase_rows(root / "baselines", BASELINE_PHASE, int(counts[BASELINE_PHASE]), None, True, plan_sha256)
    rows = candidate + baselines
    index = {(row["dataset"], row["model"], row["horizon"], row["seed"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate test coordinate")
    baseline_models = tuple(phases[BASELINE_PHASE]["models"])
    summary = []
    for dataset in DATASETS:
        for baseline in baseline_models:
            crows = [index[(dataset, CANDIDATE_MODEL, 12, seed)] for seed in SEEDS]
            brows = [index[(dataset, baseline, 12, seed)] for seed in SEEDS]
            ratios = [left["scaled_mse"] / right["scaled_mse"] for left, right in zip(crows, brows)]
            logs = [math.log(value) for value in ratios]
            upper = math.exp(statistics.fmean(logs) + stats.t.ppf(0.95, len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs)))
            summary.append({"dataset": dataset, "baseline": baseline,
                            "candidate_mse_mean": statistics.fmean(row["scaled_mse"] for row in crows),
                            "baseline_mse_mean": statistics.fmean(row["scaled_mse"] for row in brows),
                            "mean_mse_change_pct": 100 * (statistics.fmean(ratios) - 1),
                            "candidate_wins": sum(value < 1 for value in ratios), "n": len(ratios),
                            "upper_ratio_95_one_sided": upper, "noninferior_ratio_1p02": upper <= 1.02,
                            "candidate_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in crows),
                            "baseline_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in brows)})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "selection_rows.csv", selection)
    _write_csv(out / "test_rows.csv", rows)
    _write_csv(out / "paired_test_summary.csv", summary)
    lines = ["# Held-out METR-LA / PEMS-BAY MambaTSv3 Representation Fusion", "", "Five fresh paired seeds; one-sided 95% upper MSE-ratio bounds use the registered 1.02 non-inferiority threshold.", "", "| Dataset | Baseline | Candidate MSE | Baseline MSE | Change | Wins | Upper ratio | Non-inferior |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in summary:
        lines.append("| {dataset} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {mean_mse_change_pct:+.3f}% | {candidate_wins}/{n} | {upper_ratio_95_one_sided:.4f} | {noninferior_ratio_1p02} |".format(**item))
    lines.extend(["", "This release is restricted to non-graph METR-LA/PEMS-BAY H12. It does not establish universal forecasting or hardware-independent speed claims."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "readout_metadata.json").write_text(json.dumps({"plan_sha256": plan_sha256, "selection_rows": len(selection), "test_rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection_rows": len(selection), "test_rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
