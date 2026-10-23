#!/usr/bin/env python3
"""Strict reader for the current-source high-dimensional linear baseline extension."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3291, 3292, 3293)
PHASES = {
    "candidate_h96_validation": ("candidate", 6, {"MambaTSv3AutoCon"}),
    "linear_baselines_h96_validation": ("baselines", 12, {"DLinear", "NLinear"}),
}
TEST_ARTIFACTS = (
    "metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_phase(root: Path, phase: str, expected: int, models: set[str]) -> tuple[str, list[dict]]:
    validation = _load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError("invalid phase validation: {}".format(phase))
    provenance = _load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError("invalid provenance: {}".format(phase))
    rows, seen = [], set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid job: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source drift: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("test boundary violation: {}".format(job_id))
        config = _load(result / "config.json")
        model = str(config.get("model"))
        dataset, seed = str(config.get("data")), int(config.get("seed", -1))
        if model not in models or dataset not in DATASETS or seed not in SEEDS:
            raise ValueError("coordinate/model mismatch: {}".format(job_id))
        if int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True:
            raise ValueError("validation contract mismatch: {}".format(job_id))
        if model == "MambaTSv3AutoCon":
            if (
                config.get("AutoCon") is not True
                or config.get("mamba_ts_variant") != "v3"
                or config.get("mamba_ts_repr_forecast_fusion") is not True
                or float(config.get("AutoCon_lambda", 0.0)) != 0.001
            ):
                raise ValueError("candidate contract mismatch: {}".format(job_id))
        elif config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError("baseline isolation mismatch: {}".format(job_id))
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in _load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("invalid losses: {}".format(job_id))
        rows.append({
            "phase": phase, "dataset": dataset, "horizon": 96, "seed": seed, "model": model,
            "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]),
        })
    if len(rows) != expected:
        raise ValueError("coverage mismatch: {}".format(phase))
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    all_rows, sources = [], set()
    for phase, (directory, expected, models) in PHASES.items():
        source, rows = _read_phase(args.artifact_root / directory, phase, expected, models)
        sources.add(source)
        all_rows.extend(rows)
    if len(sources) != 1:
        raise ValueError("phases do not share one source snapshot")
    index = {(row["dataset"], row["model"], row["seed"]): row for row in all_rows}
    if len(index) != len(all_rows):
        raise ValueError("duplicate validation coordinate")
    summary = []
    for dataset in DATASETS:
        candidate = [index[(dataset, "MambaTSv3AutoCon", seed)] for seed in SEEDS]
        for baseline in ("DLinear", "NLinear"):
            comparator = [index[(dataset, baseline, seed)] for seed in SEEDS]
            ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, comparator)]
            summary.append({
                "dataset": dataset, "horizon": 96, "baseline": baseline,
                "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate),
                "baseline_mse_mean": statistics.fmean(row["best_val_mse"] for row in comparator),
                "candidate_vs_baseline_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
                "candidate_wins": sum(value < 1.0 for value in ratios), "n": len(ratios),
                "candidate_elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in candidate),
                "baseline_elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in comparator),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_rows.csv", all_rows)
    _write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = [
        "# Electricity / Traffic H96 DLinear-NLinear Extension", "",
        "Validation-only fresh-seed extension. No held-out metric, prediction, or streaming artifact was produced.", "",
        "| Dataset | H | Baseline | Candidate MSE | Baseline MSE | Candidate vs baseline | Wins | Candidate job sec | Baseline job sec |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {horizon} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | "
            "{candidate_vs_baseline_pct:+.3f}% | {candidate_wins}/{n} | {candidate_elapsed_sec_mean:.1f} | "
            "{baseline_elapsed_sec_mean:.1f} |".format(**row)
        )
    lines.extend([
        "",
        "Negative candidate-vs-baseline change favors MambaTSv3AutoCon. Job seconds are full validation job times, not an inference-speed measurement.",
        "This extension may complete a validation baseline matrix only; held-out release still requires a separate human-approved protocol.",
    ])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({
        "source_tree_sha256": next(iter(sources)), "rows": len(all_rows), "paired_cells": len(summary),
        "test_metrics_produced": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
