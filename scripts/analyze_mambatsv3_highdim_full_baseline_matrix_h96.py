#!/usr/bin/env python3
"""Join only source-identical validation-only Electricity/Traffic H96 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3291, 3292, 3293)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")
EXPECTED = {
    "candidate": ("candidate_h96_validation", 6, {"MambaTSv3AutoCon"}),
    "linear": ("linear_baselines_h96_validation", 12, {"DLinear", "NLinear"}),
    "remaining": ("remaining_baselines_h96_validation", 30, {"RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2"}),
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(root: Path, label: str) -> tuple[str, list[dict]]:
    phase, expected_count, models = EXPECTED[label]
    validation = _load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected_count:
        raise ValueError("invalid phase validation: {}".format(label))
    provenance = _load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError("invalid provenance: {}".format(label))
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source drift: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("test boundary violation: {}".format(job_id))
        config = _load(result / "config.json")
        dataset, model, seed = str(config.get("data")), str(config.get("model")), int(config.get("seed", -1))
        if dataset not in DATASETS or model not in models or seed not in SEEDS:
            raise ValueError("coordinate/model mismatch: {}".format(job_id))
        if int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True:
            raise ValueError("validation-only mismatch: {}".format(job_id))
        if model == "MambaTSv3AutoCon":
            if config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not True or float(config.get("AutoCon_lambda", 0.0)) != 0.001:
                raise ValueError("candidate contract mismatch: {}".format(job_id))
        elif config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError("baseline isolation mismatch: {}".format(job_id))
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in _load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("invalid loss series: {}".format(job_id))
        rows.append({"dataset": dataset, "horizon": 96, "seed": seed, "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected_count:
        raise ValueError("coverage mismatch: {}".format(label))
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--linear-root", type=Path, required=True)
    parser.add_argument("--remaining-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources, rows = set(), []
    for label, root in (("candidate", args.candidate_root), ("linear", args.linear_root), ("remaining", args.remaining_root)):
        source, current = _read(root, label)
        sources.add(source)
        rows.extend(current)
    if len(sources) != 1:
        raise ValueError("cannot join artifacts from different source snapshots")
    index = {(row["dataset"], row["model"], row["seed"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate coordinate")
    models = ("DLinear", "NLinear", "RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")
    expected_coordinates = {(dataset, model, seed) for dataset in DATASETS for model in ("MambaTSv3AutoCon",) + models for seed in SEEDS}
    if set(index) != expected_coordinates:
        raise ValueError("incomplete full baseline matrix")
    summary = []
    for dataset in DATASETS:
        candidate = [index[(dataset, "MambaTSv3AutoCon", seed)] for seed in SEEDS]
        for baseline in models:
            comparator = [index[(dataset, baseline, seed)] for seed in SEEDS]
            ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, comparator)]
            summary.append({
                "dataset": dataset, "horizon": 96, "baseline": baseline,
                "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate),
                "baseline_mse_mean": statistics.fmean(row["best_val_mse"] for row in comparator),
                "candidate_vs_baseline_pct": 100 * (statistics.fmean(ratios) - 1.0),
                "candidate_wins": sum(value < 1.0 for value in ratios), "n": len(ratios),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_rows.csv", rows)
    _write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = [
        "# Electricity / Traffic H96 Full Same-Source Baseline Matrix", "",
        "Validation-only, three fresh paired seeds, one immutable source snapshot. The reader verified full 48-row coverage and rejected test artifacts before calculating this table.", "",
        "| Dataset | H | Baseline | Candidate MSE | Baseline MSE | Candidate vs baseline | Wins |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append("| {dataset} | {horizon} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {candidate_vs_baseline_pct:+.3f}% | {candidate_wins}/{n} |".format(**row))
    lines.extend(["", "Negative change favors MambaTSv3AutoCon. This table is validation evidence only; it is not a held-out result and does not demonstrate an independent relation-loss effect or universal speed superiority."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
