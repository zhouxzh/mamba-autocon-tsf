#!/usr/bin/env python3
"""Strict validation-only readout for the STGCN traffic baseline expansion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SEEDS = (3071, 3072, 3073)
PHASES = {
    "candidate_h12_validation": ("candidate", 6, "MambaTSv3AutoCon"),
    "stgcn_h12_validation": ("stgcn", 6, "STGCN"),
}
GRAPH_HASHES = {
    "metr_la": "97cfc5c63ab90bc6517108881f1aa3c26c62fbc548584b4ec76b28b0e16eac89",
    "pems_bay": "3567957c59f994a678526725b810cf29031affe03dca99c72d0c4eba9e8761c7",
}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_phase(root: Path, phase: str, expected: int, model: str) -> tuple[str, list[dict]]:
    validation = _load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected:
        raise ValueError("invalid phase validation: {}".format(phase))
    provenance = _load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError("missing strict provenance: {}".format(phase))
    manifest = root / "logs" / "job_manifest.jsonl"
    rows, seen = [], set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid job: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source drift: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        if any((result / artifact).exists() for artifact in TEST_ARTIFACTS):
            raise ValueError("test artifact present: {}".format(job_id))
        config = _load(result / "config.json")
        if config.get("model") != model or config.get("skip_test") is not True:
            raise ValueError("model/test contract mismatch: {}".format(job_id))
        if int(config.get("pred_len", -1)) != 12 or int(config.get("seed", -1)) not in SEEDS:
            raise ValueError("shape/seed contract mismatch: {}".format(job_id))
        if model == "MambaTSv3AutoCon":
            if (
                config.get("AutoCon") is not True
                or config.get("mamba_ts_variant") != "v3"
                or config.get("mamba_ts_repr_forecast_fusion") is not True
                or float(config.get("AutoCon_lambda", 0.0)) != 0.001
            ):
                raise ValueError("candidate contract mismatch: {}".format(job_id))
        else:
            data = str(config.get("data"))
            if config.get("AutoCon") is not False or config.get("mamba_ts_graph_spatial") is not True:
                raise ValueError("STGCN graph contract missing: {}".format(job_id))
            if str(config.get("mamba_ts_graph_expected_sha256", "")) != GRAPH_HASHES.get(data):
                raise ValueError("STGCN graph hash mismatch: {}".format(job_id))
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in _load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("invalid validation loss: {}".format(job_id))
        rows.append({
            "phase": phase,
            "dataset": str(config["data"]),
            "horizon": int(config["pred_len"]),
            "seed": int(config["seed"]),
            "model": model,
            "best_val_mse": min(losses),
            "elapsed_sec": float(record["elapsed_sec"]),
        })
    if len(rows) != expected:
        raise ValueError("coverage mismatch: {}".format(phase))
    return str(source), rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, sources = [], set()
    for phase, (directory, expected, model) in PHASES.items():
        source, current = _read_phase(args.artifact_root / directory, phase, expected, model)
        sources.add(source)
        rows.extend(current)
    if len(sources) != 1:
        raise ValueError("phases do not share a source snapshot")
    index = {(row["phase"], row["dataset"], row["seed"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate validation coordinate")
    summary = []
    for dataset in sorted({row["dataset"] for row in rows}):
        candidate = [index[("candidate_h12_validation", dataset, seed)] for seed in SEEDS]
        stgcn = [index[("stgcn_h12_validation", dataset, seed)] for seed in SEEDS]
        ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, stgcn)]
        summary.append({
            "dataset": dataset,
            "horizon": 12,
            "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate),
            "stgcn_mse_mean": statistics.fmean(row["best_val_mse"] for row in stgcn),
            "candidate_vs_stgcn_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
            "candidate_wins": sum(ratio < 1.0 for ratio in ratios),
            "n": len(ratios),
            "candidate_elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in candidate),
            "stgcn_elapsed_sec_mean": statistics.fmean(row["elapsed_sec"] for row in stgcn),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_rows.csv", rows)
    _write_csv(args.output_dir / "paired_summary.csv", summary)
    lines = [
        "# METR-LA / PEMS-BAY STGCN Baseline Expansion",
        "",
        "Validation-only fresh-seed baseline expansion. No held-out test metrics or artifacts are included.",
        "",
        "| Dataset | H | Candidate MSE | STGCN MSE | Candidate vs STGCN | Wins | Candidate job sec | STGCN job sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {horizon} | {candidate_mse_mean:.6f} | {stgcn_mse_mean:.6f} | "
            "{candidate_vs_stgcn_pct:+.3f}% | {candidate_wins}/{n} | "
            "{candidate_elapsed_sec_mean:.1f} | {stgcn_elapsed_sec_mean:.1f} |".format(**row)
        )
    lines.extend([
        "",
        "Negative candidate-vs-STGCN change favors the frozen MambaTSv3AutoCon candidate. "
        "Elapsed seconds are full validation job times and are not an inference-latency claim.",
        "",
        "This comparison only closes a graph-temporal baseline gap. A new held-out release would require a separately registered protocol.",
    ])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({
        "source_tree_sha256": next(iter(sources)), "rows": len(rows), "paired_cells": len(summary),
        "test_metrics_produced": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_cells": len(summary), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
