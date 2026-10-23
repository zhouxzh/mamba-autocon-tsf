#!/usr/bin/env python3
"""Strict reader for the light MambaTS AutoCon-on/off stride-24 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "weather", "electricity", "traffic")
SEEDS = (3391, 3392, 3393)
PHASES = {"autocon": ("mambats_autocon_h96_validation", 12, "MambaTSAutoCon"), "plain": ("mambats_plain_h96_validation", 12, "MambaTS")}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_phase(root: Path, label: str) -> tuple[str, list[dict]]:
    phase, expected, model = PHASES[label]
    validation, provenance = load(root / "phase_validation.json"), load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected or not source:
        raise ValueError(f"invalid phase: {label}")
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if not (result / "training_complete.json").is_file() or any((result / artifact).exists() for artifact in TEST_ARTIFACTS):
            raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json")
        dataset, seed = str(config.get("data")), int(config.get("seed", -1))
        if dataset not in DATASETS or seed not in SEEDS or config.get("model") != model or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True or int(config.get("patch_len", -1)) != 24 or int(config.get("patch_stride", -1)) != 24 or config.get("mamba_ts_variant") != "v1" or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError(f"coordinate or backbone mismatch: {job_id}")
        if label == "autocon":
            if config.get("AutoCon") is not True or float(config.get("AutoCon_lambda", 0.0)) != 0.001:
                raise ValueError(f"AutoCon candidate contract mismatch: {job_id}")
        elif config.get("AutoCon") is not False:
            raise ValueError(f"plain control contract mismatch: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation loss: {job_id}")
        rows.append({"dataset": dataset, "seed": seed, "model": model, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"])})
    if len(rows) != expected:
        raise ValueError(f"coverage mismatch: {label}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autocon-root", type=Path, required=True)
    parser.add_argument("--plain-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_a, autocon = read_phase(args.autocon_root, "autocon")
    source_b, plain = read_phase(args.plain_root, "plain")
    if source_a != source_b:
        raise ValueError("source mismatch")
    index = {(row["dataset"], row["model"], row["seed"]): row for row in autocon + plain}
    expected = {(dataset, model, seed) for dataset in DATASETS for model in ("MambaTSAutoCon", "MambaTS") for seed in SEEDS}
    if set(index) != expected or len(index) != 24:
        raise ValueError("incomplete matrix")
    summary = []
    for dataset in DATASETS:
        candidate = [index[(dataset, "MambaTSAutoCon", seed)] for seed in SEEDS]
        baseline = [index[(dataset, "MambaTS", seed)] for seed in SEEDS]
        ratios = [left["best_val_mse"] / right["best_val_mse"] for left, right in zip(candidate, baseline)]
        summary.append({"dataset": dataset, "horizon": 96, "autocon_mse_mean": statistics.fmean(row["best_val_mse"] for row in candidate), "plain_mse_mean": statistics.fmean(row["best_val_mse"] for row in baseline), "autocon_vs_plain_pct": 100 * (statistics.fmean(ratios) - 1), "autocon_wins": sum(value < 1 for value in ratios), "n": 3})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("validation_rows.csv", autocon + plain), ("paired_summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    lines = ["# Lightweight MambaTS AutoCon-On versus Plain H96", "", "Validation-only, four datasets, three paired fresh seeds, 24/24 successful jobs, one immutable source snapshot. Negative change favors the train-time AutoCon candidate.", "", "| Dataset | AutoCon MSE | Plain MambaTS MSE | AutoCon vs plain | Wins |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {autocon_mse_mean:.6f} | {plain_mse_mean:.6f} | {autocon_vs_plain_pct:+.3f}% | {autocon_wins}/{n} |".format(**row) for row in summary)
    lines.extend(["", "This is a local AutoCon-on/off validation comparison. It does not establish held-out accuracy or universal causal attribution."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": 24, "paired_cells": 4, "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": 24, "paired_cells": 4, "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
