#!/usr/bin/env python3
"""Strict reader for the validation-only RLinear baseline extension."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


PHASES = {
    "rlinear_core_h96_validation": ("core_h96", 15, {"ETTh1": 96, "weather": 96, "exchange_rate": 96, "electricity": 96, "traffic": 96}),
    "rlinear_ili_h24_validation": ("ili_h24", 3, {"national_illness": 24}),
}
SEEDS = (3131, 3132, 3133)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _phase_rows(root: Path, expected_jobs: int, coordinates: dict[str, int]) -> list[dict]:
    validation = _load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_jobs:
        raise ValueError("phase validation failed: {}".format(root))
    source_hash = _load(root / "provenance.json").get("source_tree_sha256")
    records = {}
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("job_id") in records or record.get("status") != "ok":
            raise ValueError("invalid manifest record")
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError("source snapshot mismatch")
        records[record["job_id"]] = record
    if len(records) != expected_jobs:
        raise ValueError("manifest coverage mismatch")

    rows = []
    for record in records.values():
        result_dir = Path(record["expected_artifact"]).parent
        config = _load(result_dir / "config.json")
        if config.get("model") != "RLinear" or config.get("skip_test") is not True:
            raise ValueError("RLinear or test-boundary contract mismatch")
        if any((result_dir / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact found in validation-only phase")
        command = record["command"]
        checkpoint_dir = Path(command[command.index("--checkpoints") + 1]) / result_dir.name
        values = [float(value) for value in _load(checkpoint_dir / "loss_logs.json").get("val_loss", [])]
        if not values:
            raise ValueError("validation trajectory missing")
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "best_val_mse": min(values), "elapsed_sec": float(record["elapsed_sec"]),
        })
    expected = {(dataset, horizon, seed) for dataset, horizon in coordinates.items() for seed in SEEDS}
    found = {(row["dataset"], row["horizon"], row["seed"]) for row in rows}
    if found != expected:
        raise ValueError("coordinate mismatch")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for phase, (output_dir, count, coordinates) in PHASES.items():
        rows.extend(_phase_rows(args.artifact_root / output_dir, count, coordinates))
    summaries = []
    for dataset, horizon in sorted({(row["dataset"], row["horizon"]) for row in rows}):
        matching = [row for row in rows if (row["dataset"], row["horizon"]) == (dataset, horizon)]
        summaries.append({
            "dataset": dataset, "horizon": horizon,
            "mean_val_mse": statistics.fmean(row["best_val_mse"] for row in matching),
            "std_val_mse": statistics.stdev(row["best_val_mse"] for row in matching),
            "mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in matching), "n": len(matching),
        })
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("validation_rows.csv", rows), ("summary.csv", summaries)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    lines = [
        "# RLinear Current Baseline Extension", "",
        "Validation-only RLinear references. No test metric or prediction artifact was read or produced.", "",
        "| Dataset | Horizon | Mean validation MSE | SD | Mean job sec | Seeds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append("| {dataset} | {horizon} | {mean_val_mse:.6f} | {std_val_mse:.6f} | {mean_elapsed_sec:.3f} | {n} |".format(**row))
    lines.extend(["", "These values are source-snapshot-bound RLinear references and must not be treated as paired deltas against historical candidate results."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "summary_rows": len(summaries), "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
