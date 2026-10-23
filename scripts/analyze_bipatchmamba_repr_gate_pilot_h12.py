#!/usr/bin/env python3
"""Strict reader for the BiPatchMamba bounded representation-gate pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "electricity", "traffic", "weather")
SEEDS = (2411, 2412)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(root: Path, directory: str) -> list[dict]:
    path = root / directory / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate(record: dict) -> tuple[Path, dict, float]:
    if record.get("status") != "ok" or record.get("returncode") != 0:
        raise ValueError(f"failed job: {record.get('job_id')}")
    if not record.get("source_tree_sha256_before") or record["source_tree_sha256_before"] != record.get("source_tree_sha256_after"):
        raise ValueError(f"source drift: {record.get('job_id')}")
    result_dir = Path(record.get("expected_artifact", "")).parent
    if not (result_dir / "training_complete.json").is_file():
        raise ValueError(f"missing completion marker: {result_dir}")
    if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
        raise ValueError(f"test artifact present: {result_dir}")
    command = record.get("command", [])
    if "--checkpoints" not in command:
        raise ValueError(f"checkpoint option missing: {record.get('job_id')}")
    loss_path = Path(command[command.index("--checkpoints") + 1]) / result_dir.name / "loss_logs.json"
    if not loss_path.is_file():
        raise ValueError(f"missing loss log: {loss_path}")
    values = [float(value) for value in _json(loss_path).get("val_loss", [])]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid validation loss: {loss_path}")
    return result_dir, _json(result_dir / "config.json"), min(values)


def _read_phase(root: Path, directory: str, phase: str, gate: bool, autocon: bool) -> tuple[list[dict], set[str]]:
    expected = {(dataset, "BiPatchMamba" if not autocon else "BiPatchMambaAutoCon", seed) for dataset in DATASETS for seed in SEEDS}
    seen, rows, hashes = set(), [], set()
    for record in _read_manifest(root, directory):
        coordinate = (str(record.get("dataset")), str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError(f"invalid or duplicate coordinate: {coordinate}")
        result_dir, config, loss = _validate(record)
        required = {"data": coordinate[0], "model": coordinate[1], "features": "M", "seq_len": 96,
                    "pred_len": 12, "enc_in": {"ETTh1": 7, "electricity": 321, "traffic": 862, "weather": 21}[coordinate[0]],
                    "c_out": {"ETTh1": 7, "electricity": 321, "traffic": 862, "weather": 21}[coordinate[0]],
                    "seed": coordinate[2], "skip_test": True, "d_model": 24, "d_ff": 48,
                    "patch_len": 24, "patch_stride": 24}
        mismatches = [key for key, value in required.items() if config.get(key) != value]
        if mismatches:
            raise ValueError(f"common config mismatch {mismatches}: {result_dir}")
        if bool(config.get("AutoCon")) != autocon or bool(config.get("mamba_bipatch_repr_gate")) != gate:
            raise ValueError(f"gate/AutoCon config mismatch: {result_dir}")
        if autocon and config.get("AutoCon_wnorm") != "MeanStd":
            raise ValueError(f"AutoCon normalization mismatch: {result_dir}")
        if autocon:
            expected_lambda = 0.0 if phase == "null_gate" else 0.001
            if config.get("AutoCon_lambda") != expected_lambda:
                raise ValueError(f"lambda mismatch: {result_dir}")
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        rows.append({"phase": phase, "dataset": coordinate[0], "model": coordinate[1], "seed": coordinate[2],
                     "best_val_mse": loss, "best_epoch": record.get("best_epoch"), "epochs_ran": record.get("epochs_ran"),
                     "elapsed_sec": record.get("elapsed_sec")})
    if seen != expected:
        raise ValueError(f"incomplete coverage for {phase}: expected {len(expected)} got {len(seen)}")
    return rows, hashes


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    plain, plain_hashes = _read_phase(root, "plain", "plain", gate=False, autocon=False)
    null, null_hashes = _read_phase(root, "null_gate", "null_gate", gate=True, autocon=True)
    active, active_hashes = _read_phase(root, "active_gate", "active_gate", gate=True, autocon=True)
    hashes = plain_hashes | null_hashes | active_hashes
    if len(hashes) != 1:
        raise ValueError(f"source snapshot differs across phases: {hashes}")
    rows = plain + null + active
    summary = []
    for dataset in DATASETS:
        get = lambda phase: [row["best_val_mse"] for row in rows if row["phase"] == phase and row["dataset"] == dataset]
        p, n, a = get("plain"), get("null_gate"), get("active_gate")
        am, nm, pm = statistics.fmean(a), statistics.fmean(n), statistics.fmean(p)
        summary.append({"dataset": dataset, "horizon": 12, "active_gate_mean_val_mse": am,
                        "null_gate_mean_val_mse": nm, "plain_mean_val_mse": pm,
                        "active_gate_vs_null_percent": 100.0 * (am / nm - 1.0),
                        "active_gate_vs_plain_percent": 100.0 * (am / pm - 1.0),
                        "active_gate_wins_vs_null": sum(x < y for x, y in zip(a, n)),
                        "active_gate_wins_vs_plain": sum(x < y for x, y in zip(a, p)),
                        "passes_autocon_noninferiority": am <= nm * 1.01,
                        "passes_autocon_improvement": am < nm and sum(x < y for x, y in zip(a, n)) >= 2,
                        "source_snapshot": next(iter(hashes)), "test_metrics_produced": False})
    pilot_pass = all(row["passes_autocon_noninferiority"] for row in summary)
    for row in summary:
        row["passes_pilot"] = pilot_pass
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "summary.csv", summary)
    lines = ["# BiPatchMamba Representation-Gate H12 Pilot", "", "Validation-only. No held-out test metric was generated or read.", "", "| Dataset | Active gate | Null gate | Plain | Active vs null | wins/null | Non-inferiority |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        lines.append("| {dataset} | {active_gate_mean_val_mse:.6f} | {null_gate_mean_val_mse:.6f} | {plain_mean_val_mse:.6f} | {active_gate_vs_null_percent:+.3f}% | {active_gate_wins_vs_null}/2 | {passes_autocon_noninferiority} |".format(**row))
    lines.extend(["", f"Source snapshot: `{next(iter(hashes))}`."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passes_pilot": pilot_pass, "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
