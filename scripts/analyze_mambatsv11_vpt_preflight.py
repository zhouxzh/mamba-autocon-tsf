#!/usr/bin/env python3
"""Strict reader for the frozen MambaTS v11 VPT ETTh1/H336 preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


PHASES = {
    "null": ("v11_autocon_null", "MambaTSv11AutoCon", 0.0),
    "active": ("v11_autocon_active", "MambaTSv11AutoCon", 0.001),
    "patchtst": ("patchtst_baseline", "PatchTST", None),
}
SEEDS = (2221, 2222, 2223)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _phase_rows(root: Path, key: str) -> tuple[list[dict], str]:
    phase, model, lambda_value = PHASES[key]
    manifest = root / key / "logs" / "job_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    expected = {(seed, model) for seed in SEEDS}
    rows, seen, source_hashes = [], set(), set()
    for record in records:
        coordinate = (int(record.get("seed")), str(record.get("model")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError("invalid or duplicate coordinate: {}".format(coordinate))
        if record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("failed job: {}".format(record.get("job_id")))
        seen.add(coordinate)
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        source_hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file():
            raise ValueError("missing training completion: {}".format(result_dir))
        if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("test artifact present: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        if config.get("model") != model or config.get("data") != "ETTh1" or int(config.get("pred_len")) != 336:
            raise ValueError("configuration mismatch: {}".format(result_dir))
        if config.get("skip_test") is not True or int(config.get("seed")) != coordinate[0]:
            raise ValueError("test/seed contract mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        losses = _load(checkpoint_root / result_dir.name / "loss_logs.json")
        values = [float(value) for value in losses.get("val_loss", [])]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("missing/non-finite validation loss: {}".format(result_dir))
        row = {
            "phase": phase,
            "model": model,
            "seed": coordinate[0],
            "best_val_mse": min(values),
            "best_epoch": int(record["best_epoch"]),
            "epochs_ran": int(record["epochs_ran"]),
            "elapsed_sec": float(record["elapsed_sec"]),
        }
        if key != "patchtst":
            if config.get("mamba_ts_variant") != "v11" or not bool(config.get("AutoCon")):
                raise ValueError("v11 AutoCon contract mismatch: {}".format(result_dir))
            if not bool(config.get("mamba_ts_repr_forecast_fusion")):
                raise ValueError("v11 direct-fusion contract mismatch: {}".format(result_dir))
            if not math.isclose(float(config.get("AutoCon_lambda")), lambda_value, abs_tol=1e-12):
                raise ValueError("AutoCon lambda mismatch: {}".format(result_dir))
            coverage = [float(value) for value in losses.get("vpt_transition_coverage", [])]
            counts = [float(value) for value in losses.get("vpt_transition_count", [])]
            if not coverage or not counts or max(coverage) < 0.95 or max(counts) <= 0.0:
                raise ValueError("VPT telemetry is absent or ineffective: {}".format(result_dir))
            row["vpt_transition_coverage_max"] = max(coverage)
            row["vpt_transition_count_max"] = max(counts)
        rows.append(row)
    if seen != expected or len(source_hashes) != 1:
        raise ValueError("coverage/source mismatch for phase {}".format(key))
    return rows, next(iter(source_hashes))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    phase_rows, hashes = {}, set()
    for key in PHASES:
        phase_rows[key], source_hash = _phase_rows(args.root.resolve(), key)
        hashes.add(source_hash)
    if len(hashes) != 1:
        raise ValueError("cross-phase source snapshot mismatch")
    by_phase = {key: {row["seed"]: row for row in rows} for key, rows in phase_rows.items()}
    values = {key: [by_phase[key][seed]["best_val_mse"] for seed in SEEDS] for key in PHASES}
    active_mean, null_mean, patch_mean = (statistics.fmean(values[key]) for key in ("active", "null", "patchtst"))
    wins_null = sum(a < n for a, n in zip(values["active"], values["null"]))
    wins_patch = sum(a < p for a, p in zip(values["active"], values["patchtst"]))
    summary = {
        "dataset": "ETTh1", "horizon": 336,
        "v11_null_mean_val_mse": null_mean,
        "v11_active_mean_val_mse": active_mean,
        "patchtst_mean_val_mse": patch_mean,
        "active_vs_null_percent": 100 * (active_mean / null_mean - 1),
        "active_vs_patchtst_percent": 100 * (active_mean / patch_mean - 1),
        "active_wins_vs_null": wins_null,
        "active_wins_vs_patchtst": wins_patch,
        "passes_autocon_gate": wins_null >= 2 and active_mean < null_mean,
        "passes_accuracy_gate": active_mean <= patch_mean,
        "source_snapshot": next(iter(hashes)),
        "test_metrics_produced": False,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", sum(phase_rows.values(), []))
    _write_csv(output / "summary.csv", [summary])
    lines = [
        "# MambaTS v11 VPT ETTh1 H336 Preflight", "",
        "Validation-only results. No test metric was generated or read.", "",
        "| v11 null MSE | v11 active MSE | PatchTST MSE | active vs null | active wins/null | active vs PatchTST | active wins/PatchTST | AutoCon gate | accuracy gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        "| {v11_null_mean_val_mse:.6f} | {v11_active_mean_val_mse:.6f} | {patchtst_mean_val_mse:.6f} | {active_vs_null_percent:+.3f}% | {active_wins_vs_null}/3 | {active_vs_patchtst_percent:+.3f}% | {active_wins_vs_patchtst}/3 | {passes_autocon_gate} | {passes_accuracy_gate} |".format(**summary),
        "",
        "The pre-registered continuation rule requires both gates. Source snapshot: `{}`.".format(summary["source_snapshot"]),
    ]
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
