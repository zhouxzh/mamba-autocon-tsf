#!/usr/bin/env python3
"""Strict validation-only readout for the MambaACF Lite AutoCon screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "traffic")
SEEDS = (2021, 2022, 2023)
ARMS = ("MambaACFAutoCon", "MambaACFLiteAutoCon")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values):
    return statistics.fmean(values)


def _finite_series(logs, key: str, result_dir: Path):
    values = logs.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError("missing {}: {}".format(key, result_dir))
    parsed = [float(value) for value in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("non-finite {}: {}".format(key, result_dir))
    return parsed


def _check_config(config, record, arm: str, result_dir: Path):
    expected = {
        "model": "MambaACFAutoCon", "data": record["dataset"], "seed": record["seed"],
        "features": "M", "pred_len": 96, "AutoCon": True, "AutoCon_lambda": 0.03,
        "AutoCon_acf_source": "raw", "AutoCon_wnorm": "MeanStd", "skip_test": True,
        "mamba_causal_embedding": True, "mamba_acf_relation_loss": True,
        "mamba_acf_relation_lag_mask": True, "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_relation_scope": "both", "mamba_acf_relation_min_lag": 0,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError("{} mismatch for {}: {!r} != {!r}".format(key, result_dir, config.get(key), wanted))
    if arm == "MambaACFAutoCon":
        shape = ([1, 2], False, 16)
    else:
        shape = ([1], True, 8)
    for key, wanted in zip(("mamba_scales", "mamba_low_rank_head", "mamba_temporal_rank"), shape):
        if config.get(key) != wanted:
            raise ValueError("{} Lite/full contract mismatch in {}".format(key, result_dir))


def _read(root: Path):
    phase = root / "full_vs_lite_h96"
    manifest = phase / "logs" / "job_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {(dataset, seed, arm) for dataset in DATASETS for seed in SEEDS for arm in ARMS}
    rows, found, source_hashes = [], set(), set()
    for record in records:
        if record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("failed record: {}".format(record.get("job_id")))
        dataset, seed, arm = str(record.get("dataset")), int(record.get("seed")), str(record.get("model"))
        coordinate = (dataset, seed, arm)
        if coordinate not in expected or coordinate in found:
            raise ValueError("invalid coordinate: {}".format(coordinate))
        found.add(coordinate)
        if int(record.get("pred_len")) != 96 or not math.isclose(float(record.get("autocon_lambda")), .03):
            raise ValueError("horizon/lambda mismatch: {}".format(record.get("job_id")))
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        source_hashes.add(before)
        artifact = Path(record["expected_artifact"])
        if artifact.name != "training_complete.json" or not artifact.is_file():
            raise ValueError("missing training completion artifact: {}".format(record.get("job_id")))
        result_dir = artifact.parent
        if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("test artifact present: {}".format(result_dir))
        config = _load(result_dir / "config.json")
        _check_config(config, record, arm, result_dir)
        sidecar = result_dir / "acf_prior.npz"
        if not sidecar.is_file():
            raise ValueError("missing train-only ACF sidecar: {}".format(result_dir))
        command = record.get("command", [])
        checkpoint_root = Path(command[command.index("--checkpoints") + 1])
        checkpoint_dir = checkpoint_root / result_dir.name
        logs = _load(checkpoint_dir / "loss_logs.json")
        val_loss = _finite_series(logs, "val_loss", result_dir)
        active = _finite_series(logs, "acf_relation_global_contrastive_active_anchor_fraction", result_dir)
        degenerate = _finite_series(logs, "acf_relation_global_degenerate_positive_anchor_fraction", result_dir)
        sign_accuracy = _finite_series(logs, "acf_relation_global_sign_accuracy", result_dir)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError("unexpected relation objective: {}".format(result_dir))
        rows.append({
            "dataset": dataset, "horizon": 96, "seed": seed, "arm": arm,
            "best_val_mse": min(val_loss), "elapsed_sec": float(record["elapsed_sec"]),
            "epochs_ran": int(record["epochs_ran"]), "best_epoch": int(record["best_epoch"]),
            "relation_active_anchor_mean": _mean(active),
            "relation_degenerate_positive_mean": _mean(degenerate),
            "relation_sign_accuracy_mean": _mean(sign_accuracy),
            "acf_prior_sha256": _sha256(sidecar),
            "result_dir": str(result_dir),
        })
    if found != expected or len(rows) != len(expected) or len(source_hashes) != 1:
        raise ValueError("coverage/source mismatch")
    return rows, next(iter(source_hashes))


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, source_hash = _read(args.root.resolve())
    keyed = {(r["dataset"], r["seed"], r["arm"]): r for r in rows}
    summary, paired = [], []
    for dataset in DATASETS:
        full = [keyed[(dataset, seed, "MambaACFAutoCon")] for seed in SEEDS]
        lite = [keyed[(dataset, seed, "MambaACFLiteAutoCon")] for seed in SEEDS]
        deltas = []
        for baseline, candidate in zip(full, lite):
            delta = candidate["best_val_mse"] - baseline["best_val_mse"]
            relative = 100.0 * delta / baseline["best_val_mse"]
            deltas.append(relative)
            paired.append({"dataset": dataset, "horizon": 96, "seed": baseline["seed"], "full_val_mse": baseline["best_val_mse"], "lite_val_mse": candidate["best_val_mse"], "lite_minus_full_mse": delta, "relative_percent": relative})
        summary.append({
            "dataset": dataset, "horizon": 96,
            "full_mean_val_mse": _mean([r["best_val_mse"] for r in full]),
            "lite_mean_val_mse": _mean([r["best_val_mse"] for r in lite]),
            "lite_vs_full_relative_percent": _mean(deltas),
            "lite_seed_improvements": sum(delta < 0 for delta in deltas),
            "full_mean_elapsed_sec": _mean([r["elapsed_sec"] for r in full]),
            "lite_mean_elapsed_sec": _mean([r["elapsed_sec"] for r in lite]),
            "lite_vs_full_elapsed_percent": 100.0 * (_mean([r["elapsed_sec"] for r in lite]) / _mean([r["elapsed_sec"] for r in full]) - 1.0),
            "lite_relation_active_anchor_mean": _mean([r["relation_active_anchor_mean"] for r in lite]),
            "lite_relation_degenerate_positive_mean": _mean([r["relation_degenerate_positive_mean"] for r in lite]),
        })
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "paired_summary.csv", paired)
    _write_csv(output / "summary.csv", summary)
    lines = ["# MambaACF Lite AutoCon screen", "", "Validation-only paired readout. No test metric was read or produced.", "", "| Dataset | Full MSE | Lite MSE | Lite vs full | Lite wins | Lite wall time change |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append("| {dataset} | {full_mean_val_mse:.6f} | {lite_mean_val_mse:.6f} | {lite_vs_full_relative_percent:+.3f}% | {lite_seed_improvements}/3 | {lite_vs_full_elapsed_percent:+.3f}% |".format(**row))
    lines.extend(["", "Positive Lite-vs-full percentage means worse validation MSE. Source snapshot: `{}`.".format(source_hash), "", "The relation objective was checked from epoch telemetry; this screen does not isolate relation-loss benefit because both arms use the same active objective."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "source_snapshot": source_hash, "test_metrics_produced": False})


if __name__ == "__main__":
    main()
