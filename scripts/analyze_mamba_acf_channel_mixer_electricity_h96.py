#!/usr/bin/env python3
"""Strict reader for the frozen Electricity H96 MambaACF mixer factorial."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch


BASELINES = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")
SEEDS = (2271, 2272, 2273)
CHANNELS = {"electricity": 321, "ETTh1": 7, "traffic": 862}
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")
MIXER_PHASES = {
    "mixer_off_relation_null": (False, 0.0),
    "mixer_off_relation_active": (False, 0.001),
    "mixer_on_relation_null": (True, 0.0),
    "mixer_on_relation_active": (True, 0.001),
}


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_root(record: dict, result_dir: Path) -> Path:
    command = record.get("command", [])
    if "--checkpoints" not in command:
        raise ValueError("checkpoint root missing from command: {}".format(record.get("job_id")))
    return Path(command[command.index("--checkpoints") + 1]) / result_dir.name


def _mixer_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu")
    key = "data_dependent_channel_mixer.residual_scale"
    if key not in state:
        raise ValueError("mixer checkpoint lacks {}: {}".format(key, path))
    value = state[key]
    return float(value.detach().cpu().item())


def _records(root: Path, phase: str) -> list[dict]:
    manifest = root / phase / "logs" / "job_manifest.jsonl"
    if not manifest.is_file():
        raise ValueError("missing manifest: {}".format(manifest))
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def _read_baselines(root: Path, dataset: str) -> tuple[list[dict], str]:
    expected = {(model, seed) for model in BASELINES for seed in SEEDS}
    rows, seen, hashes = [], set(), set()
    for record in _records(root, "baselines"):
        coordinate = (str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError("invalid or duplicate baseline coordinate: {}".format(coordinate))
        _validate_record(record)
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        result_dir = Path(record["expected_artifact"]).parent
        config = _json(result_dir / "config.json")
        _validate_common_config(config, coordinate[0], coordinate[1], dataset)
        required = {
            "AutoCon": False,
            "mamba_use_acf_memory": False,
            "mamba_acf_relation_loss": False,
            "mamba_acf_relation_lag_mask": False,
            "mamba_acf_relation_min_lag": 0,
        }
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("baseline leaked candidate option: {}".format(result_dir))
        rows.append(_loss_row(record, result_dir, "strong_baselines", coordinate))
    if seen != expected or len(hashes) != 1:
        raise ValueError("baseline coverage or source snapshot mismatch")
    return rows, next(iter(hashes))


def _read_mixer_phase(root: Path, phase: str, dataset: str) -> tuple[list[dict], str]:
    enabled, expected_lambda = MIXER_PHASES[phase]
    expected = {("MambaACFAutoCon", seed) for seed in SEEDS}
    rows, seen, hashes = [], set(), set()
    for record in _records(root, phase):
        coordinate = (str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError("invalid or duplicate mixer coordinate: {}".format(coordinate))
        _validate_record(record)
        seen.add(coordinate)
        hashes.add(record["source_tree_sha256_before"])
        result_dir = Path(record["expected_artifact"]).parent
        config = _json(result_dir / "config.json")
        _validate_common_config(config, coordinate[0], coordinate[1], dataset)
        required = {
            "AutoCon": True,
            "AutoCon_wnorm": "MeanStd",
            "AutoCon_lambda": expected_lambda,
            "AutoCon_acf_source": "raw",
            "mamba_use_acf_memory": True,
            "mamba_acf_lag_mode": "train_peaks_available",
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_min_lag": 96,
            "mamba_data_dependent_channel_mixer": enabled,
            "mamba_data_dependent_channel_mixer_rank": 4,
        }
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("mixer config mismatch: {}".format(result_dir))
        if not (result_dir / "acf_prior.npz").is_file():
            raise ValueError("missing train-only ACF sidecar: {}".format(result_dir))
        row = _loss_row(record, result_dir, phase, coordinate)
        checkpoint = _checkpoint_root(record, result_dir) / "checkpoint.pth"
        state = torch.load(checkpoint, map_location="cpu")
        mixer_key = "data_dependent_channel_mixer.residual_scale"
        if enabled:
            if mixer_key not in state:
                raise ValueError("enabled mixer has no checkpoint state: {}".format(result_dir))
            scale = _mixer_scale(checkpoint)
            if not math.isfinite(scale) or abs(scale) <= 1e-8:
                raise ValueError("enabled mixer is inactive after training: {}".format(result_dir))
            row["mixer_residual_scale"] = scale
        elif mixer_key in state:
            raise ValueError("disabled mixer leaked checkpoint state: {}".format(result_dir))
        rows.append(row)
    if seen != expected or len(hashes) != 1:
        raise ValueError("coverage or source snapshot mismatch for {}".format(phase))
    return rows, next(iter(hashes))


def _validate_record(record: dict) -> None:
    if record.get("status") != "ok" or record.get("returncode") != 0:
        raise ValueError("failed job: {}".format(record.get("job_id")))
    before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
    if not before or before != after:
        raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
    result_dir = Path(record.get("expected_artifact", "")).parent
    if not (result_dir / "training_complete.json").is_file():
        raise ValueError("missing completion artifact: {}".format(result_dir))
    if any((result_dir / artifact).exists() for artifact in TEST_ARTIFACTS):
        raise ValueError("test artifact present: {}".format(result_dir))


def _validate_common_config(config: dict, model: str, seed: int, dataset: str) -> None:
    if config.get("model") != model or config.get("data") != dataset:
        raise ValueError("model/dataset config mismatch")
    if int(config.get("pred_len")) != 96 or int(config.get("seq_len")) != 96:
        raise ValueError("horizon/context config mismatch")
    if config.get("features") != "M" or int(config.get("enc_in")) != CHANNELS[dataset]:
        raise ValueError("multivariate channel config mismatch")
    if config.get("skip_test") is not True or int(config.get("seed")) != seed:
        raise ValueError("test/seed config mismatch")


def _loss_row(record: dict, result_dir: Path, phase: str, coordinate: tuple[str, int]) -> dict:
    logs = _json(_checkpoint_root(record, result_dir) / "loss_logs.json")
    losses = [float(value) for value in logs.get("val_loss", [])]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("non-finite validation loss: {}".format(result_dir))
    return {
        "phase": phase,
        "model": coordinate[0],
        "seed": coordinate[1],
        "best_val_mse": min(losses),
        "best_epoch": int(record["best_epoch"]),
        "epochs_ran": int(record["epochs_ran"]),
        "elapsed_sec": float(record["elapsed_sec"]),
    }


def _phase_values(rows: list[dict], phase: str) -> list[float]:
    by_seed = {row["seed"]: row["best_val_mse"] for row in rows if row["phase"] == phase}
    if set(by_seed) != set(SEEDS):
        raise ValueError("incomplete rows for {}".format(phase))
    return [by_seed[seed] for seed in SEEDS]


def _paired_percent(treatment: list[float], reference: list[float]) -> float:
    return 100.0 * (statistics.fmean(treatment) / statistics.fmean(reference) - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(CHANNELS), default="electricity")
    parser.add_argument("--seeds", default="2271,2272,2273")
    args = parser.parse_args()
    global SEEDS
    SEEDS = tuple(int(value) for value in args.seeds.split(",") if value)
    if len(SEEDS) != 3 or len(set(SEEDS)) != 3:
        raise ValueError("--seeds must contain exactly three unique integers")
    root, output = args.root.resolve(), args.output_dir.resolve()
    rows, hashes = [], set()
    baseline_rows, source_hash = _read_baselines(root, args.dataset)
    rows.extend(baseline_rows)
    hashes.add(source_hash)
    for phase in MIXER_PHASES:
        phase_rows, source_hash = _read_mixer_phase(root, phase, args.dataset)
        rows.extend(phase_rows)
        hashes.add(source_hash)
    if len(hashes) != 1:
        raise ValueError("source snapshot differs across factorial phases")

    values = {phase: _phase_values(rows, phase) for phase in MIXER_PHASES}
    baseline_means = {
        model: statistics.fmean(row["best_val_mse"] for row in baseline_rows if row["model"] == model)
        for model in BASELINES
    }
    strongest = min(baseline_means, key=baseline_means.get)
    primary, off_active, on_null = (
        values["mixer_on_relation_active"],
        values["mixer_off_relation_active"],
        values["mixer_on_relation_null"],
    )
    mixer_null_effect = _paired_percent(values["mixer_on_relation_null"], values["mixer_off_relation_null"])
    mixer_active_effect = _paired_percent(primary, off_active)
    relation_on_effect = _paired_percent(primary, on_null)
    relation_off_effect = _paired_percent(values["mixer_off_relation_active"], values["mixer_off_relation_null"])
    primary_mean = statistics.fmean(primary)
    summary = {
        "dataset": args.dataset, "horizon": 96, "seeds": list(SEEDS),
        "mixer_on_relation_active_mean_val_mse": primary_mean,
        "mixer_off_relation_active_mean_val_mse": statistics.fmean(off_active),
        "mixer_on_relation_null_mean_val_mse": statistics.fmean(on_null),
        "mixer_on_relation_active_vs_off_active_percent": mixer_active_effect,
        "mixer_on_relation_active_vs_on_null_percent": relation_on_effect,
        "mixer_on_relation_null_vs_off_null_percent": mixer_null_effect,
        "mixer_off_relation_active_vs_off_null_percent": relation_off_effect,
        "mixer_relation_interaction_percentage_points": mixer_active_effect - mixer_null_effect,
        "mixer_wins_vs_off_active": sum(a < b for a, b in zip(primary, off_active)),
        "relation_wins_with_mixer": sum(a < b for a, b in zip(primary, on_null)),
        "strongest_baseline": strongest,
        "strongest_baseline_mean_val_mse": baseline_means[strongest],
        "primary_vs_strongest_percent": 100.0 * (primary_mean / baseline_means[strongest] - 1.0),
        "passes_mixer_gate": primary_mean < statistics.fmean(off_active) and sum(a < b for a, b in zip(primary, off_active)) >= 2,
        "passes_external_accuracy_gate": primary_mean <= baseline_means[strongest],
        "passes_relation_noninferiority_gate": primary_mean <= statistics.fmean(on_null) * 1.01,
        "passes_relation_improvement_gate": primary_mean < statistics.fmean(on_null) and sum(a < b for a, b in zip(primary, on_null)) >= 2,
        "source_snapshot": next(iter(hashes)),
        "test_metrics_produced": False,
    }
    summary["passes_branch_gate"] = bool(
        summary["passes_mixer_gate"]
        and summary["passes_external_accuracy_gate"]
        and summary["passes_relation_noninferiority_gate"]
    )
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "validation_rows.csv", rows)
    _csv(output / "summary.csv", [summary])
    lines = [
        "# MambaACF Channel-State Mixer {} H96 Readout".format(args.dataset),
        "",
        "Validation-only. No test metric was generated or read.",
        "",
        "| Primary candidate | Mixer active control | Mixer null | Strongest baseline | Mixer gate | External gate | Relation non-inferiority | Branch gate |",
        "| ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        "| {mixer_on_relation_active_mean_val_mse:.6f} | {mixer_off_relation_active_mean_val_mse:.6f} ({mixer_on_relation_active_vs_off_active_percent:+.3f}%, {mixer_wins_vs_off_active}/3 wins) | {mixer_on_relation_null_mean_val_mse:.6f} ({mixer_on_relation_active_vs_on_null_percent:+.3f}%, {relation_wins_with_mixer}/3 wins) | {strongest_baseline} ({strongest_baseline_mean_val_mse:.6f}; {primary_vs_strongest_percent:+.3f}%) | {passes_mixer_gate} | {passes_external_accuracy_gate} | {passes_relation_noninferiority_gate} | {passes_branch_gate} |".format(**summary),
        "",
        "Factorial mixer effect under relation-null/active: {:+.3f}% / {:+.3f}%; interaction: {:+.3f} percentage points.".format(mixer_null_effect, mixer_active_effect, summary["mixer_relation_interaction_percentage_points"]),
        "",
        "Source snapshot: `{}`.".format(summary["source_snapshot"]),
    ]
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
