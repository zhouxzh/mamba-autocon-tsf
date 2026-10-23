#!/usr/bin/env python3
"""Strict reader for a frozen H96 variable-Mamba selector baseline screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch


PHASES = {
    "off": ("selector_off", "NLinearVariableMambaAutoCon"),
    "on": ("selector_on", "NLinearVariableMambaAutoCon"),
    "baselines": ("strong_baselines", None),
}
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu")
    value = state.get("acf_memory_selector_scale")
    if value is None:
        raise ValueError("selector checkpoint lacks acf_memory_selector_scale: {}".format(path))
    return float(value.detach().cpu().item())


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in value.split(",") if seed)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("--seeds must contain exactly three unique integers")
    return seeds


def _read_phase(
    root: Path,
    key: str,
    dataset: str,
    horizon: int,
    seeds: tuple[int, ...],
) -> tuple[list[dict], str]:
    phase, expected_model = PHASES[key]
    records = [json.loads(line) for line in (root / key / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected_models = BASELINES if key == "baselines" else (expected_model,)
    expected = {(model, seed) for model in expected_models for seed in seeds}
    rows, seen, hashes = [], set(), set()
    for record in records:
        coordinate = (str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError("invalid or duplicate coordinate: {}".format(coordinate))
        if record.get("status") != "ok" or record.get("returncode") != 0:
            raise ValueError("failed job: {}".format(record.get("job_id")))
        seen.add(coordinate)
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError("source snapshot drift: {}".format(record.get("job_id")))
        hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file():
            raise ValueError("missing completion: {}".format(result_dir))
        if any((result_dir / artifact).exists() for artifact in TEST_ARTIFACTS):
            raise ValueError("test artifact present: {}".format(result_dir))
        config = _json(result_dir / "config.json")
        if (
            config.get("model") != coordinate[0]
            or config.get("data") != dataset
            or int(config.get("pred_len")) != horizon
        ):
            raise ValueError("config mismatch: {}".format(result_dir))
        if config.get("skip_test") is not True or int(config.get("seed")) != coordinate[1]:
            raise ValueError("test/seed contract mismatch: {}".format(result_dir))
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        checkpoint_dir = checkpoint_root / result_dir.name
        logs = _json(checkpoint_dir / "loss_logs.json")
        losses = [float(value) for value in logs.get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("non-finite val losses: {}".format(result_dir))
        row = {"phase": phase, "model": coordinate[0], "seed": coordinate[1], "best_val_mse": min(losses), "best_epoch": int(record["best_epoch"]), "epochs_ran": int(record["epochs_ran"]), "elapsed_sec": float(record["elapsed_sec"])}
        if key in {"off", "on"}:
            selector_enabled = key == "on"
            required = {"AutoCon": True, "mamba_use_acf_memory": True, "mamba_variable_repr_fusion": True, "mamba_acf_relation_loss": True, "mamba_variable_acf_memory_selector": selector_enabled}
            if any(config.get(name) != value for name, value in required.items()):
                raise ValueError("selector config mismatch: {}".format(result_dir))
            effective_lags = logs.get("acf_relation_effective_lags", [])
            if not effective_lags or not any(int(lag) <= int(config.get("seq_len")) for lag in effective_lags):
                raise ValueError("no legal train-only lag: {}".format(result_dir))
            if not (result_dir / "acf_prior.npz").is_file():
                raise ValueError("missing ACF sidecar: {}".format(result_dir))
            if selector_enabled:
                scale = _checkpoint_scale(checkpoint_dir / "checkpoint.pth")
                if not math.isfinite(scale) or abs(scale) <= 1e-8:
                    raise ValueError("inactive selector: {}".format(result_dir))
                row["selector_scale"] = scale
        rows.append(row)
    if seen != expected or len(hashes) != 1:
        raise ValueError("coverage/source mismatch for phase {}".format(key))
    return rows, next(iter(hashes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="traffic")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seeds", type=_parse_seeds, default=(2241, 2242, 2243))
    args = parser.parse_args()
    root = args.root.resolve()
    phase_rows, hashes = {}, set()
    for key in PHASES:
        phase_rows[key], source_hash = _read_phase(
            root, key, args.dataset, args.horizon, args.seeds
        )
        hashes.add(source_hash)
    if len(hashes) != 1:
        raise ValueError("source snapshot differs across phases")
    by_phase = {key: {(row["model"], row["seed"]): row for row in rows} for key, rows in phase_rows.items()}
    on = [by_phase["on"][("NLinearVariableMambaAutoCon", seed)]["best_val_mse"] for seed in args.seeds]
    off = [by_phase["off"][("NLinearVariableMambaAutoCon", seed)]["best_val_mse"] for seed in args.seeds]
    baseline_means = {
        model: statistics.fmean(
            by_phase["baselines"][(model, seed)]["best_val_mse"]
            for seed in args.seeds
        )
        for model in BASELINES
    }
    on_mean, off_mean = statistics.fmean(on), statistics.fmean(off)
    strongest_model = min(baseline_means, key=baseline_means.get)
    strongest_mean = baseline_means[strongest_model]
    summary = {"dataset": args.dataset, "horizon": args.horizon, "seeds": list(args.seeds), "selector_on_mean_val_mse": on_mean, "selector_off_mean_val_mse": off_mean, "selector_vs_off_percent": 100 * (on_mean / off_mean - 1), "selector_wins_vs_off": sum(a < b for a, b in zip(on, off)), "strongest_baseline": strongest_model, "strongest_baseline_mean_val_mse": strongest_mean, "selector_vs_strongest_percent": 100 * (on_mean / strongest_mean - 1), "passes_selector_gate": on_mean < off_mean and sum(a < b for a, b in zip(on, off)) >= 2, "passes_accuracy_gate": on_mean <= strongest_mean, "source_snapshot": next(iter(hashes)), "test_metrics_produced": False}
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    _csv(output / "validation_rows.csv", sum(phase_rows.values(), []))
    _csv(output / "summary.csv", [summary])
    title_dataset = str(args.dataset).upper()
    lines = [f"# Variable-Mamba ACF Selector {title_dataset} H{args.horizon} Strong-Baseline Readout", "", "Validation-only. No test metric was generated or read.", "", "| Selector off | Selector on | On vs off | wins/off | Strongest baseline | Selector vs strongest | selector gate | accuracy gate |", "| ---: | ---: | ---: | ---: | --- | ---: | --- | --- |", "| {selector_off_mean_val_mse:.6f} | {selector_on_mean_val_mse:.6f} | {selector_vs_off_percent:+.3f}% | {selector_wins_vs_off}/3 | {strongest_baseline} ({strongest_baseline_mean_val_mse:.6f}) | {selector_vs_strongest_percent:+.3f}% | {passes_selector_gate} | {passes_accuracy_gate} |".format(**summary), "", "Source snapshot: `{}`.".format(summary["source_snapshot"])]
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
