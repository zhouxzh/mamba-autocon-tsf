#!/usr/bin/env python3
"""Strict validation-only reader for the METR-LA/PEMS-BAY selector screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_training_plan import _job_specs, _load_json, _validate_plan


PHASES = {
    "selector_off_h12": ("NLinearVariableMambaAutoCon", False),
    "selector_on_h12": ("NLinearVariableMambaAutoCon", True),
    "external_context_h12": (None, False),
}
BASELINES = ("DLinear", "NLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3")
DATASETS = ("metr_la", "pems_bay")
SEEDS = (2711, 2712, 2713)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint_scale(path: Path) -> float:
    state = torch.load(path, map_location="cpu")
    value = state.get("acf_memory_selector_scale")
    if value is None:
        raise ValueError(f"selector checkpoint lacks acf_memory_selector_scale: {path}")
    return float(value.detach().cpu().item())


def _phase_rows(root: Path, phase_name: str, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    phase = next(item for item in plan["phases"] if item["name"] == phase_name)
    phase_root = root / phase_name
    validation = _json(phase_root / "phase_validation.json")
    if validation.get("valid") is not True:
        raise ValueError(f"invalid phase validation: {phase_name}")
    expected_jobs = len(_job_specs(plan, phase, {}, allow_unresolved=True))
    if validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"phase job count mismatch: {phase_name}")
    records = _manifest(phase_root / "logs" / "job_manifest.jsonl")
    if len(records) != expected_jobs or len({r.get("job_id") for r in records}) != expected_jobs:
        raise ValueError(f"manifest coverage mismatch: {phase_name}")
    expected_models = BASELINES if phase_name == "external_context_h12" else ("NLinearVariableMambaAutoCon",)
    expected = {(dataset, model, seed) for dataset in DATASETS for model in expected_models for seed in SEEDS}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    source_hashes: set[str] = set()
    for record in records:
        dataset, model, seed = str(record["dataset"]), str(record["model"]), int(record["seed"])
        coordinate = (dataset, model, seed)
        if coordinate not in expected or coordinate in seen:
            raise ValueError(f"unexpected or duplicate coordinate: {coordinate}")
        if record.get("status") != "ok" or int(record.get("returncode", 1)) != 0:
            raise ValueError(f"failed job: {record.get('job_id')}")
        before, after = record.get("source_tree_sha256_before"), record.get("source_tree_sha256_after")
        if not before or before != after:
            raise ValueError(f"source snapshot drift: {record.get('job_id')}")
        source_hashes.add(before)
        result_dir = Path(record["expected_artifact"]).parent
        if not (result_dir / "training_complete.json").is_file():
            raise ValueError(f"missing training completion: {result_dir}")
        if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"test artifact present: {result_dir}")
        config = _json(result_dir / "config.json")
        if config.get("data") != dataset or config.get("model") != model or int(config.get("pred_len")) != 12:
            raise ValueError(f"config mismatch: {result_dir}")
        if config.get("skip_test") is not True or int(config.get("seed")) != seed:
            raise ValueError(f"test/seed contract mismatch: {result_dir}")
        checkpoint_root = Path(record["command"][record["command"].index("--checkpoints") + 1])
        checkpoint_dir = checkpoint_root / result_dir.name
        losses = [float(value) for value in _json(checkpoint_dir / "loss_logs.json").get("val_loss", [])]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"invalid validation losses: {result_dir}")
        logs = _json(checkpoint_dir / "loss_logs.json")
        row: dict[str, Any] = {
            "phase": phase_name, "dataset": dataset, "model": model, "seed": seed,
            "best_val_mse": min(losses), "best_epoch": int(record["best_epoch"]),
            "epochs_ran": int(record["epochs_ran"]), "elapsed_sec": float(record["elapsed_sec"]),
            "acf_relation_effective_lags": json.dumps(logs.get("acf_relation_effective_lags", [])),
            "acf_relation_global_valid_fraction": float(max(logs.get("acf_relation_global_valid_fraction", [float("nan")])))
            if logs.get("acf_relation_global_valid_fraction") else float("nan"),
            "source_tree_sha256": before,
        }
        selector_on = phase_name == "selector_on_h12"
        if phase_name in ("selector_off_h12", "selector_on_h12"):
            required = {
                "AutoCon": True, "mamba_use_acf_memory": True, "mamba_variable_repr_fusion": True,
                "mamba_acf_relation_loss": True, "mamba_variable_acf_memory_selector": selector_on,
            }
            if any(config.get(key) != value for key, value in required.items()):
                raise ValueError(f"selector config mismatch: {result_dir}")
            legal_lags = [int(lag) for lag in logs.get("acf_relation_effective_lags", []) if int(lag) <= int(config["seq_len"])]
            if not legal_lags or not (result_dir / "acf_prior.npz").is_file():
                raise ValueError(f"missing legal train-only ACF evidence: {result_dir}")
            if selector_on:
                scale = _checkpoint_scale(checkpoint_dir / "checkpoint.pth")
                if not math.isfinite(scale) or abs(scale) <= 1e-8:
                    raise ValueError(f"selector remained inactive: {result_dir}")
                row["selector_scale"] = scale
        else:
            if any(config.get(key) not in (False, 0, None) for key in ("AutoCon", "mamba_use_acf_memory", "mamba_variable_repr_fusion", "mamba_acf_relation_loss")):
                raise ValueError(f"external baseline contains candidate-only objective: {result_dir}")
        seen.add(coordinate)
        rows.append(row)
    if seen != expected or len(source_hashes) != 1:
        raise ValueError(f"coverage/source mismatch: {phase_name}")
    return rows, next(iter(source_hashes))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path, root, output = args.plan.resolve(), args.root.resolve(), args.output_dir.resolve()
    plan = _json(plan_path)
    errors = _validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    all_rows: list[dict[str, Any]] = []
    phase_hashes: set[str] = set()
    for phase_name in PHASES:
        rows, source_hash = _phase_rows(root, phase_name, plan)
        all_rows.extend(rows)
        phase_hashes.add(source_hash)
    if len(phase_hashes) != 1:
        raise ValueError("source snapshot differs across phases")
    by_key = {(row["phase"], row["dataset"], row["model"], row["seed"]): row for row in all_rows}
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        on = [by_key[("selector_on_h12", dataset, "NLinearVariableMambaAutoCon", seed)]["best_val_mse"] for seed in SEEDS]
        off = [by_key[("selector_off_h12", dataset, "NLinearVariableMambaAutoCon", seed)]["best_val_mse"] for seed in SEEDS]
        baseline_means = {model: statistics.fmean(by_key[("external_context_h12", dataset, model, seed)]["best_val_mse"] for seed in SEEDS) for model in BASELINES}
        strongest = min(baseline_means, key=baseline_means.get)
        on_mean, off_mean = statistics.fmean(on), statistics.fmean(off)
        summaries.append({
            "dataset": dataset, "horizon": 12, "selector_on_mean_val_mse": on_mean,
            "selector_off_mean_val_mse": off_mean, "selector_vs_off_percent": 100 * (on_mean / off_mean - 1),
            "selector_wins_vs_off": sum(a < b for a, b in zip(on, off)), "strongest_baseline": strongest,
            "strongest_baseline_mean_val_mse": baseline_means[strongest],
            "selector_vs_strongest_percent": 100 * (on_mean / baseline_means[strongest] - 1),
            "passes_selector_gate": on_mean < off_mean and sum(a < b for a, b in zip(on, off)) >= 2 and max((a / b - 1) * 100 for a, b in zip(on, off)) <= 2,
            "passes_accuracy_gate": on_mean <= baseline_means[strongest] * 1.02,
            **{f"{model}_mean_val_mse": value for model, value in baseline_means.items()},
        })
    result = {"source_tree_sha256": next(iter(phase_hashes)), "plan_sha256": _sha256(plan_path), "rows": len(all_rows), "summaries": summaries, "test_metrics_produced": False}
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", all_rows)
    _write_csv(output / "summary.csv", summaries)
    (output / "readout.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# NLinearVariableMamba AutoCon + ACF selector H12 screen", "", "Validation-only readout; no test metrics were generated or read.", "", "| Dataset | selector off | selector on | on vs off | wins / 3 | strongest baseline | vs strongest | selector gate | accuracy gate |", "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |"]
    for item in summaries:
        lines.append(f"| {item['dataset']} | {item['selector_off_mean_val_mse']:.6f} | {item['selector_on_mean_val_mse']:.6f} | {item['selector_vs_off_percent']:+.3f}% | {item['selector_wins_vs_off']}/3 | {item['strongest_baseline']} ({item['strongest_baseline_mean_val_mse']:.6f}) | {item['selector_vs_strongest_percent']:+.3f}% | {item['passes_selector_gate']} | {item['passes_accuracy_gate']} |")
    lines += ["", f"Rows: `{len(all_rows)}`; plan SHA256: `{result['plan_sha256']}`; source snapshot: `{result['source_tree_sha256']}`."]
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
