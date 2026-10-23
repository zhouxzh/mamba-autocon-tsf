#!/usr/bin/env python3
"""Strict validation-only reader for the Solar variable-axis Mamba matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_DEFAULT = ROOT / "docs/research/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928.json"
ARTIFACT_DEFAULT = ROOT / "artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928"
SEEDS = (3281, 3282, 3283)
VARIABLE_MODEL = "NLinearVariableMambaAutoCon"
EXTERNAL_BASELINES = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")
TEST_NAMES = {"metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}")
    return number


def phase_plan(plan: dict[str, Any], name: str) -> dict[str, Any]:
    for phase in plan.get("phases", []):
        if isinstance(phase, dict) and phase.get("name") == name:
            return phase
    raise ValueError(f"missing phase {name}")


def load_phase(root: Path, phase: dict[str, Any], expected_models: tuple[str, ...]) -> tuple[list[dict[str, Any]], str]:
    validation = read_json(root / "phase_validation.json")
    expected_count = len(SEEDS) * len(expected_models)
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_count:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = read_json(root / "provenance.json")
    source_hash = str(provenance.get("source_tree_sha256", ""))
    if not source_hash:
        raise ValueError(f"missing source hash: {root}")
    if (root / "test_results").exists() and any((root / "test_results").rglob("*")):
        raise ValueError(f"test artifacts present: {root}")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        model, seed = str(record.get("model")), int(record.get("seed"))
        coordinate = (model, seed)
        if coordinate in seen or model not in expected_models or seed not in SEEDS:
            raise ValueError(f"unexpected or duplicate coordinate: {record.get('job_id')}")
        if record.get("status") != "ok" or record.get("returncode") != 0 or record.get("dataset") != "solar":
            raise ValueError(f"failed record: {record.get('job_id')}")
        if record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError(f"source drift: {record.get('job_id')}")
        result = Path(str(record.get("expected_artifact"))).parent
        if not (result / "training_complete.json").is_file() or any((result / name).exists() for name in TEST_NAMES):
            raise ValueError(f"incomplete or test-contaminated result: {result}")
        config = read_json(result / "config.json")
        if (config.get("model"), int(config.get("seed")), config.get("data"), int(config.get("pred_len"))) != (model, seed, "solar", 96):
            raise ValueError(f"config coordinate mismatch: {result}")
        if config.get("skip_test") is not True:
            raise ValueError(f"test execution not disabled: {result}")
        if phase["name"] in {"variable_active", "variable_fusion_off", "variable_relation_null"}:
            expected_fusion = phase["name"] != "variable_fusion_off"
            expected_lambda = 0.0 if phase["name"] == "variable_relation_null" else 0.0001
            if (
                config.get("AutoCon") is not True
                or config.get("mamba_variable_repr_fusion") is not expected_fusion
                or config.get("mamba_acf_relation_loss") is not True
                or not math.isclose(finite(config.get("AutoCon_lambda"), "AutoCon_lambda"), expected_lambda)
            ):
                raise ValueError(f"mechanism config mismatch: {result}")
        elif phase["name"] == "external_baselines":
            if config.get("AutoCon") is True or config.get("mamba_variable_repr_fusion") is True:
                raise ValueError(f"baseline received candidate-only mechanism: {result}")
        elif phase["name"] == "temporal_mambatsv3_context":
            if not (
                config.get("AutoCon") is True
                and config.get("mamba_ts_variant") == "v3"
                and config.get("mamba_ts_repr_forecast_fusion") is True
            ):
                raise ValueError(f"temporal Mamba context mismatch: {result}")
        checkpoint = root / "checkpoints" / result.name / "loss_logs.json"
        losses = [finite(value, f"val_loss:{record.get('job_id')}") for value in read_json(checkpoint).get("val_loss", [])]
        if not losses:
            raise ValueError(f"missing validation loss: {record.get('job_id')}")
        rows.append({"phase": phase["name"], "model": model, "seed": seed, "best_val_mse": min(losses), "elapsed_sec": finite(record.get("elapsed_sec"), "elapsed_sec")})
        seen.add(coordinate)
    expected = {(model, seed) for model in expected_models for seed in SEEDS}
    if seen != expected:
        raise ValueError(f"coverage mismatch: {root}")
    return rows, source_hash


def paired(rows_by_phase: dict[str, list[dict[str, Any]]], left: str, right: str, label: str) -> dict[str, Any]:
    left_by_seed = {row["seed"]: row["best_val_mse"] for row in rows_by_phase[left]}
    right_by_seed = {row["seed"]: row["best_val_mse"] for row in rows_by_phase[right]}
    deltas = [left_by_seed[seed] - right_by_seed[seed] for seed in SEEDS]
    return {"contrast": label, "left": left, "right": right, "left_mean": statistics.fmean(left_by_seed.values()), "right_mean": statistics.fmean(right_by_seed.values()), "delta": statistics.fmean(deltas), "relative_delta_pct": 100.0 * statistics.fmean(deltas) / statistics.fmean(right_by_seed.values()), "left_wins": sum(delta < 0 for delta in deltas), "n": len(deltas)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    plan = read_json(args.plan.resolve())
    if plan.get("status") != "exploratory_nlinear_variable_mamba_solar_h96_mechanism_strong_baseline" or plan.get("execution", {}).get("authorized") is not False:
        raise ValueError("unexpected exploratory plan contract")
    artifact = args.artifact_root.resolve()
    phase_specs = (("variable_active", "active", (VARIABLE_MODEL,)), ("variable_fusion_off", "fusion_off", (VARIABLE_MODEL,)), ("variable_relation_null", "relation_null", (VARIABLE_MODEL,)), ("external_baselines", "baselines", EXTERNAL_BASELINES), ("temporal_mambatsv3_context", "mambatsv3_context", ("MambaTSv3AutoCon",)))
    rows_by_phase: dict[str, list[dict[str, Any]]] = {}
    source_hashes: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    for name, directory, models in phase_specs:
        rows, source_hash = load_phase(artifact / directory, phase_plan(plan, name), models)
        rows_by_phase[name] = rows
        source_hashes.add(source_hash)
        all_rows.extend(rows)
    if len(source_hashes) != 1:
        raise ValueError("source snapshots differ across phases")
    summaries = []
    for phase, rows in rows_by_phase.items():
        for model in sorted({row["model"] for row in rows}):
            values = [row for row in rows if row["model"] == model]
            summaries.append({"phase": phase, "model": model, "mean_val_mse": statistics.fmean(row["best_val_mse"] for row in values), "std_val_mse": statistics.stdev(row["best_val_mse"] for row in values), "mean_elapsed_sec": statistics.fmean(row["elapsed_sec"] for row in values), "n": len(values)})
    contrasts = [paired(rows_by_phase, "variable_active", "variable_fusion_off", "Active vs fusion-off"), paired(rows_by_phase, "variable_active", "variable_relation_null", "Active vs relation-null")]
    active = {row["seed"]: row["best_val_mse"] for row in rows_by_phase["variable_active"]}
    for baseline in EXTERNAL_BASELINES + ("MambaTSv3AutoCon",):
        phase = "external_baselines" if baseline in EXTERNAL_BASELINES else "temporal_mambatsv3_context"
        reference = {row["seed"]: row["best_val_mse"] for row in rows_by_phase[phase] if row["model"] == baseline}
        deltas = [active[seed] - reference[seed] for seed in SEEDS]
        contrasts.append({"contrast": f"Active vs {baseline}", "left": "variable_active", "right": baseline, "left_mean": statistics.fmean(active.values()), "right_mean": statistics.fmean(reference.values()), "delta": statistics.fmean(deltas), "relative_delta_pct": 100.0 * statistics.fmean(deltas) / statistics.fmean(reference.values()), "left_wins": sum(delta < 0 for delta in deltas), "n": len(deltas)})
    output = (args.output_dir or artifact / "readout").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("validation_rows.csv", all_rows), ("summary.csv", summaries), ("paired_summary.csv", contrasts)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    lines = ["# Solar H96 Variable-Axis Mamba Mechanism and Strong-Baseline Validation", "", "Validation-only. This reader rejects held-out test artifacts and reports every registered mechanism contrast.", "", "| Arm | Model | Mean validation MSE | SD | Mean job sec |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append("| {phase} | {model} | {mean_val_mse:.6f} | {std_val_mse:.6f} | {mean_elapsed_sec:.3f} |".format(**row))
    lines.extend(["", "## Paired Active Deltas", "", "Negative delta favors the active variable-axis Mamba.", "", "| Contrast | Active MSE | Reference MSE | Delta | Relative delta | Active wins |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in contrasts:
        lines.append("| {contrast} | {left_mean:.6f} | {right_mean:.6f} | {delta:.6f} | {relative_delta_pct:.2f}% | {left_wins}/{n} |".format(**row))
    lines.extend(["", f"All {len(all_rows)} rows share source snapshot `{next(iter(source_hashes))}`. This is validation-only and does not support a held-out or causal claim unless the registered control contrast passes."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "contrasts": len(contrasts), "source_tree_sha256": next(iter(source_hashes)), "test_metrics_produced": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
