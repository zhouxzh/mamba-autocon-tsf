#!/usr/bin/env python3
"""Analyze the paired raw-source ACF representation-fusion screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

from scripts.analyze_mamba_relation_null_control import (
    RELATION_LOG_KEYS,
    _finite,
    _hash_map_for_record,
    _manifest_records,
    _read_json,
    _relative,
    _resolve,
    _sha256,
    _summary_series,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    ("acf_repr_fusion_off_raw_h96", False),
    ("acf_repr_fusion_on_raw_h96", True),
)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)


def _load_phase(plan: Mapping[str, Any], phase_name: str, expected_fusion: bool) -> List[Dict[str, Any]]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    phase = next((item for item in phases if isinstance(item, Mapping) and item.get("name") == phase_name), None)
    if phase is None:
        raise ValueError(f"plan has no phase {phase_name!r}")
    phase_root = _resolve(str(phase.get("output_root", "")))
    validation_path = phase_root / "phase_validation.json"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ValueError(f"invalid phase validation: {validation_path}")
    records = _manifest_records(phase_root, validation)
    result_root = phase_root / "results"
    checkpoint_root = phase_root / "checkpoints"
    rows: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        if record.get("status") != "ok" or record.get("phase") != phase_name:
            raise ValueError(f"unsuccessful record: {record.get('job_id')}")
        dataset = str(record.get("dataset"))
        seed = int(record.get("seed"))
        coordinate = (dataset, seed)
        if coordinate in seen:
            raise ValueError(f"duplicate coordinate: {phase_name} {coordinate}")
        seen.add(coordinate)
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected coordinate: {phase_name} {coordinate}")
        if not math.isclose(float(record.get("autocon_lambda")), 0.03, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"unexpected AutoCon lambda: {record.get('job_id')}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected artifact: {record.get('job_id')}")
        metrics_path = _resolve(str(expected_artifact))
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not metrics_path.is_relative_to(result_root):
            raise ValueError(f"metrics artifact outside result root: {record.get('job_id')}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        config = _read_json(result_dir / "config.json")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {record.get('job_id')}")
        expected_config = {
            "model": "MambaACFAutoCon",
            "features": "M",
            "pred_len": 96,
            "AutoCon": True,
            "AutoCon_lambda": 0.03,
            "AutoCon_acf_source": "raw",
            "mamba_variable_mixer": True,
            "mamba_use_acf_memory": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_target_mode": "hard_selected",
            "mamba_acf_repr_fusion": expected_fusion,
        }
        for key, expected in expected_config.items():
            actual = config.get(key)
            if isinstance(expected, bool):
                actual = bool(actual)
            elif isinstance(expected, float):
                try:
                    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError(f"config {key} mismatch: {record.get('job_id')}")
                    continue
                except (TypeError, ValueError):
                    raise ValueError(f"config {key} mismatch: {record.get('job_id')}")
            if actual != expected:
                raise ValueError(f"config {key} mismatch: {record.get('job_id')}: {actual!r} != {expected!r}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        diagnostics = {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS}
        sidecar_hash = hashes.get("acf_prior.npz")
        if not sidecar_hash:
            raise ValueError(f"missing sidecar hash: {record.get('job_id')}")
        rows.append({
            "phase": phase_name,
            "fusion": expected_fusion,
            "dataset": dataset,
            "seed": seed,
            "mse": _finite(scaled.get("mse"), f"{record.get('job_id')} mse"),
            "mae": _finite(scaled.get("mae"), f"{record.get('job_id')} mae"),
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{record.get('job_id')} elapsed"),
            "job_id": str(record.get("job_id")),
            "sidecar_sha256": sidecar_hash,
            "diagnostics": diagnostics,
            "autocon_loss_mean": _summary_series(logs, "autocon_loss", logs_path)["mean"],
        })
    expected_coordinates = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    if seen != expected_coordinates:
        raise ValueError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows


def _paired_test(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    if scipy_stats is not None and len(numbers) > 1:
        result = scipy_stats.ttest_1samp(numbers, 0.0)
        t_stat, p_value = float(result.statistic), float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat, p_value = mean / (sd / math.sqrt(len(numbers))), None
    else:
        t_stat, p_value = (0.0 if mean == 0 else math.copysign(math.inf, mean)), None
    return {
        "n": len(numbers), "mean": mean, "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat, "two_sided_p": p_value,
    }


def _build(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {(bool(row["fusion"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    pairs = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            off = indexed[(False, dataset, seed)]
            on = indexed[(True, dataset, seed)]
            if off["sidecar_sha256"] != on["sidecar_sha256"]:
                raise ValueError(f"sidecar differs between fusion arms: {dataset} seed {seed}")
            pairs.append({
                "dataset": dataset, "seed": seed,
                "off_mse": off["mse"], "on_mse": on["mse"],
                "delta_mse": on["mse"] - off["mse"],
                "relative_delta_mse": _relative(on["mse"] - off["mse"], off["mse"]),
                "off_mae": off["mae"], "on_mae": on["mae"],
                "delta_mae": on["mae"] - off["mae"],
                "relative_delta_mae": _relative(on["mae"] - off["mae"], off["mae"]),
                "off_elapsed_sec": off["elapsed_sec"], "on_elapsed_sec": on["elapsed_sec"],
                "sidecar_sha256_equal": True,
                "off_job_id": off["job_id"], "on_job_id": on["job_id"],
            })

    def contrast(metric: str, subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        deltas = [float(item[f"delta_{metric}"]) for item in subset]
        result = _paired_test(deltas)
        result["reference_mean"] = statistics.fmean(item[f"off_{metric}"] for item in subset)
        result["treatment_mean"] = statistics.fmean(item[f"on_{metric}"] for item in subset)
        result["relative_delta_mean"] = _relative(result["mean"], result["reference_mean"])
        return result

    by_dataset = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        by_dataset[dataset] = {"n": len(subset), "mse": contrast("mse", subset), "mae": contrast("mae", subset)}
    diagnostics = {}
    for fusion, label in ((False, "off"), (True, "on")):
        subset = [row for row in rows if bool(row["fusion"]) is fusion]
        diagnostics[label] = {
            key: {"mean_over_runs": statistics.fmean(row["diagnostics"][key]["mean"] for row in subset)}
            for key in RELATION_LOG_KEYS
        }
        diagnostics[label]["autocon_loss_mean_over_runs"] = statistics.fmean(row["autocon_loss_mean"] for row in subset)
    return {
        "schema_version": 1,
        "status": "exploratory_acf_representation_fusion_screen_not_confirmatory",
        "plan": str(plan["_path"]),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "phases": [name for name, _ in PHASES],
        "rows": list(rows), "pairs": pairs,
        "contrasts": {"fusion_on_minus_off": {"mse": contrast("mse", pairs), "mae": contrast("mae", pairs)}, "by_dataset": by_dataset},
        "diagnostics": diagnostics,
        "sidecar_pair_count": sum(item["sidecar_sha256_equal"] for item in pairs),
        "evidence_boundary": "Paired exploratory raw-source representation-fusion evidence; not confirmatory or publication-ready.",
    }


def _markdown(result: Mapping[str, Any]) -> str:
    contrast = result["contrasts"]["fusion_on_minus_off"]
    lines = [
        "# MambaACF ACF Representation Fusion Screen",
        "",
        "Status: **exploratory; not confirmatory**.",
        "",
        "Both arms use the same raw-source train-only sidecar, MambaACF variable-token AutoCon model, relation target, sampler, seeds, and budget. The only registered change is whether the ReZero ACF context adapter is enabled in the representation path.",
        "",
        "| Contrast (fusion on - off) | n | Mean delta | Relative delta | Wins lower | Two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in (("MSE", contrast["mse"]), ("MAE", contrast["mae"])):
        p = values.get("two_sided_p")
        lines.append(f"| {label} | {values['n']} | {values['mean']:.6g} | {values['relative_delta_mean']:.3%} | {values['wins_lower']} | {'n/a' if p is None else f'{p:.4g}'} |")
    lines += [
        "",
        "Negative deltas favor fusion. The result is descriptive and does not establish a general accuracy or speed claim.",
        "",
        "| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in result["contrasts"]["by_dataset"].items():
        lines.append(f"| {dataset} | {values['n']} | {values['mse']['mean']:.6g} | {values['mse']['relative_delta_mean']:.3%} | {values['mae']['mean']:.6g} | {values['mae']['relative_delta_mean']:.3%} |")
    lines += [
        "",
        "## Boundaries",
        "",
        "- This screen tests representation-path injection only; the forecast branch is unchanged by the ReZero adapter at initialization.",
        "- Raw ACF makes the adapter's signal transform strictly causal at the source level. Residual/trend exploratory screens do not have that guarantee.",
        "- It does not test longer horizons, parameter-matched PatchTST baselines, multi-lag balanced sampling, or end-to-end runtime.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    plan_path = _resolve(args.plan)
    plan = dict(_read_json(plan_path))
    plan["_path"] = str(plan_path)
    rows: List[Mapping[str, Any]] = []
    for phase_name, fusion in PHASES:
        rows.extend(_load_phase(plan, phase_name, fusion))
    result = _build(plan, rows)
    output_json, output_md = _resolve(args.output_json), _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pairs": len(result["pairs"]), "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
