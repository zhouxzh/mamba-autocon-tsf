#!/usr/bin/env python
"""Summarise screening results and resource-v2 measurements.

The command is intentionally descriptive.  It reads ``config.json`` together
with ``stream_metrics.json`` or ``metrics.npy`` below one or more explicitly
provided result roots, then writes a new analysis directory.  It never edits
the input roots and it does not select a model or a hyper-parameter.

Typical usage::

    python scripts/analyze_screening_matrix.py \
        --results-root artifacts/fair_pilot_20260831/results \
        --results-root artifacts/extension_pilot_20260831/traffic_h96 \
        --resource-root artifacts/resource_v2_screen \
        --baseline PatchTST --baseline DLinear \
        --output-dir artifacts/screening_analysis_20260831

``paired_delta`` is candidate minus baseline MSE, so a negative value is
descriptively lower error.  A pair is emitted as ``ambiguous`` rather than
silently choosing among duplicate coordinates.  One-seed or short-budget
aggregates are labelled ``exploratory``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_identity import canonical_model


SCHEMA_VERSION = 1
DEFAULT_BASELINE = "PatchTST"
DEFAULT_SHORT_BUDGET_EPOCHS = 3
METRIC_ORDER = ("mae", "mse", "rmse", "mape", "mspe")
RESOURCE_NUMERIC_FIELDS = (
    "parameter_total",
    "parameter_trainable",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p95",
    "peak_memory_mb",
    "throughput_sequences_per_sec",
)

# Fields describing the shared data/training/evaluation contract.  Model
# architecture and AutoCon-specific knobs are deliberately excluded: those
# are the treatment being compared, not a reason to reject a descriptive
# delta.  The resulting signature is used only to label comparability.
PROTOCOL_FIELDS = (
    "features",
    "target",
    "freq",
    "embed",
    "seq_len",
    "label_len",
    "pred_len",
    "train_epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "lradj",
    "min_learning_rate",
    "train_ratio",
    "drop_last_train",
    "num_workers",
    "use_amp",
    "stream_metrics",
    "origin_stride",
    "loss",
)


def _bool(value: Any, default: bool = False) -> bool:
    """Parse persisted CLI booleans without treating ``"false"`` as true."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _metric_from_stream(path: Path) -> Optional[float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    scaled = payload.get("scaled") if isinstance(payload, dict) else None
    if isinstance(scaled, dict):
        value = _finite(scaled.get("mse"))
        if value is not None:
            return value
    if isinstance(payload, dict):
        for key in ("scaled_mse", "mse"):
            value = _finite(payload.get(key))
            if value is not None:
                return value
    return None


def _metric_from_json(path: Path) -> Optional[float]:
    """Read a legacy metrics JSON variant when one is present."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    scaled = payload.get("scaled")
    if isinstance(scaled, dict):
        value = _finite(scaled.get("mse"))
        if value is not None:
            return value
    for key in ("scaled_mse", "mse"):
        value = _finite(payload.get(key))
        if value is not None:
            return value
    return None


def read_scaled_mse(result_dir: Path) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Read the preferred compact MSE artifact.

    Stream metrics take precedence because they are the current memory-safe
    artifact.  The optional third return value is a diagnostic when an
    artifact exists but cannot be parsed.
    """

    stream_path = result_dir / "stream_metrics.json"
    stream_issue = None
    if stream_path.is_file():
        value = _metric_from_stream(stream_path)
        if value is not None:
            return value, "stream_metrics.json", None
        # A partially written stream artifact should not hide a valid legacy
        # metrics.npy file left by the same run.  Keep the diagnostic only if
        # all fallbacks fail.
        stream_issue = "invalid_stream_metrics"

    metrics_path = result_dir / "metrics.npy"
    if metrics_path.is_file():
        try:
            values = np.asarray(np.load(metrics_path, allow_pickle=False), dtype=float).reshape(-1)
            if values.size >= 2:
                value = _finite(values[1])
                if value is not None:
                    return value, "metrics.npy", None
        except (OSError, ValueError, TypeError, EOFError):
            pass
        return None, "metrics.npy", stream_issue or "invalid_metrics_npy"

    # A few older experiments exported a JSON object instead of metrics.npy.
    for name in ("metrics.json", "metric.json"):
        path = result_dir / name
        if path.is_file():
            value = _metric_from_json(path)
            if value is not None:
                return value, name, None
            return None, name, stream_issue or "invalid_metrics_json"
    return None, None, stream_issue or "metric_artifact_missing"


def _epoch_value(config: Dict[str, Any]) -> Optional[int]:
    for key in ("train_epochs", "epochs", "max_epochs"):
        value = _integer(config.get(key))
        if value is not None:
            return value
    return None


def _phase_label(config: Dict[str, Any], root: Path) -> str:
    for key in ("phase", "plan_phase", "phase_name"):
        if config.get(key) not in (None, ""):
            return str(config[key])
    # ``des`` is often the only persisted human label for pilot phases.
    if config.get("des") not in (None, ""):
        return str(config["des"])
    return root.name


def _run_exploratory(config: Dict[str, Any], epochs: Optional[int], short_budget_epochs: int) -> bool:
    if _bool(config.get("exploratory", False)):
        return True
    if epochs is not None and epochs <= short_budget_epochs:
        return True
    label = " ".join(str(config.get(key, "")) for key in ("des", "phase", "model_id")).lower()
    return any(token in label for token in ("pilot", "smoke", "screen"))


def _required_coordinate(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[int]]:
    dataset = config.get("data", config.get("dataset"))
    features = config.get("features", "")
    seq_len = _integer(config.get("seq_len"))
    horizon = _integer(config.get("pred_len", config.get("horizon")))
    seed = _integer(config.get("run_seed", config.get("seed")))
    if dataset in (None, "") or seq_len is None or horizon is None or seed is None:
        return None, None, None, None, None
    return str(dataset), str(features), seq_len, horizon, seed


def discover_metric_rows(
    results_roots: Sequence[Path],
    short_budget_epochs: int = DEFAULT_SHORT_BUDGET_EPOCHS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Discover valid metric rows and non-fatal input issues.

    A config path is read at most once even if two roots overlap.  Invalid or
    incomplete artifacts are retained in ``issues`` for auditability and are
    excluded from numerical summaries.
    """

    rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    seen_configs = set()
    for raw_root in results_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            issues.append({"kind": "missing_results_root", "path": str(root)})
            continue
        for config_path in sorted(root.rglob("config.json")):
            config_path = config_path.resolve()
            if config_path in seen_configs:
                continue
            seen_configs.add(config_path)
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                issues.append({"kind": "invalid_config", "path": str(config_path), "detail": type(exc).__name__})
                continue
            if not isinstance(config, dict):
                issues.append({"kind": "invalid_config", "path": str(config_path), "detail": "not_object"})
                continue
            dataset, features, seq_len, horizon, seed = _required_coordinate(config)
            if dataset is None:
                issues.append({"kind": "incomplete_config", "path": str(config_path)})
                continue
            result_dir = config_path.parent
            mse, metric_source, metric_issue = read_scaled_mse(result_dir)
            if mse is None:
                issues.append({
                    "kind": metric_issue or "metric_unavailable",
                    "path": str(result_dir),
                    "config_path": str(config_path),
                })
                continue
            epochs = _epoch_value(config)
            rows.append({
                "dataset": dataset,
                "model": canonical_model(config),
                "features": features,
                "seq_len": seq_len,
                "horizon": horizon,
                "pred_len": horizon,
                "seed": seed,
                "scaled_mse": mse,
                "metric_source": metric_source,
                "train_epochs": epochs,
                "short_budget": bool(epochs is not None and epochs <= short_budget_epochs),
                "run_exploratory": _run_exploratory(config, epochs, short_budget_epochs),
                "phase": _phase_label(config, root),
                "source_root": str(root),
                "config_path": str(config_path),
                "result_dir": str(result_dir),
                "setting": config.get("setting"),
            })
    return rows, issues


def _core_key(row: Dict[str, Any], include_model: bool = True) -> Tuple[Any, ...]:
    values: Tuple[Any, ...] = (
        str(row.get("dataset", "")),
        str(row.get("features", "")),
        int(row.get("seq_len", 0)),
        int(row.get("horizon", row.get("pred_len", 0))),
        int(row.get("seed", 0)),
    )
    return values + ((str(row.get("model", "")),) if include_model else ())


def _join_values(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in sorted(set(values), key=str))


def _group_rows(rows: Sequence[Dict[str, Any]], include_model: bool = True) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_core_key(row, include_model=include_model)].append(row)
    return grouped


def summarize_seed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate config artifacts to one auditable seed coordinate."""

    result: List[Dict[str, Any]] = []
    for key, group in sorted(_group_rows(rows).items(), key=lambda item: item[0]):
        dataset, features, seq_len, horizon, seed, model = key
        values = [float(item["scaled_mse"]) for item in group if _finite(item.get("scaled_mse")) is not None]
        unique_values = []
        for value in values:
            if not any(math.isclose(value, old, rel_tol=1e-10, abs_tol=1e-12) for old in unique_values):
                unique_values.append(value)
        # If duplicate artifacts disagree, do not manufacture a point estimate.
        mse = unique_values[0] if len(unique_values) == 1 else None
        result.append({
            "dataset": dataset,
            "model": model,
            "features": features,
            "seq_len": seq_len,
            "horizon": horizon,
            "pred_len": horizon,
            "seed": seed,
            "scaled_mse": mse,
            "n_runs": len(group),
            "duplicate_count": max(0, len(group) - 1),
            "duplicate_conflict": bool(len(unique_values) > 1),
            "metric_sources": _join_values(item.get("metric_source") for item in group),
            "source_roots": _join_values(item.get("source_root") for item in group),
            "phases": _join_values(item.get("phase") for item in group),
            "train_epochs_min": min((item["train_epochs"] for item in group if item.get("train_epochs") is not None), default=None),
            "train_epochs_max": max((item["train_epochs"] for item in group if item.get("train_epochs") is not None), default=None),
            "short_budget": any(bool(item.get("short_budget")) for item in group),
            # Every seed row represents one seed by construction; this explicit
            # label prevents a single-seed pilot from being mistaken for a
            # replicated estimate in downstream tables.
            "one_seed": True,
            "run_exploratory": any(bool(item.get("run_exploratory")) for item in group),
            "exploratory": True,
            "config_paths": _join_values(item.get("config_path") for item in group),
        })
    return result


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    finite = np.asarray([value for value in values if _finite(value) is not None], dtype=float)
    if finite.size == 0:
        return None, None
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def summarize_models(seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate seed rows without ranking or selecting a model."""

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = (
            row["dataset"], row["model"], row["features"], row["seq_len"], row["horizon"]
        )
        grouped[key].append(row)
    result: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        dataset, model, features, seq_len, horizon = key
        valid = [float(row["scaled_mse"]) for row in group if _finite(row.get("scaled_mse")) is not None]
        mean, std = _mean_std(valid)
        n_seeds = len({int(row["seed"]) for row in group if _finite(row.get("scaled_mse")) is not None})
        short_budget = any(bool(row.get("short_budget")) for row in group)
        result.append({
            "dataset": dataset,
            "model": model,
            "features": features,
            "seq_len": seq_len,
            "horizon": horizon,
            "pred_len": horizon,
            "n_seeds": n_seeds,
            "seeds": _join_values(row["seed"] for row in group),
            "n_runs": sum(int(row.get("n_runs", 1)) for row in group),
            "duplicate_seed_count": sum(int(row.get("duplicate_count", 0)) for row in group),
            "duplicate_conflict": any(bool(row.get("duplicate_conflict")) for row in group),
            "scaled_mse_mean": mean,
            "scaled_mse_std": std,
            "scaled_mse_median": float(np.median(valid)) if valid else None,
            "scaled_mse_min": min(valid) if valid else None,
            "scaled_mse_max": max(valid) if valid else None,
            "short_budget": short_budget,
            "one_seed": n_seeds <= 1,
            "exploratory": bool(
                n_seeds <= 1
                or short_budget
                or any(row.get("run_exploratory", False) for row in group)
            ),
            "source_roots": _join_values(
                root for row in group for root in str(row.get("source_roots", "")).split(",") if root
            ),
            "phases": _join_values(
                phase for row in group for phase in str(row.get("phases", "")).split(",") if phase
            ),
        })
    return result


def _canonical_baselines(baselines: Sequence[str]) -> List[str]:
    values = []
    for baseline in baselines:
        name = canonical_model(str(baseline).strip())
        if name and name not in values:
            values.append(name)
    return values or [DEFAULT_BASELINE]


def paired_deltas(seed_rows: Sequence[Dict[str, Any]], baselines: Sequence[str]) -> List[Dict[str, Any]]:
    """Compute candidate-minus-baseline deltas at matched seed coordinates."""

    baseline_names = _canonical_baselines(baselines)
    grouped = _group_rows(seed_rows)
    by_pair_key: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for key, group in grouped.items():
        dataset, features, seq_len, horizon, seed, model = key
        by_pair_key[(dataset, features, seq_len, horizon, seed, model)].extend(group)

    candidate_groups = {}
    for key, group in by_pair_key.items():
        model = key[-1]
        if model not in baseline_names:
            candidate_groups[key] = group

    rows: List[Dict[str, Any]] = []
    for candidate_key, candidate_group in sorted(candidate_groups.items(), key=lambda item: item[0]):
        dataset, features, seq_len, horizon, seed, candidate_model = candidate_key
        for baseline in baseline_names:
            baseline_key = (dataset, features, seq_len, horizon, seed, baseline)
            baseline_group = by_pair_key.get(baseline_key, [])
            candidate_values = [row.get("scaled_mse") for row in candidate_group if _finite(row.get("scaled_mse")) is not None]
            baseline_values = [row.get("scaled_mse") for row in baseline_group if _finite(row.get("scaled_mse")) is not None]
            status = "paired"
            candidate_ambiguous = (
                len(candidate_group) != 1
                or len(candidate_values) != 1
                or any(int(row.get("n_runs", 1)) != 1 for row in candidate_group)
            )
            baseline_ambiguous = (
                len(baseline_group) != 1
                or len(baseline_values) != 1
                or any(int(row.get("n_runs", 1)) != 1 for row in baseline_group)
            )
            if candidate_ambiguous:
                status = (
                    "ambiguous_candidate"
                    if len(candidate_group) > 1
                    or any(int(row.get("n_runs", 1)) != 1 for row in candidate_group)
                    else "invalid_candidate"
                )
            elif not baseline_group:
                status = "missing_baseline"
            elif baseline_ambiguous:
                status = "ambiguous_baseline"
            candidate_mse = float(candidate_values[0]) if len(candidate_values) == 1 else None
            baseline_mse = float(baseline_values[0]) if len(baseline_values) == 1 else None
            delta = (
                candidate_mse - baseline_mse
                if status == "paired" and candidate_mse is not None and baseline_mse is not None
                else None
            )
            relative = delta / baseline_mse if delta is not None and baseline_mse not in (None, 0.0) else None
            all_rows = candidate_group + baseline_group
            rows.append({
                "dataset": dataset,
                "candidate_model": candidate_model,
                "baseline_model": baseline,
                "features": features,
                "seq_len": seq_len,
                "horizon": horizon,
                "pred_len": horizon,
                "seed": seed,
                "status": status,
                "candidate_scaled_mse": candidate_mse,
                "baseline_scaled_mse": baseline_mse,
                "paired_delta": delta,
                "relative_delta": relative,
                "candidate_runs": len(candidate_group),
                "baseline_runs": len(baseline_group),
                "short_budget": any(bool(row.get("short_budget")) for row in all_rows),
                "exploratory": True,
            })
    return rows


def summarize_paired_deltas(delta_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        key = (
            row["dataset"], row["candidate_model"], row["baseline_model"],
            row["features"], row["seq_len"], row["horizon"],
        )
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        valid = [float(row["paired_delta"]) for row in group if row.get("status") == "paired" and _finite(row.get("paired_delta")) is not None]
        relative = [float(row["relative_delta"]) for row in group if row.get("status") == "paired" and _finite(row.get("relative_delta")) is not None]
        mean, std = _mean_std(valid)
        rel_mean, rel_std = _mean_std(relative)
        status_counts = Counter(str(row.get("status")) for row in group)
        n_pairs = len(valid)
        result.append({
            "dataset": key[0],
            "candidate_model": key[1],
            "baseline_model": key[2],
            "features": key[3],
            "seq_len": key[4],
            "horizon": key[5],
            "n_pairs": n_pairs,
            "paired_seeds": _join_values(row["seed"] for row in group if row.get("status") == "paired"),
            "paired_delta_mean": mean,
            "paired_delta_std": std,
            "relative_delta_mean": rel_mean,
            "relative_delta_std": rel_std,
            "status_counts": ";".join(f"{name}:{count}" for name, count in sorted(status_counts.items())),
            "short_budget": any(bool(row.get("short_budget")) for row in group),
            "one_seed": n_pairs <= 1,
            "exploratory": bool(n_pairs < 2 or any(bool(row.get("short_budget")) for row in group)),
        })
    return result


def _resource_record_rows(payload: Dict[str, Any], path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    records = payload.get("resource_records") if isinstance(payload, dict) else None
    if records is None and isinstance(payload, dict) and "model" in payload:
        records = [payload]
    if not isinstance(records, list):
        return [], "resource_records_missing"
    protocol = payload.get("benchmark_protocol", {}) if isinstance(payload, dict) else {}
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        architecture = record.get("architecture") if isinstance(record.get("architecture"), dict) else {}
        model_config = dict(architecture)
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        # Resource records retain the validated runner job id under ``source``
        # rather than duplicating it in ``architecture``.  Include it when
        # available so a variable-token resource measurement joins the same
        # reporting arm as its forecast metrics.
        if "plan_job_id" not in model_config and source.get("job_id") not in (None, ""):
            model_config["plan_job_id"] = source["job_id"]
        model_config.update({
            key: record[key]
            for key in ("model", "AutoCon", "plan_job_id", "mamba_variable_mixer")
            if key in record
        })
        model = canonical_model(model_config)
        dataset = record.get("dataset", architecture.get("dataset", ""))
        features = record.get("features", architecture.get("features", ""))
        seq_len = _integer(record.get("seq_len", architecture.get("seq_len")))
        horizon = _integer(record.get("pred_len", architecture.get("pred_len")))
        parameters = record.get("parameters", {}) if isinstance(record.get("parameters"), dict) else {}
        gpu = record.get("gpu_inference") if isinstance(record.get("gpu_inference"), dict) else {}
        if not gpu:
            gpu = record.get("cpu_inference") if isinstance(record.get("cpu_inference"), dict) else {}
        memory_field = "steady_state_peak_allocated_mb"
        memory = _finite(gpu.get(memory_field))
        if memory is None:
            memory_field = "steady_state_peak_reserved_mb"
            memory = _finite(gpu.get(memory_field))
        rows.append({
            "model": model,
            "dataset": str(dataset),
            "features": str(features),
            "seq_len": seq_len,
            "horizon": horizon,
            "pred_len": horizon,
            "parameter_total": _finite(parameters.get("total")),
            "parameter_trainable": _finite(parameters.get("trainable")),
            "latency_ms_mean": _finite(gpu.get("latency_ms_mean")),
            "latency_ms_median": _finite(gpu.get("latency_ms_median")),
            "latency_ms_p95": _finite(gpu.get("latency_ms_p95")),
            "peak_memory_mb": memory,
            "memory_field": memory_field if memory is not None else None,
            "throughput_sequences_per_sec": _finite(gpu.get("synthetic_input_sequences_per_sec")),
            "device": gpu.get("device", protocol.get("device")),
            "benchmark_batch_size": _integer(gpu.get("benchmark_batch_size", protocol.get("benchmark_batch_size"))),
            "timed_forwards": _integer(gpu.get("timed_forwards", protocol.get("timed_forwards"))),
            "resource_file": str(path.resolve()),
            "resource_index": index,
            "resource_signature": record.get("resource_signature_sha256"),
        })
    return rows, None


def discover_resource_rows(resource_jsons: Sequence[Path], resource_roots: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    paths = []
    seen = set()
    for raw_path in list(resource_jsons):
        path = Path(raw_path).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            paths.append(path)
    for raw_root in resource_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("model_resources.json")):
            path = path.resolve()
            if path not in seen:
                seen.add(path)
                paths.append(path)
    rows, issues = [], []
    for path in paths:
        if not path.is_file():
            issues.append({"kind": "missing_resource_json", "path": str(path)})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            issues.append({"kind": "invalid_resource_json", "path": str(path), "detail": type(exc).__name__})
            continue
        parsed, issue = _resource_record_rows(payload, path)
        rows.extend(parsed)
        if issue:
            issues.append({"kind": issue, "path": str(path)})
    return rows, issues


def summarize_resources(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("dataset", ""), row.get("model", ""), row.get("features", ""), row.get("seq_len"), row.get("horizon"))
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        output = {
            "dataset": key[0], "model": key[1], "features": key[2], "seq_len": key[3], "horizon": key[4], "pred_len": key[4],
            "n_records": len(group),
            "resource_files": _join_values(row.get("resource_file") for row in group),
            "devices": _join_values(row.get("device") for row in group if row.get("device") not in (None, "")),
            "batch_sizes": _join_values(row.get("benchmark_batch_size") for row in group if row.get("benchmark_batch_size") is not None),
            "measurement_only": True,
        }
        for field in RESOURCE_NUMERIC_FIELDS:
            values = [row[field] for row in group if _finite(row.get(field)) is not None]
            mean, std = _mean_std(values)
            output[f"{field}_mean"] = mean
            output[f"{field}_std"] = std
        output["protocol_consistent"] = len({(row.get("device"), row.get("benchmark_batch_size"), row.get("timed_forwards")) for row in group}) <= 1
        result.append(output)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_safe(row.get(field)) for field in (fields or ["empty"])})


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], limit: int) -> List[str]:
    shown = list(rows[:limit])
    if not shown:
        return ["（无可用记录。）"]
    def cell(value: Any) -> str:
        if value is None:
            return ""
        text = str(_json_safe(value)).replace("|", "\\|").replace("\n", " ")
        if isinstance(value, float):
            return f"{value:.6g}"
        return text
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in shown)
    if len(rows) > limit:
        lines.append(f"\n仅显示前 {limit} 行；完整记录见 CSV/JSON。")
    return lines


def render_markdown(payload: Dict[str, Any], limit: int = 200) -> str:
    counts = payload.get("counts", {})
    lines = [
        "# Screening matrix analysis",
        "",
        "本报告只做描述性汇总，不进行模型、超参数或结果目录选择。paired delta 定义为候选模型 scaled MSE 减去基线 scaled MSE；负值表示候选误差较低。",
        "",
        f"- 结果根目录：{len(payload.get('results_roots', []))}",
        f"- 有效训练记录：{counts.get('metric_rows', 0)}；seed 坐标：{counts.get('seed_rows', 0)}；模型聚合：{counts.get('model_rows', 0)}",
        f"- paired delta：{counts.get('delta_rows', 0)}（有效配对汇总 {counts.get('paired_summary_rows', 0)}）",
        f"- 资源记录：{counts.get('resource_rows', 0)}；资源聚合：{counts.get('resource_summary_rows', 0)}",
        "",
        "`exploratory=true` 表示只有一个有效 seed、预算不超过阈值，或配置显式/标签表明是 pilot/screen/smoke；这类数字不能作为确认性结论。资源测量是 synthetic forward benchmark，不等价于端到端训练速度。",
        "",
        "## Model summaries",
        "",
    ]
    lines.extend(_markdown_table(payload.get("model_summary", []), ["dataset", "model", "horizon", "n_seeds", "scaled_mse_mean", "scaled_mse_std", "exploratory"], limit))
    lines.extend(["", "## Paired deltas", ""])
    lines.extend(_markdown_table(payload.get("paired_summary", []), ["dataset", "candidate_model", "baseline_model", "horizon", "n_pairs", "paired_delta_mean", "relative_delta_mean", "exploratory"], limit))
    lines.extend(["", "## Resource summaries", ""])
    lines.extend(_markdown_table(payload.get("resource_summary", []), ["dataset", "model", "horizon", "parameter_total_mean", "latency_ms_mean_mean", "peak_memory_mb_mean", "protocol_consistent"], limit))
    if payload.get("issues"):
        lines.extend(["", "## Input issues", "", f"发现 {len(payload['issues'])} 个不可计入汇总的输入问题；详情见 `issues.csv`。"])
    return "\n".join(lines) + "\n"


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def build_payload(
    results_roots: Sequence[Path],
    resource_jsons: Sequence[Path],
    resource_roots: Sequence[Path],
    baselines: Sequence[str],
    short_budget_epochs: int = DEFAULT_SHORT_BUDGET_EPOCHS,
) -> Dict[str, Any]:
    raw_rows, result_issues = discover_metric_rows(results_roots, short_budget_epochs)
    seed_rows = summarize_seed_rows(raw_rows)
    model_rows = summarize_models(seed_rows)
    baseline_names = _canonical_baselines(baselines)
    delta_rows = paired_deltas(seed_rows, baseline_names)
    paired_summary = summarize_paired_deltas(delta_rows)
    resource_rows, resource_issues = discover_resource_rows(resource_jsons, resource_roots)
    resource_summary = summarize_resources(resource_rows)
    issues = result_issues + resource_issues
    return _json_safe({
        "schema_version": SCHEMA_VERSION,
        "analysis": "screening_matrix_descriptive",
        "results_roots": [str(Path(root).expanduser().resolve()) for root in results_roots],
        "resource_jsons": [str(Path(path).expanduser().resolve()) for path in resource_jsons],
        "resource_roots": [str(Path(root).expanduser().resolve()) for root in resource_roots],
        "baselines": baseline_names,
        "short_budget_epochs": short_budget_epochs,
        "counts": {
            "metric_rows": len(raw_rows), "seed_rows": len(seed_rows), "model_rows": len(model_rows),
            "delta_rows": len(delta_rows), "paired_summary_rows": len(paired_summary),
            "resource_rows": len(resource_rows), "resource_summary_rows": len(resource_summary), "issues": len(issues),
        },
        "raw_results": raw_rows,
        "seed_summary": seed_rows,
        "model_summary": model_rows,
        "paired_deltas": delta_rows,
        "paired_summary": paired_summary,
        "resource_rows": resource_rows,
        "resource_summary": resource_summary,
        "issues": issues,
    })


def write_outputs(payload: Dict[str, Any], output_dir: Path, input_roots: Sequence[Path], overwrite: bool = False, markdown_limit: int = 200) -> Path:
    """Write all analysis artifacts, refusing accidental input-root writes."""

    output = Path(output_dir).expanduser().resolve()
    input_paths = [Path(root).expanduser().resolve() for root in input_roots]
    if any(_path_is_under(output, root) for root in input_paths):
        raise ValueError("output directory must not be equal to or below a supplied input root")
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "screening_matrix.json": json.dumps(_json_safe(payload), ensure_ascii=True, indent=2) + "\n",
        "screening_matrix.md": render_markdown(payload, limit=markdown_limit),
    }
    csv_rows = {
        "raw_results.csv": payload.get("raw_results", []),
        "seed_summary.csv": payload.get("seed_summary", []),
        "model_summary.csv": payload.get("model_summary", []),
        "paired_deltas.csv": payload.get("paired_deltas", []),
        "paired_summary.csv": payload.get("paired_summary", []),
        "resource_rows.csv": payload.get("resource_rows", []),
        "resource_summary.csv": payload.get("resource_summary", []),
        "issues.csv": payload.get("issues", []),
    }
    targets = [output / name for name in files]
    targets.extend(output / name for name in csv_rows)
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing output files: " + ", ".join(existing))
    for path, text in files.items():
        (output / path).write_text(text, encoding="utf-8")
    for name, rows in csv_rows.items():
        _write_csv(output / name, rows, overwrite=True)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True, help="phase result root; repeat for multiple phases")
    parser.add_argument("--resource-json", type=Path, action="append", default=[], help="explicit resource_v2 model_resources.json; repeatable")
    parser.add_argument("--resource-root", type=Path, action="append", default=[], help="root searched recursively for model_resources.json; repeatable")
    parser.add_argument("--baseline", action="append", default=[], help="baseline model name; repeatable (default: PatchTST)")
    parser.add_argument("--short-budget-epochs", type=int, default=DEFAULT_SHORT_BUDGET_EPOCHS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--markdown-limit", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace existing analysis files")
    args = parser.parse_args(argv)
    baselines = args.baseline or [DEFAULT_BASELINE]
    try:
        payload = build_payload(args.results_root, args.resource_json, args.resource_root, baselines, args.short_budget_epochs)
        output = write_outputs(payload, args.output_dir, list(args.results_root) + list(args.resource_json) + list(args.resource_root), args.overwrite, args.markdown_limit)
    except (OSError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(
        f"Analysed {payload['counts']['metric_rows']} metric rows, "
        f"{payload['counts']['delta_rows']} paired rows and "
        f"{payload['counts']['resource_rows']} resource rows into {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
