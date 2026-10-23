#!/usr/bin/env python3
"""Strict readout for the exploratory MambaACF fair-baseline screen.

The reader audits completed phase artifacts before calculating descriptive
paired contrasts.  It deliberately keeps external baseline comparisons
separate from the mechanism pairings (memory -> relation -> calibration),
never selects a checkpoint from test metrics, and never changes artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# When invoked as ``python scripts/analyze_*.py``, Python puts only the
# scripts directory on ``sys.path``.  Add the repository root before loading
# sibling analyzers so both direct and module-style calls share one import
# contract.
_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional reporting dependency
    scipy_stats = None

try:
    from scripts.analyze_mamba_acf_horizon_calibration_screen import (
        _calibration_contract,
        _canonical,
        _checkpoint_state,
        _load_sidecar,
    )
    from scripts.analyze_mamba_relation_null_control import (
        _finite,
        _hash_map_for_record,
        _manifest_records,
        _read_json,
        _resolve,
        _sha256,
        _summary_series,
    )
except ModuleNotFoundError:  # Support direct ``python scripts/...`` calls.
    from analyze_mamba_acf_horizon_calibration_screen import (  # type: ignore
        _calibration_contract,
        _canonical,
        _checkpoint_state,
        _load_sidecar,
    )
    from analyze_mamba_relation_null_control import (  # type: ignore
        _finite,
        _hash_map_for_record,
        _manifest_records,
        _read_json,
        _resolve,
        _sha256,
        _summary_series,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
BASELINE_MODELS = ("DLinear", "PatchTST", "Mamba1", "Mamba2")
ACF_PHASES = (
    ("acf_memory_h96", "memory", "MambaACF", False, False),
    ("acf_memory_relation_h96", "relation", "MambaACFAutoCon", True, False),
    (
        "acf_memory_relation_calibrated_h96",
        "calibrated",
        "MambaACFAutoCon",
        True,
        True,
    ),
)
BASELINE_PHASE = ("plain_baselines_h96", "plain")

RELATION_LOG_KEYS = (
    "acf_relation_global_valid_fraction",
    "acf_relation_global_active_anchor_fraction",
    "acf_relation_global_candidate_fraction",
    "acf_relation_global_candidate_active_anchor_fraction",
    "acf_relation_global_positive_count_mean",
    "acf_relation_global_high_acf_excluded_fraction",
    "acf_relation_global_high_acf_candidate_count",
    "acf_relation_global_high_acf_excluded_count",
    "acf_relation_global_high_acf_conditional_excluded_fraction",
    "acf_relation_global_sign_target_count",
    "acf_relation_global_sign_accuracy",
    "acf_relation_global_sign_hinge",
)
RUNTIME_LOG_KEYS = (
    "acf_runtime_gate_mean",
    "acf_runtime_gate_std",
    "acf_seasonal_rms",
    "acf_reliability_mean",
    "acf_effective_memory_rms",
)
CALIBRATION_LOG_KEYS = (
    "acf_calibration_amp_mean",
    "acf_calibration_amp_std",
    "acf_calibration_amp_min",
    "acf_calibration_amp_max",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{label} is not an explicit boolean: {value!r}")


def _int_list(value: Any, label: str, *, nonempty: bool = True) -> List[int]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{label} must be a non-empty list")
    result: List[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} contains a boolean")
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-integer") from exc
        if parsed <= 0 or parsed != item:
            raise ValueError(f"{label} contains an invalid value")
        result.append(parsed)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate values")
    return result


def _optional_series(logs: Mapping[str, Any], key: str, path: Path, *, required: bool) -> Dict[str, Any]:
    values = logs.get(key)
    if values is None and not required:
        return {"n": 0, "finite_n": 0, "mean": None, "min": None, "max": None, "last": None}
    if not isinstance(values, list) or not values:
        raise ValueError(f"missing diagnostic {key}: {path}")
    numbers: List[float] = []
    null_count = 0
    for value in values:
        if value is None:
            null_count += 1
        else:
            numbers.append(_finite(value, f"{path.name}:{key}"))
    if required and (not numbers or null_count):
        raise ValueError(f"diagnostic {key} has null/non-numeric epochs: {path}")
    return {
        "n": len(values),
        "finite_n": len(numbers),
        "null_n": null_count,
        "mean": statistics.fmean(numbers) if numbers else None,
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
        "last": numbers[-1] if numbers else None,
    }


def _state_numel(state: Mapping[str, Any]) -> int:
    """Return a transparent checkpoint-state size, not a claimed trainable count."""
    total = 0
    for value in state.values():
        numel = getattr(value, "numel", None)
        if callable(numel):
            total += int(numel())
        else:
            shape = getattr(value, "shape", None)
            if shape is not None:
                total += math.prod(int(item) for item in shape)
    return int(total)


def _config_value(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def _require_config(config: Mapping[str, Any], expected: Mapping[str, Any], job_id: str) -> None:
    for key, wanted in expected.items():
        if key not in config:
            raise ValueError(f"config {key} is missing for {job_id}")
        actual = config[key]
        if isinstance(wanted, bool):
            actual = _bool(actual, f"config {key} for {job_id}")
        elif isinstance(wanted, int) and not isinstance(wanted, bool):
            try:
                actual = int(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} is not an integer for {job_id}") from exc
        elif isinstance(wanted, float):
            try:
                if not math.isclose(float(actual), wanted, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} mismatch for {job_id}") from exc
            continue
        if actual != wanted:
            raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {wanted!r}")


def _shared_acf_signature(config: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Fields that must remain invariant through the staged ACF contrast."""
    keys = (
        "features", "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out",
        "d_model", "d_ff", "e_layers", "n_heads", "d_layers", "factor", "dropout",
        "embed", "activation", "mamba_d_state", "mamba_d_conv", "mamba_expand",
        "mamba_headdim", "mamba_scales", "mamba_token_stride", "mamba_bidirectional",
        "mamba_causal_embedding", "mamba_channel_independent", "mamba_channel_gate",
        "mamba_variable_mixer", "mamba_variable_group_size", "mamba_acf_channel_order",
        "mamba_acf_channel_conditioning", "mamba_acf_channel_gate", "mamba_acf_repr_fusion",
        "mamba_patch_mixer", "mamba_patch_order", "mamba_local_mixer", "mamba_low_rank_head",
        "mamba_temporal_rank", "mamba_acf_lags", "mamba_acf_lag_mode", "mamba_acf_max_lag",
        "mamba_acf_top_k", "mamba_acf_min_lag", "mamba_acf_min_abs", "mamba_acf_confidence_z",
        "mamba_acf_min_separation", "mamba_acf_detrend", "mamba_acf_chunk", "mamba_use_acf_memory",
        "mamba_dynamic_acf", "mamba_acf_temperature", "mamba_acf_step_reliability",
        "AutoCon_acf_source", "batch_size", "train_epochs", "patience", "learning_rate",
        "lradj", "min_learning_rate", "weight_decay", "num_workers", "drop_last_train",
        "train_ratio", "use_amp", "mamba_acf_horizon_calibration_rank",
    )
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"staged ACF config fields missing: {missing}")
    return tuple(_canonical(config[key]) for key in keys)


def _validate_baseline_config(config: Mapping[str, Any], model: str, dataset: str, job_id: str) -> Dict[str, Any]:
    channels = 7 if dataset == "ETTh1" else 862
    _require_config(
        config,
        {
            "model": model,
            "features": "M",
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": channels,
            "dec_in": channels,
            "c_out": channels,
            "AutoCon": False,
            "mamba_use_acf_memory": False,
            "mamba_acf_relation_loss": False,
            "mamba_acf_relation_min_lag": 0,
            "mamba_acf_horizon_calibration": False,
        },
        job_id,
    )
    return {
        "signature": tuple(
            _canonical(config.get(key))
            for key in (
                "features", "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out",
                "d_model", "d_ff", "e_layers", "n_heads", "dropout", "batch_size", "train_epochs",
                "patience", "learning_rate", "lradj", "weight_decay",
            )
        )
    }


def _validate_acf_config(
    config: Mapping[str, Any],
    model: str,
    dataset: str,
    job_id: str,
    relation: bool,
    calibrated: bool,
) -> Dict[str, Any]:
    channels = 7 if dataset == "ETTh1" else 862
    expected = {
        "model": model,
        "features": "M",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "AutoCon": relation,
        "mamba_causal_embedding": True,
        "mamba_use_acf_memory": True,
        "mamba_dynamic_acf": False,
        "mamba_acf_lag_mode": "train_peaks_available",
        "mamba_acf_relation_loss": relation,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_relation_min_lag": 96 if relation else 0,
        "mamba_acf_horizon_calibration": calibrated,
        "mamba_acf_horizon_calibration_rank": 2,
        "mamba_variable_mixer": False,
        "mamba_variable_group_size": 1,
        "AutoCon_acf_source": "raw",
        "train_epochs": 8,
        "patience": 3,
    }
    if relation:
        expected.update({"AutoCon_lambda": 0.03, "mamba_acf_relation_lag_mask": True})
    else:
        expected.update({"mamba_acf_relation_lag_mask": False})
    _require_config(config, expected, job_id)
    requested = _int_list(config.get("mamba_acf_prior_lags"), "mamba_acf_prior_lags")
    effective = _int_list(config.get("mamba_acf_effective_lags"), "mamba_acf_effective_lags")
    prior = _int_list(config.get("mamba_acf_prior_lags"), "mamba_acf_prior_lags")
    if sorted(effective) != sorted(prior):
        raise ValueError(f"effective/prior lag mismatch for {job_id}")
    if len(effective) != 3:
        raise ValueError(f"expected three ACF lag slots for {job_id}, got {effective}")
    if str(config.get("mamba_acf_effective_lag_source")) != "train_peaks_available":
        raise ValueError(f"ACF lag source mismatch for {job_id}")
    if _bool(config.get("mamba_acf_effective_lag_fallback"), "lag fallback"):
        raise ValueError(f"ACF lag fallback used for {job_id}")
    relation_lags = []
    if relation:
        relation_lags = _int_list(config.get("mamba_acf_relation_effective_lags"), "relation lags")
        if any(lag < 96 for lag in relation_lags):
            raise ValueError(f"relation lag below 96 for {job_id}")
        if str(config.get("mamba_acf_relation_lag_source")) != "train_peak_subset":
            raise ValueError(f"relation lag source mismatch for {job_id}")
        if _bool(config.get("mamba_acf_relation_lag_fallback"), "relation fallback"):
            raise ValueError(f"relation lag fallback used for {job_id}")
    prior_shape = config.get("mamba_acf_prior_shape")
    if not isinstance(prior_shape, list) or len(prior_shape) != 2:
        raise ValueError(f"invalid prior shape for {job_id}")
    if [int(item) for item in prior_shape][1] != len(effective):
        raise ValueError(f"prior width mismatch for {job_id}")
    prior_sha = str(config.get("mamba_acf_prior_sha256", ""))
    if len(prior_sha) != 64 or any(char not in "0123456789abcdefABCDEF" for char in prior_sha):
        raise ValueError(f"invalid prior hash for {job_id}")
    return {
        "requested_lags": requested,
        "effective_lags": effective,
        "relation_lags": relation_lags,
        "prior_sha256": prior_sha,
        "signature": _shared_acf_signature(config),
    }


def _phase_from_plan(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    phase = next((item for item in phases if isinstance(item, Mapping) and item.get("name") == name), None)
    if phase is None:
        raise ValueError(f"plan has no phase {name!r}")
    return phase


def _load_phase(plan: Mapping[str, Any], name: str, kind: str, model_names: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    phase = _phase_from_plan(plan, name)
    root = _resolve(str(phase.get("output_root", "")))
    validation_path = root / "phase_validation.json"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != name:
        raise ValueError(f"invalid phase validation: {validation_path}")
    records = _manifest_records(root, validation)
    result_root, checkpoint_root = root / "results", root / "checkpoints"
    expected_models = tuple(model_names or ())
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, str]] = set()
    for record in records:
        if record.get("status") != "ok" or record.get("phase") != name:
            raise ValueError(f"unsuccessful record in {name}: {record.get('job_id')}")
        dataset, seed, model = str(record.get("dataset")), int(record.get("seed")), str(record.get("model"))
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected coordinate in {name}: {dataset}/{seed}")
        if expected_models and model not in expected_models:
            raise ValueError(f"unexpected model in {name}: {model}")
        coordinate = (dataset, seed, model)
        if coordinate in seen:
            raise ValueError(f"duplicate coordinate in {name}: {coordinate}")
        seen.add(coordinate)
        job_id = str(record.get("job_id"))
        metrics_ref = record.get("expected_artifact")
        if not metrics_ref:
            raise ValueError(f"missing metrics artifact for {job_id}")
        metrics_path = _resolve(str(metrics_ref))
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not _inside(metrics_path, result_root):
            raise ValueError(f"metrics artifact outside phase result root: {job_id}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        config = _read_json(result_dir / "config.json")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {job_id}")
        if int(stream.get("pred_len", -1)) != EXPECTED_HORIZON:
            raise ValueError(f"stream horizon mismatch: {job_id}")
        mse, mae = _finite(scaled.get("mse"), f"{job_id} scaled mse"), _finite(scaled.get("mae"), f"{job_id} scaled mae")
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        state = _checkpoint_state(checkpoint_path)
        row: Dict[str, Any] = {
            "phase": name, "kind": kind, "dataset": dataset, "seed": seed, "model": model,
            "job_id": job_id, "mse": mse, "mae": mae,
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "epochs_ran": int(record.get("epochs_ran")), "best_epoch": int(record.get("best_epoch")),
            "state_numel": _state_numel(state), "artifact_hashes": dict(hashes),
        }
        if kind == "plain":
            row["config_info"] = _validate_baseline_config(config, model, dataset, job_id)
            row["sidecar_sha256"] = None
            row["effective_lags"] = []
            row["relation_lags"] = []
            row["runtime"] = {}
            row["relation_diagnostics"] = {}
            row["calibration"] = {key: None for key in CALIBRATION_LOG_KEYS}
        else:
            relation = kind in {"relation", "calibrated"}
            calibrated = kind == "calibrated"
            info = _validate_acf_config(config, model, dataset, job_id, relation, calibrated)
            prior_path = result_dir / "acf_prior.npz"
            if "acf_prior.npz" not in hashes or info["prior_sha256"] != hashes["acf_prior.npz"]:
                raise ValueError(f"sidecar hash mismatch for {job_id}")
            sidecar = _load_sidecar(prior_path, config, job_id)
            logs_path = checkpoint_dir / "loss_logs.json"
            logs = _read_json(logs_path)
            # The plain MambaACF forecasting loop predates runtime telemetry;
            # AutoCon arms log it through their extended loss recorder.  Keep
            # this absence visible rather than rejecting an otherwise valid
            # static-memory result, while retaining the strict requirement for
            # relation/calibration runs where the telemetry is available.
            runtime = {
                key: _optional_series(logs, key, logs_path, required=relation)
                for key in RUNTIME_LOG_KEYS
            }
            relation_diagnostics = {}
            if relation:
                if logs.get("acf_relation_objective_version") != 3:
                    raise ValueError(f"relation objective version is not 3: {job_id}")
                relation_diagnostics = {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS}
            calibration = {
                key: _optional_series(logs, key, logs_path, required=calibrated)
                for key in CALIBRATION_LOG_KEYS
            }
            state_info = _calibration_contract(state, checkpoint_path, calibrated, len(info["effective_lags"]))
            row.update({
                "config_info": info, "sidecar_sha256": hashes["acf_prior.npz"],
                "sidecar": sidecar, "effective_lags": info["effective_lags"],
                "relation_lags": info["relation_lags"], "runtime": runtime,
                "relation_diagnostics": relation_diagnostics, "calibration": calibration,
                "calibration_state": state_info,
            })
        rows.append(row)
    expected = {(dataset, seed, model) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS for model in expected_models}
    if expected and seen != expected:
        raise ValueError(f"{name} coordinate matrix is incomplete: expected {len(expected)}, got {len(seen)}")
    return rows


def _paired_test(deltas: Sequence[float]) -> Dict[str, Any]:
    values = [float(item) for item in deltas]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    if scipy_stats is not None and len(values) > 1:
        result = scipy_stats.ttest_1samp(values, 0.0)
        t_stat, p_value = float(result.statistic), float(result.pvalue)
    else:
        t_stat = mean / (sd / math.sqrt(len(values))) if math.isfinite(sd) and sd > 0 else math.nan
        p_value = None
    return {
        "n": len(values), "mean": mean, "sd": sd,
        "wins_lower": sum(value < 0 for value in values), "ties": sum(value == 0 for value in values),
        "t_statistic": t_stat, "two_sided_p": p_value,
    }


def _relative(delta: float, reference: float) -> float:
    return delta / reference if reference else math.nan


def _mean_available(values: Sequence[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _contrast(pairs: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    deltas = [float(item[f"delta_{metric}"]) for item in pairs]
    refs = [float(item[f"reference_{metric}"]) for item in pairs]
    treatments = [float(item[f"treatment_{metric}"]) for item in pairs]
    result = _paired_test(deltas)
    ref_mean = statistics.fmean(refs)
    result.update({"reference_mean": ref_mean, "treatment_mean": statistics.fmean(treatments), "relative_delta_mean": _relative(result["mean"], ref_mean)})
    return result


def _make_contrast(rows: Sequence[Mapping[str, Any]], reference_arm: str, treatment_arm: str, metrics: Sequence[str]) -> Dict[str, Any]:
    ref = {(str(row["dataset"]), int(row["seed"])): row for row in rows if row["arm"] == reference_arm}
    treatment = {(str(row["dataset"]), int(row["seed"])): row for row in rows if row["arm"] == treatment_arm}
    coords = sorted(set(ref) & set(treatment))
    if len(coords) != 6:
        raise ValueError(f"incomplete pairing {reference_arm} -> {treatment_arm}: {len(coords)}")
    pairs: List[Dict[str, Any]] = []
    for dataset, seed in coords:
        left, right = ref[(dataset, seed)], treatment[(dataset, seed)]
        item: Dict[str, Any] = {"dataset": dataset, "seed": seed}
        for metric in metrics:
            item[f"reference_{metric}"] = left[metric]
            item[f"treatment_{metric}"] = right[metric]
            item[f"delta_{metric}"] = right[metric] - left[metric]
        pairs.append(item)
    return {"reference_arm": reference_arm, "treatment_arm": treatment_arm, "pairs": pairs, "metrics": {metric: _contrast(pairs, metric) for metric in metrics}}


def _validate_staged_invariants(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_arm = {(str(row["arm"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    invariant_count = 0
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            memory = by_arm[("memory", dataset, seed)]
            relation = by_arm[("relation", dataset, seed)]
            calibrated = by_arm[("calibrated", dataset, seed)]
            if memory["sidecar_sha256"] != relation["sidecar_sha256"] or relation["sidecar_sha256"] != calibrated["sidecar_sha256"]:
                raise ValueError(f"ACF sidecar mismatch at {dataset}/{seed}")
            if memory["effective_lags"] != relation["effective_lags"] or relation["effective_lags"] != calibrated["effective_lags"]:
                raise ValueError(f"ACF effective lag mismatch at {dataset}/{seed}")
            if relation["relation_lags"] != calibrated["relation_lags"]:
                raise ValueError(f"relation lag mismatch at {dataset}/{seed}")
            if memory["config_info"]["signature"] != relation["config_info"]["signature"]:
                raise ValueError(f"memory/relation trunk mismatch at {dataset}/{seed}")
            if relation["config_info"]["signature"] != calibrated["config_info"]["signature"]:
                raise ValueError(f"relation/calibration trunk mismatch at {dataset}/{seed}")
            invariant_count += 1
    return {"paired_coordinates": invariant_count, "sidecar_hashes_equal": True, "shared_acf_signature_equal": True}


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    # Attach stable arm labels while preserving model names in every row.
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["arm"] = item.get("arm", item.get("kind"))
        normalized.append(item)
    expected_arms = set(BASELINE_MODELS) | {"memory", "relation", "calibrated"}
    observed = {str(row["arm"]) for row in normalized}
    if observed != expected_arms:
        raise ValueError(f"readout arms mismatch: {sorted(observed)}")
    invariants = _validate_staged_invariants(normalized)
    metrics = ("mse", "mae", "elapsed_sec")
    contrasts: Dict[str, Any] = {}
    for model in BASELINE_MODELS:
        contrasts[f"memory_minus_{model}"] = _make_contrast(normalized, model, "memory", metrics)
    contrasts["relation_minus_memory"] = _make_contrast(normalized, "memory", "relation", metrics)
    contrasts["calibrated_minus_relation"] = _make_contrast(normalized, "relation", "calibrated", metrics)
    contrasts["relation_minus_Mamba2"] = _make_contrast(normalized, "Mamba2", "relation", metrics)
    contrasts["calibrated_minus_Mamba2"] = _make_contrast(normalized, "Mamba2", "calibrated", metrics)

    diagnostics: Dict[str, Any] = {}
    for arm in sorted(expected_arms):
        subset = [row for row in normalized if row["arm"] == arm]
        if arm in BASELINE_MODELS:
            diagnostics[arm] = {
                "n": len(subset), "state_numel_mean": statistics.fmean(row["state_numel"] for row in subset),
                "state_numel_values": [row["state_numel"] for row in subset],
            }
            continue
        diagnostics[arm] = {
            "n": len(subset),
            "state_numel_mean": statistics.fmean(row["state_numel"] for row in subset),
            "acf_effective_memory_rms_mean": _mean_available(
                row["runtime"]["acf_effective_memory_rms"]["mean"] for row in subset
            ),
            "acf_runtime_gate_mean": _mean_available(
                row["runtime"]["acf_runtime_gate_mean"]["mean"] for row in subset
            ),
            "relation_sign_accuracy_mean": (
                statistics.fmean(row["relation_diagnostics"]["acf_relation_global_sign_accuracy"]["mean"] for row in subset)
                if subset[0]["relation_diagnostics"] else None
            ),
            "calibration_amp_mean": (
                statistics.fmean(row["calibration"]["acf_calibration_amp_mean"]["mean"] for row in subset if row["calibration"]["acf_calibration_amp_mean"]["mean"] is not None)
                if any(row["calibration"]["acf_calibration_amp_mean"]["mean"] is not None for row in subset) else None
            ),
            "calibration_state_parameter_count": [row["calibration_state"]["parameter_count"] for row in subset],
        }
    return {
        "schema_version": 1,
        "status": "exploratory_fair_baseline_screen_not_confirmatory",
        "plan": str(plan.get("_path", "")),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "rows": normalized,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
        "invariants": invariants,
        "evidence_boundary": "Descriptive exploratory screen only; no test-based selection, universal claim, or publication claim.",
    }


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MambaACF Fair Baseline Screen",
        "",
        "Status: **exploratory; not confirmatory**.",
        "",
        "All arms use ETTh1/traffic, H96, seeds 2021-2023, the same split and eight-epoch budget. The three ACF arms are staged mechanism pairings; DLinear, PatchTST, Mamba-1 and Mamba-2 are external reference baselines and are not parameter-count matched.",
        "",
        "## Staged Contrasts",
        "",
        "| Contrast (treatment - reference) | n | MSE delta | MSE relative | MAE delta | MAE relative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    preferred = [
        ("memory - DLinear", "memory_minus_DLinear"),
        ("memory - PatchTST", "memory_minus_PatchTST"),
        ("memory - Mamba1", "memory_minus_Mamba1"),
        ("memory - Mamba2", "memory_minus_Mamba2"),
        ("relation - memory", "relation_minus_memory"),
        ("calibrated - relation", "calibrated_minus_relation"),
    ]
    for label, key in preferred:
        values = result["contrasts"][key]["metrics"]
        mse, mae = values["mse"], values["mae"]
        lines.append(f"| {label} | {mse['n']} | {mse['mean']:.6g} | {mse['relative_delta_mean']:.3%} | {mae['mean']:.6g} | {mae['relative_delta_mean']:.3%} |")
    lines += [
        "",
        "Negative accuracy deltas favor the treatment. Any p-value in the JSON is an unadjusted descriptive paired t-test and must not be used for model selection.",
        "",
        "## Diagnostics",
        "",
        "| Arm | Checkpoint state elements (mean) | Effective memory RMS | Relation sign accuracy | Calibration amplitude |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("DLinear", "PatchTST", "Mamba1", "Mamba2", "memory", "relation", "calibrated"):
        item = result["diagnostics"][arm]
        fmt = lambda value: "n/a" if value is None else f"{float(value):.6g}"
        lines.append(f"| {arm} | {item['state_numel_mean']:.1f} | {fmt(item.get('acf_effective_memory_rms_mean'))} | {fmt(item.get('relation_sign_accuracy_mean'))} | {fmt(item.get('calibration_amp_mean'))} |")
    lines += [
        "",
        "## Boundaries",
        "",
        "- Checkpoint state elements are reported as a transparent size diagnostic, not as a trainable-parameter claim.",
        "- The staged ACF comparisons require identical raw train-only sidecar hashes and lag resolution at each dataset-seed coordinate.",
        "- Short training, six paired coordinates per contrast, non-matched external capacity, and no independent replication prevent a publication-level claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/mamba_acf_fair_baseline_screen_plan.json"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", "--output-markdown", dest="output_md", type=Path, default=None)
    args = parser.parse_args()
    plan_path = _resolve(args.plan)
    plan = dict(_read_json(plan_path))
    plan["_path"] = str(plan_path)
    rows: List[Dict[str, Any]] = []
    baseline_rows = _load_phase(plan, BASELINE_PHASE[0], BASELINE_PHASE[1], BASELINE_MODELS)
    for row in baseline_rows:
        row["arm"] = row["model"]
    rows.extend(baseline_rows)
    for phase_name, kind, model, relation, calibrated in ACF_PHASES:
        phase_rows = _load_phase(plan, phase_name, kind, (model,))
        for row in phase_rows:
            row["arm"] = kind
        rows.extend(phase_rows)
    result = build_readout(plan, rows)
    output_json = _resolve(args.output_json or Path("artifacts/paper_final/mamba_acf_fair_baseline_screen_20260901/readout.json"))
    output_md = _resolve(args.output_md or Path("docs/research/mamba_acf_fair_baseline_screen_readout_20260901.md"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "paired_coordinates_per_contrast": 6, "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
