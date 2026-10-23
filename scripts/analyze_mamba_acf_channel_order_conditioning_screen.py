#!/usr/bin/env python3
"""Strict readout for the MambaACF channel-order/conditioning screen.

The reader is intentionally read-only.  It verifies the three-arm registered
matrix, immutable provenance, manifests and artifact hashes, executable alias
mapping, train-only ACF sidecars, relation diagnostics, and the two treatment
flags before producing paired descriptive contrasts.  The only planned
contrasts are strength-order minus input-order and conditioning minus the
input-order control; no interaction or cross-plan pooling is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.acf_channel_order import signed_acf_profile_similarity_order

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - reporting remains usable without SciPy.
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
    from scripts.analyze_mamba_variable_memory_relation_screen import (
        _artifact_paths,
        _hash_file_map,
        _inside,
        _live_source_mismatches,
        _phase_from_plan,
        _phase_root_from_plan,
        _validate_provenance,
    )
    from scripts.run_training_plan import _config_hash, _job_specs, _validate_plan
except ModuleNotFoundError:  # Support ``python scripts/...`` direct calls.
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
    from analyze_mamba_variable_memory_relation_screen import (  # type: ignore
        _artifact_paths,
        _hash_file_map,
        _inside,
        _live_source_mismatches,
        _phase_from_plan,
        _phase_root_from_plan,
        _validate_provenance,
    )
    from run_training_plan import _config_hash, _job_specs, _validate_plan  # type: ignore


EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
EXPECTED_FEATURES = "M"
EXPECTED_JOBS = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)
PHASES: Tuple[Tuple[str, str], ...] = (
    ("order_input_control_h96", "input_control"),
    ("order_strength_h96", "strength"),
    ("conditioning_input_h96", "conditioning"),
)
ALIASES = {
    # alias, executable, relation, memory, lambda, order, conditioning
    "input_control": ("MambaACFVariableAutoCon", "MambaACFAutoCon", True, True, 0.03, "input", False),
    "strength": ("MambaACFVariableAutoCon", "MambaACFAutoCon", True, True, 0.03, "strength", False),
    "conditioning": ("MambaACFVariableAutoCon", "MambaACFAutoCon", True, True, 0.03, "input", True),
}
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


def _explicit_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{label} is not an explicit boolean: {value!r}")


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed != value:
        raise ValueError(f"{label} must be integral: {value!r}")
    return parsed


def _positive_lags(value: Any, label: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = [_int(item, f"{label} item") for item in value]
    if any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} contains invalid or duplicate lags")
    return result


def _optional_series(
    logs: Mapping[str, Any], key: str, path: Path, *, required: bool
) -> Dict[str, Any]:
    values = logs.get(key)
    if values is None and not required:
        return {"n": 0, "finite_n": 0, "null_n": 0, "mean": None, "min": None, "max": None, "last": None}
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


def _cfg_bool(config: Mapping[str, Any], key: str, expected: bool, job_id: str) -> None:
    if key not in config or _explicit_bool(config[key], f"config {key} for {job_id}") != expected:
        raise ValueError(f"config {key} mismatch for {job_id}")


def _cfg_equal(config: Mapping[str, Any], key: str, expected: Any, job_id: str) -> None:
    if key not in config:
        raise ValueError(f"config {key} is missing for {job_id}")
    actual = config[key]
    if isinstance(expected, float):
        try:
            same = math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            same = False
    elif isinstance(expected, bool):
        same = _explicit_bool(actual, f"config {key} for {job_id}") == expected
    else:
        same = actual == expected
    if not same:
        raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {expected!r}")


def _trunk_signature(config: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Fields that must remain fixed in every variable-token arm."""
    keys = (
        "features", "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out",
        "d_model", "d_ff", "e_layers", "n_heads", "d_layers", "factor", "dropout",
        "embed", "activation", "mamba_d_state", "mamba_d_conv", "mamba_expand",
        "mamba_headdim", "mamba_scales", "mamba_token_stride", "mamba_bidirectional",
        "mamba_causal_embedding", "mamba_channel_independent", "mamba_channel_gate",
        "mamba_variable_mixer", "mamba_variable_mixer_depth", "mamba_variable_group_size",
        "mamba_variable_permutation", "mamba_patch_mixer",
        "mamba_patch_mixer_depth", "mamba_patch_order", "mamba_local_mixer",
        "mamba_local_patch_len", "mamba_local_patch_stride", "mamba_local_width",
        "mamba_local_rank", "mamba_local_depth", "mamba_low_rank_head", "mamba_temporal_rank",
        "mamba_acf_lags", "mamba_acf_lag_mode", "mamba_acf_max_lag", "mamba_acf_top_k",
        "mamba_acf_min_lag", "mamba_acf_min_abs", "mamba_acf_confidence_z",
        "mamba_acf_min_separation", "mamba_acf_detrend", "mamba_acf_chunk",
        "mamba_acf_temperature", "mamba_acf_step_reliability", "AutoCon_acf_source",
        "batch_size", "train_epochs", "patience", "learning_rate", "lradj",
        "min_learning_rate", "weight_decay", "num_workers", "drop_last_train",
        "train_ratio", "use_amp", "mamba_acf_horizon_calibration_rank",
    )
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"common variable-token config fields missing: {missing}")
    return tuple(_canonical(config[key]) for key in keys)


def _phase_contract(plan: Mapping[str, Any], phase_name: str, kind: str) -> Dict[str, Any]:
    phase = _phase_from_plan(plan, phase_name)
    models = phase.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError(f"{phase_name} must contain one model alias")
    alias, executable, relation, memory, lambda_value, order, conditioning = ALIASES[kind]
    if str(models[0]) != alias:
        raise ValueError(f"{phase_name} model alias mismatch: {models!r} != {[alias]!r}")
    if phase.get("autocon_lambda") is not None:
        try:
            declared_lambda = float(phase["autocon_lambda"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid lambda in {phase_name}") from exc
        if not math.isclose(declared_lambda, lambda_value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"registered lambda mismatch in {phase_name}")
    return {
        "phase": phase,
        "alias": alias,
        "executable": executable,
        "relation": relation,
        "memory": memory,
        "lambda": lambda_value,
        "calibrated": False,
        "order": order,
        "conditioning": conditioning,
        "features": str(phase.get("features", plan.get("global", {}).get("features", "S"))),
        "horizon": int((phase.get("horizons") or [EXPECTED_HORIZON])[0]),
    }


def _command_value(command: Sequence[Any], name: str) -> Optional[str]:
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _validate_command(record: Mapping[str, Any], contract: Mapping[str, Any], job_id: str) -> None:
    command = record.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"missing command for {job_id}")
    if record.get("command_sha256") != hashlib.sha256(
        json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest():
        raise ValueError(f"command hash mismatch for {job_id}")
    if _command_value(command, "--model") != contract["executable"]:
        raise ValueError(f"executable model command mismatch for {job_id}")
    if "--mamba_variable_mixer" not in command:
        raise ValueError(f"variable mixer flag missing for {job_id}")
    memory_arg = (_command_value(command, "--mamba_use_acf_memory") or "").lower()
    if memory_arg != ("true" if contract["memory"] else "false"):
        raise ValueError(f"memory command flag mismatch for {job_id}")
    if contract["relation"]:
        if "--AutoCon" not in command or "--mamba_acf_relation_loss" not in command:
            raise ValueError(f"relation command flags missing for {job_id}")
        if not math.isclose(float(_command_value(command, "--AutoCon_lambda")), contract["lambda"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"command lambda mismatch for {job_id}")
    elif "--AutoCon" in command or "--mamba_acf_relation_loss" in command:
        raise ValueError(f"plain variable arm unexpectedly has AutoCon flags: {job_id}")
    relation_min = _command_value(command, "--mamba_acf_relation_min_lag")
    expected_min = "96" if contract["relation"] else "0"
    if relation_min != expected_min:
        raise ValueError(f"relation minimum-lag command mismatch for {job_id}")
    if "--mamba_acf_horizon_calibration" in command:
        raise ValueError(f"calibration command flag mismatch for {job_id}")
    order_arg = (_command_value(command, "--mamba_acf_channel_order") or "").lower()
    if order_arg != contract["order"]:
        raise ValueError(f"channel-order command mismatch for {job_id}: {order_arg!r}")
    conditioning_enabled = "--mamba_acf_channel_conditioning" in command
    if conditioning_enabled != bool(contract["conditioning"]):
        raise ValueError(f"conditioning command flag mismatch for {job_id}")
    if (_command_value(command, "--AutoCon_acf_source") or "").lower() != "raw":
        raise ValueError(f"command ACF source is not raw for {job_id}")


def _validate_config(
    config: Mapping[str, Any], record: Mapping[str, Any], contract: Mapping[str, Any], dataset: str, job_id: str
) -> Dict[str, Any]:
    channels = 7 if dataset == "ETTh1" else 862
    for key, value in {
        "model": contract["executable"],
        "features": EXPECTED_FEATURES,
        "seq_len": 96,
        "label_len": 48,
        "pred_len": EXPECTED_HORIZON,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "AutoCon": contract["relation"],
        "mamba_variable_mixer": True,
        "mamba_variable_group_size": 1,
        "mamba_causal_embedding": True,
        "mamba_acf_channel_order": contract["order"],
        "mamba_acf_channel_conditioning": contract["conditioning"],
        "mamba_use_acf_memory": contract["memory"],
        "mamba_dynamic_acf": False,
        "mamba_acf_relation_loss": contract["relation"],
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_relation_lag_mask": contract["relation"],
        "mamba_acf_relation_min_lag": 96 if contract["relation"] else 0,
        "mamba_acf_horizon_calibration": False,
        "mamba_acf_horizon_calibration_rank": 2,
        "AutoCon_acf_source": "raw",
        "train_epochs": 8,
        "patience": 3,
        "batch_size": 32,
        "skip_test": False,
        "stream_metrics": True,
        "stream_sample_windows": 1,
    }.items():
        if isinstance(value, bool):
            _cfg_bool(config, key, value, job_id)
        elif isinstance(value, float):
            _cfg_equal(config, key, value, job_id)
        else:
            _cfg_equal(config, key, value, job_id)
    if contract["relation"]:
        _cfg_equal(config, "AutoCon_lambda", contract["lambda"], job_id)
    # ``run.py`` persists the parser's default AutoCon_lambda even when the
    # AutoCon branch is disabled.  It is inert metadata for plain MambaACF;
    # only AutoCon=true arms bind this value to the registered contrast.
    if int(record.get("seed")) != int(config.get("run_seed")):
        raise ValueError(f"run seed mismatch for {job_id}")
    prior_lags = _positive_lags(config.get("mamba_acf_prior_lags"), f"prior lags {job_id}")
    effective_lags = _positive_lags(config.get("mamba_acf_effective_lags"), f"effective lags {job_id}")
    if effective_lags != prior_lags or len(effective_lags) != 3:
        raise ValueError(f"effective/prior lag contract failed for {job_id}")
    if str(config.get("mamba_acf_effective_lag_source")) not in {"train_peaks_available", "persisted_sidecar"}:
        raise ValueError(f"effective lag source mismatch for {job_id}")
    if _explicit_bool(config.get("mamba_acf_effective_lag_fallback"), "lag fallback"):
        raise ValueError(f"ACF lag fallback used for {job_id}")
    relation_lags: List[int] = []
    if contract["relation"]:
        relation_lags = _positive_lags(config.get("mamba_acf_relation_effective_lags"), f"relation lags {job_id}")
        if any(lag < 96 for lag in relation_lags):
            raise ValueError(f"relation lag below 96 for {job_id}")
        if str(config.get("mamba_acf_relation_lag_source")) not in {"train_peak_subset", "persisted_sidecar"}:
            raise ValueError(f"relation lag source mismatch for {job_id}")
        if _explicit_bool(config.get("mamba_acf_relation_lag_fallback"), "relation fallback"):
            raise ValueError(f"relation lag fallback used for {job_id}")
    prior_sha = str(config.get("mamba_acf_prior_sha256", ""))
    if len(prior_sha) != 64 or any(char not in "0123456789abcdefABCDEF" for char in prior_sha):
        raise ValueError(f"invalid prior hash for {job_id}")
    return {
        "effective_lags": effective_lags,
        "relation_lags": relation_lags,
        "prior_sha256": prior_sha,
        "trunk_signature": _trunk_signature(config),
        "channel_order": contract["order"],
        "channel_conditioning": bool(contract["conditioning"]),
    }


def _state_scalar(state: Mapping[str, Any], key: str, path: Path) -> Optional[float]:
    """Read a finite scalar treatment diagnostic from a checkpoint state."""
    value = state.get(key)
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        if array.size != 1 or not np.isfinite(array).all():
            raise ValueError
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError, ImportError) as exc:
        raise ValueError(f"checkpoint scalar {key} is invalid: {path}") from exc


def _order_activation(path: Path, config: Mapping[str, Any], job_id: str) -> Dict[str, Any]:
    """Recompute the exact static order from the persisted train-only sidecar."""
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["acf_values"], dtype=float)
            lags = np.asarray(payload["lags"], dtype=int).reshape(-1)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"cannot replay channel order for {job_id}: {path}") from exc
    if values.ndim != 2 or values.shape[1] != lags.size or values.shape[0] <= 0:
        raise ValueError(f"invalid compact sidecar for channel-order replay: {job_id}")
    peak_index = np.abs(values).argmax(axis=1)
    peak = values[np.arange(values.shape[0]), peak_index]
    peak_lag = lags[peak_index]
    order_mode = str(config.get("mamba_acf_channel_order", "input")).lower()
    if order_mode == "strength":
        order = np.asarray(
            sorted(
                range(values.shape[0]),
                key=lambda index: (-abs(float(peak[index])), int(peak_lag[index]), index),
            ),
            dtype=int,
        )
    elif order_mode == "similarity":
        # Use the same O(CK)-memory helper as the model so replay cannot drift
        # from the train-time order rule.
        order = signed_acf_profile_similarity_order(values)
    else:
        order = np.arange(values.shape[0], dtype=int)
    return {
        "order_indices": order.tolist(),
        "order_is_identity": bool(np.array_equal(order, np.arange(values.shape[0]))),
        "order_changed": bool(not np.array_equal(order, np.arange(values.shape[0]))),
        "target_input_index": int(values.shape[0] - 1),
        "target_scan_position": int(np.flatnonzero(order == values.shape[0] - 1)[0]),
        "dominant_peak_abs_mean": float(np.abs(peak).mean()),
        "dominant_peak_abs_min": float(np.abs(peak).min()),
        "dominant_peak_abs_max": float(np.abs(peak).max()),
        "dominant_peak_lag_counts": {
            str(int(lag)): int(np.sum(peak_lag == lag)) for lag in sorted(set(peak_lag.tolist()))
        },
    }


def _load_one_phase(
    plan: Mapping[str, Any], plan_path: Path, phase_name: str, kind: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    contract = _phase_contract(plan, phase_name, kind)
    phase = contract["phase"]
    phase_root = _phase_root_from_plan(plan, phase)
    validation_path = phase_root / "phase_validation.json"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ValueError(f"invalid phase validation: {validation_path}")
    if validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError(f"phase validation plan hash mismatch: {phase_name}")
    records = _manifest_records(phase_root, validation)
    if len(records) != EXPECTED_JOBS or int(validation.get("expected_jobs", -1)) != EXPECTED_JOBS:
        raise ValueError(f"unexpected six-coordinate coverage for {phase_name}")
    if validation.get("errors") not in (None, []) or validation.get("status_counts") != {"ok": EXPECTED_JOBS}:
        raise ValueError(f"phase validation contains failures: {phase_name}")
    ids = [str(item.get("job_id", "")) for item in records]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"manifest job IDs are missing/duplicated: {phase_name}")
    hash_by_job = validation.get("artifact_hashes_by_job")
    if not isinstance(hash_by_job, Mapping) or set(hash_by_job) != set(ids):
        raise ValueError(f"artifact hash map does not cover {phase_name}")
    provenance = _validate_provenance(phase_root, phase, plan_path, plan, validation, records)
    raw_provenance = _read_json(phase_root / "provenance.json")
    result_root = (phase_root / "results").resolve()
    checkpoint_root = (phase_root / "checkpoints").resolve()
    registered_jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
    registered_by_coordinate = {
        (str(job["dataset"]), int(job["seed"])): job for job in registered_jobs
    }
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for record in records:
        job_id = str(record.get("job_id"))
        if record.get("phase") != phase_name or record.get("status") != "ok":
            raise ValueError(f"unsuccessful record in {phase_name}: {job_id}")
        if str(record.get("model")) != contract["alias"]:
            raise ValueError(f"manifest alias mismatch for {job_id}")
        dataset = str(record.get("dataset"))
        seed = _int(record.get("seed"), f"{job_id} seed")
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected coordinate for {job_id}")
        coordinate = (dataset, seed)
        if coordinate in seen:
            raise ValueError(f"duplicate coordinate for {phase_name}: {coordinate}")
        seen.add(coordinate)
        registered_job = registered_by_coordinate.get(coordinate)
        if registered_job is None:
            raise ValueError(f"coordinate is not in the registered job expansion: {job_id}")
        expected_config_hash = _config_hash(
            registered_job, str(record.get("provenance_id", ""))
        )
        if str(record.get("config_hash", "")) != expected_config_hash:
            raise ValueError(f"registered config hash mismatch for {job_id}")
        if str(record.get("features")) != EXPECTED_FEATURES or _int(record.get("pred_len"), f"{job_id} horizon") != EXPECTED_HORIZON:
            raise ValueError(f"feature/horizon mismatch for {job_id}")
        manifest_lambda = record.get("autocon_lambda")
        if contract["relation"]:
            if manifest_lambda is None or not math.isclose(float(manifest_lambda), contract["lambda"], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"manifest lambda mismatch for {job_id}")
        elif manifest_lambda is not None:
            raise ValueError(f"plain arm unexpectedly has a manifest lambda for {job_id}")
        _validate_command(record, contract, job_id)
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected artifact for {job_id}")
        metrics_path = _resolve(str(expected_artifact))
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not _inside(metrics_path, result_root):
            raise ValueError(f"metrics artifact outside result root for {job_id}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        record_hashes = record.get("artifact_hashes")
        if not isinstance(record_hashes, Mapping) or dict(record_hashes) != dict(hash_by_job[job_id]):
            raise ValueError(f"manifest/validation artifact hashes differ for {job_id}")
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        if hashes != dict(record_hashes):
            raise ValueError(f"artifact hash mismatch for {job_id}")
        config = _read_json(result_dir / "config.json")
        info = _validate_config(config, record, contract, dataset, job_id)
        stream = _read_json(result_dir / "stream_metrics.json")
        if stream.get("schema_version") != 1 or stream.get("aggregation") != "streaming_window_weighted":
            raise ValueError(f"invalid streaming metric schema for {job_id}")
        if _int(stream.get("pred_len"), f"stream horizon {job_id}") != EXPECTED_HORIZON:
            raise ValueError(f"stream horizon mismatch for {job_id}")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section for {job_id}")
        mse = _finite(scaled.get("mse"), f"{job_id} scaled mse")
        mae = _finite(scaled.get("mae"), f"{job_id} scaled mae")
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        state = _checkpoint_state(checkpoint_path)
        calibration_state = _calibration_contract(state, checkpoint_path, contract["calibrated"], len(info["effective_lags"]))
        sidecar_path = result_dir / "acf_prior.npz"
        sidecar_hash = hashes.get("acf_prior.npz")
        if sidecar_hash is None or sidecar_hash != info["prior_sha256"]:
            raise ValueError(f"raw train-only sidecar hash missing/mismatched for {job_id}")
        sidecar = _load_sidecar(sidecar_path, config, job_id)
        if sidecar["lags"] != info["effective_lags"]:
            raise ValueError(f"sidecar/effective lag mismatch for {job_id}")
        order_activation = _order_activation(sidecar_path, config, job_id)
        if order_activation["order_changed"] != (contract["order"] != "input"):
            raise ValueError(f"replayed channel order activation mismatch for {job_id}")
        conditioning_scale = _state_scalar(
            state, "acf_conditioning_scale", checkpoint_path
        )
        if contract["conditioning"] and conditioning_scale is None:
            raise ValueError(f"conditioning checkpoint scale is missing for {job_id}")
        if not contract["conditioning"] and conditioning_scale is not None:
            raise ValueError(f"control checkpoint unexpectedly has conditioner scale for {job_id}")
        variable_scale = _state_scalar(state, "variable_residual_scale", checkpoint_path)
        if variable_scale is None:
            raise ValueError(f"variable residual scale is missing for {job_id}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        relation_diagnostics: Dict[str, Any] = {}
        if contract["relation"]:
            if logs.get("acf_relation_objective_version") != 3:
                raise ValueError(f"relation objective version is not 3 for {job_id}")
            relation_diagnostics = {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS}
            if relation_diagnostics["acf_relation_global_valid_fraction"]["min"] <= 0.0:
                raise ValueError(f"relation coverage is non-positive for {job_id}")
        elif logs.get("acf_relation_objective_version") not in (None, "", 0):
            raise ValueError(f"plain variable arm unexpectedly has relation metadata: {job_id}")
        runtime = {key: _optional_series(logs, key, logs_path, required=contract["relation"]) for key in RUNTIME_LOG_KEYS}
        calibration = {key: _optional_series(logs, key, logs_path, required=contract["calibrated"]) for key in CALIBRATION_LOG_KEYS}
        rows.append({
            "phase": phase_name,
            "kind": kind,
            "arm": kind,
            "alias": contract["alias"],
            "executable_model": contract["executable"],
            "dataset": dataset,
            "seed": seed,
            "mse": mse,
            "mae": mae,
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "epochs_ran": _int(record.get("epochs_ran"), f"{job_id} epochs_ran"),
            "best_epoch": _int(record.get("best_epoch"), f"{job_id} best_epoch"),
            "job_id": job_id,
            "state_numel": _state_numel(state),
            "config_sha256": _sha256(result_dir / "config.json"),
            "artifact_hashes": dict(hashes),
            "sidecar_sha256": sidecar_hash,
            "sidecar": sidecar,
            "effective_lags": info["effective_lags"],
            "relation_lags": info["relation_lags"],
            "trunk_signature": info["trunk_signature"],
            "runtime": runtime,
            "relation_diagnostics": relation_diagnostics,
            "calibration": calibration,
            "calibration_state": calibration_state,
            "channel_order": contract["order"],
            "channel_conditioning": bool(contract["conditioning"]),
            "order_activation": order_activation,
            "acf_conditioning_scale": conditioning_scale,
            "variable_residual_scale": variable_scale,
        })
    expected = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    if seen != expected:
        raise ValueError(f"coordinate matrix is incomplete for {phase_name}: {seen}")
    return rows, {
        "phase": phase_name,
        "kind": kind,
        "root": str(phase_root),
        "validation": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "manifest": str(phase_root / "logs" / "job_manifest.jsonl"),
        "manifest_sha256": _sha256(phase_root / "logs" / "job_manifest.jsonl"),
        "provenance": provenance,
        "launch_authorization_binding_sha256": raw_provenance.get("launch_authorization_binding_sha256"),
        "launch_sidecar_sha256": (
            raw_provenance.get("launch_authorization", {}).get("sidecar_sha256")
            if isinstance(raw_provenance.get("launch_authorization"), Mapping)
            else None
        ),
        "expected_jobs": EXPECTED_JOBS,
        "observed_jobs": len(rows),
        "valid": True,
    }


def _paired_test(deltas: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in deltas]
    if not values:
        raise ValueError("empty paired delta vector")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    p_value: Optional[float] = None
    if scipy_stats is not None and len(values) > 1:
        result = scipy_stats.ttest_1samp(values, 0.0)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat = mean / (sd / math.sqrt(len(values)))
    else:
        t_stat = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
    }


def _relative(delta: float, reference: float) -> float:
    return delta / reference if reference else math.nan


def _make_contrast(rows: Sequence[Mapping[str, Any]], reference: str, treatment: str) -> Dict[str, Any]:
    left = {(str(row["dataset"]), int(row["seed"])): row for row in rows if row.get("arm") == reference}
    right = {(str(row["dataset"]), int(row["seed"])): row for row in rows if row.get("arm") == treatment}
    coordinates = sorted(set(left) & set(right))
    if len(coordinates) != EXPECTED_JOBS:
        raise ValueError(f"incomplete pairing {reference} -> {treatment}: {len(coordinates)}")
    pairs = []
    for dataset, seed in coordinates:
        reference_row, treatment_row = left[(dataset, seed)], right[(dataset, seed)]
        pairs.append({
            "dataset": dataset,
            "seed": seed,
            "reference_mse": float(reference_row["mse"]),
            "treatment_mse": float(treatment_row["mse"]),
            "delta_mse": float(treatment_row["mse"] - reference_row["mse"]),
            "reference_mae": float(reference_row["mae"]),
            "treatment_mae": float(treatment_row["mae"]),
            "delta_mae": float(treatment_row["mae"] - reference_row["mae"]),
            "reference_job_id": reference_row.get("job_id"),
            "treatment_job_id": treatment_row.get("job_id"),
        })
    def _metric_summary(items: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
        deltas = [float(item[f"delta_{metric}"]) for item in items]
        summary = _paired_test(deltas)
        references = [float(item[f"reference_{metric}"]) for item in items]
        treatments = [float(item[f"treatment_{metric}"]) for item in items]
        summary.update({
            "reference_mean": statistics.fmean(references),
            "treatment_mean": statistics.fmean(treatments),
            "relative_delta_mean": _relative(summary["mean"], statistics.fmean(references)),
            "delta_values": deltas,
        })
        return summary

    metrics = {}
    for metric in ("mse", "mae"):
        metrics[metric] = _metric_summary(pairs, metric)
    by_dataset: Dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        if len(subset) != len(EXPECTED_SEEDS):
            raise ValueError(f"incomplete per-dataset pairing {reference} -> {treatment}: {dataset}")
        by_dataset[dataset] = {
            "n": len(subset),
            "metrics": {metric: _metric_summary(subset, metric) for metric in ("mse", "mae")},
        }
    return {
        "reference_arm": reference,
        "treatment_arm": treatment,
        "pairs": pairs,
        "metrics": metrics,
        "by_dataset": by_dataset,
    }


def _validate_invariants(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {(str(row["arm"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            coordinate_rows = [indexed[(arm, dataset, seed)] for arm in ("input_control", "strength", "conditioning")]
            sidecars = {row.get("sidecar_sha256") for row in coordinate_rows}
            if len(sidecars) != 1:
                raise ValueError(f"sidecar identity differs at {dataset}/{seed}")
            lags = {tuple(row.get("effective_lags", [])) for row in coordinate_rows}
            if len(lags) != 1:
                raise ValueError(f"effective ACF lags differ at {dataset}/{seed}")
            signatures = {row.get("trunk_signature") for row in coordinate_rows}
            if len(signatures) != 1:
                raise ValueError(f"variable-token trunk differs at {dataset}/{seed}")
            if len({row.get("executable_model") for row in coordinate_rows}) != 1:
                raise ValueError(f"executable mismatch at {dataset}/{seed}")
            relation_lags = {tuple(row.get("relation_lags", [])) for row in coordinate_rows}
            if len(relation_lags) != 1 or not next(iter(relation_lags)):
                raise ValueError(f"relation lag identity differs/empty at {dataset}/{seed}")
            if any(row.get("channel_order") != ("strength" if row.get("arm") == "strength" else "input") for row in coordinate_rows):
                raise ValueError(f"registered channel order mismatch at {dataset}/{seed}")
            if any(bool(row.get("channel_conditioning")) != (row.get("arm") == "conditioning") for row in coordinate_rows):
                raise ValueError(f"registered conditioning flag mismatch at {dataset}/{seed}")
            strength_row = indexed[("strength", dataset, seed)]
            if not strength_row["order_activation"]["order_changed"]:
                raise ValueError(f"strength order did not change identity at {dataset}/{seed}")
            if not indexed[("input_control", dataset, seed)]["order_activation"]["order_is_identity"]:
                raise ValueError(f"input control order is not identity at {dataset}/{seed}")
            if indexed[("conditioning", dataset, seed)]["acf_conditioning_scale"] is None:
                raise ValueError(f"conditioning scale missing at {dataset}/{seed}")
    return {
        "paired_coordinates": EXPECTED_JOBS,
        "sidecar_hashes_equal": True,
        "sidecar_hashes_equal_per_coordinate": True,
        "effective_lags_equal": True,
        "variable_token_trunk_equal": True,
        "relation_lags_equal": True,
        "input_order_identity": True,
        "strength_order_replayed": True,
        "conditioning_scale_present": True,
    }


def _stopping_decision(contrast: Mapping[str, Any], label: str) -> Dict[str, Any]:
    deltas = [float(item["delta_mse"]) for item in contrast["pairs"]]
    wins_by_dataset = {
        dataset: sum(float(item["delta_mse"]) < 0 for item in contrast["pairs"] if item["dataset"] == dataset)
        for dataset in EXPECTED_DATASETS
    }
    meets = len(deltas) >= 6 and sum(value < 0 for value in deltas) >= 4 and all(value >= 1 for value in wins_by_dataset.values())
    return {
        "criterion": f"{label} is favorable on at least 4/6 coordinates and at least 1/3 coordinate in each dataset (descriptive screen rule only)",
        "favorable_coordinates": sum(value < 0 for value in deltas),
        "favorable_coordinates_by_dataset": wins_by_dataset,
        "meets_descriptive_screen": meets,
        "action": "continue only to separately reviewed frozen replication" if meets else "stop exploratory program; do not add arms, seeds, horizons, or interactions",
    }


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], validation: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    normalized = [dict(row) for row in rows]
    expected_arms = {kind for _, kind in PHASES}
    if {str(row.get("arm")) for row in normalized} != expected_arms:
        raise ValueError("readout arms do not match the three registered phases")
    if len(normalized) != len(PHASES) * EXPECTED_JOBS:
        raise ValueError(f"expected {len(PHASES) * EXPECTED_JOBS} rows, found {len(normalized)}")
    invariants = _validate_invariants(normalized)
    contrasts = {
        "strength_minus_input": _make_contrast(normalized, "input_control", "strength"),
        "conditioning_minus_input": _make_contrast(normalized, "input_control", "conditioning"),
    }
    stopping = {
        "strength_minus_input": _stopping_decision(
            contrasts["strength_minus_input"], "strength-order minus input-order"
        ),
        "conditioning_minus_input": _stopping_decision(
            contrasts["conditioning_minus_input"], "conditioning minus input-order"
        ),
    }
    diagnostics: Dict[str, Any] = {}
    for arm in sorted(expected_arms):
        subset = [row for row in normalized if row["arm"] == arm]
        runtime_values = [row.get("runtime", {}).get("acf_effective_memory_rms", {}).get("mean") for row in subset]
        runtime_values = [float(value) for value in runtime_values if value is not None]
        sign_values = [row.get("relation_diagnostics", {}).get("acf_relation_global_sign_accuracy", {}).get("mean") for row in subset]
        sign_values = [float(value) for value in sign_values if value is not None]
        conditioning_values = [row.get("acf_conditioning_scale") for row in subset]
        conditioning_values = [float(value) for value in conditioning_values if value is not None]
        order_changed = [bool(row.get("order_activation", {}).get("order_changed")) for row in subset]
        diagnostics[arm] = {
            "n": len(subset),
            "state_numel_mean": statistics.fmean(float(row["state_numel"]) for row in subset),
            "acf_effective_memory_rms_mean": statistics.fmean(runtime_values) if runtime_values else None,
            "relation_sign_accuracy_mean": statistics.fmean(sign_values) if sign_values else None,
            "acf_conditioning_scale_mean": statistics.fmean(conditioning_values) if conditioning_values else None,
            "acf_conditioning_scale_abs_mean": statistics.fmean(abs(value) for value in conditioning_values) if conditioning_values else None,
            "order_changed_count": sum(order_changed),
            "order_changed_fraction": statistics.fmean(order_changed) if order_changed else None,
            "variable_residual_scale_mean": statistics.fmean(float(row["variable_residual_scale"]) for row in subset),
            "state_numel_values": sorted({int(row["state_numel"]) for row in subset}),
        }
    plan_path = Path(str(plan.get("_path", ""))).resolve()
    source_drift = {
        item["phase"]: list(item.get("provenance", {}).get("live_source_mismatches", []))
        for item in (validation or [])
    }
    snapshot_hashes_valid = all(
        bool(item.get("provenance", {}).get("source_tree_sha256"))
        and not item.get("provenance", {}).get("snapshot_hash_error")
        for item in (validation or [])
    )
    return {
        "schema_version": 1,
        "status": "exploratory_mamba_acf_channel_order_conditioning_screen_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "rows": normalized,
        "validation": list(validation or []),
        "replay": {
            "phase_validations_all_valid": all(item.get("valid") is True for item in (validation or [])),
            "artifact_hashes_replayed": True,
            "immutable_snapshot_hashes_valid": snapshot_hashes_valid,
            "live_source_mismatches_by_phase": source_drift,
            "live_source_matches_snapshot": all(not values for values in source_drift.values()),
        },
        "invariants": invariants,
        "contrasts": contrasts,
        "stopping_decision": stopping,
        "diagnostics": diagnostics,
        "evidence_boundary": "Descriptive exploratory evidence only; no test-based selection, order-conditioning interaction claim, pooled cross-plan inference, or publication claim.",
    }


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MambaACF Channel Order and Conditioning Screen",
        "",
        "> Exploratory, non-confirmatory readout. All rows use the same memory-backed global-relation `MambaACFVariableAutoCon` architecture; only train-prior order or conditioning is changed.",
        "",
        "## Contrasts",
        "",
        "| Treatment - input-order control | n | MSE delta | MSE relative | MSE wins | MAE delta | p (descriptive) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [
        ("strength order", "strength_minus_input"),
        ("ACF token conditioning", "conditioning_minus_input"),
    ]
    for label, key in labels:
        metric = result["contrasts"][key]["metrics"]
        mse, mae = metric["mse"], metric["mae"]
        lines.append(f"| {label} | {mse['n']} | {_fmt(mse['mean'])} | {_fmt(100 * mse['relative_delta_mean'], 3)}% | {mse['wins_lower']}/{mse['n']} | {_fmt(mae['mean'])} | {_fmt(mse['two_sided_p'], 4)} |")
    lines += [
        "",
        "### Per-dataset paired MSE",
        "",
        "| Contrast | Dataset | n | Reference mean | Treatment mean | Delta | Wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in labels:
        for dataset in EXPECTED_DATASETS:
            metric = result["contrasts"][key]["by_dataset"][dataset]["metrics"]["mse"]
            lines.append(
                f"| {label} | {dataset} | {metric['n']} | {_fmt(metric['reference_mean'])} | "
                f"{_fmt(metric['treatment_mean'])} | {_fmt(metric['mean'])} | "
                f"{metric['wins_lower']}/{metric['n']} |"
            )
    lines += [
        "",
        "Negative deltas favor the treatment. Paired t statistics and p-values are descriptive only; no test metric is used for selection.",
        "",
        "## Invariants",
        "",
        "- Every arm has six exact dataset-seed coordinates, immutable provenance, complete artifact hashes, and a raw train-only `acf_prior.npz`.",
        "- Sidecar hashes, effective forecast lags, relation candidate lags, and all non-treatment trunk settings agree at each coordinate.",
        "- Strength order is replayed from the sidecar and must be non-identity; input control must be identity. The conditioner scale is read from checkpoint state.",
        "- The omitted strength-plus-conditioning interaction and all comparisons with other plans are outside this readout.",
        "- Replay status is recorded in the JSON. The immutable launch snapshots validate the artifacts; any post-launch live-source drift is reported and is not treated as a rerun.",
        "",
        "## Screening Decision",
        "",
        f"- Strength-order rule: `{result['stopping_decision']['strength_minus_input']['favorable_coordinates']}/6` favorable; by dataset `{result['stopping_decision']['strength_minus_input']['favorable_coordinates_by_dataset']}`.",
        f"- Conditioning rule: `{result['stopping_decision']['conditioning_minus_input']['favorable_coordinates']}/6` favorable; by dataset `{result['stopping_decision']['conditioning_minus_input']['favorable_coordinates_by_dataset']}`.",
        "- Any favorable direction is exploratory only and cannot support a publication claim without a separately reviewed frozen replication.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/mamba_acf_channel_order_conditioning_screen_plan.json"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", "--output-markdown", dest="output_md", type=Path, default=None)
    args = parser.parse_args(argv)
    plan_path = _resolve(args.plan)
    try:
        plan = dict(_read_json(plan_path))
        plan["_path"] = str(plan_path)
        errors = _validate_plan(plan)
        if errors:
            raise ValueError("plan validation failed: " + "; ".join(errors))
        rows: List[Dict[str, Any]] = []
        validation: List[Dict[str, Any]] = []
        for phase_name, kind in PHASES:
            phase_rows, phase_meta = _load_one_phase(plan, plan_path, phase_name, kind)
            rows.extend(phase_rows)
            validation.append(phase_meta)
        result = build_readout(plan, rows, validation)
        output_json = _resolve(args.output_json or Path("artifacts/paper_final/mamba_acf_channel_order_conditioning_screen_20260902/readout.json"))
        output_md = _resolve(args.output_md or Path("docs/research/mamba_acf_channel_order_conditioning_screen_readout_20260902.md"))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        output_md.write_text(markdown(result), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "paired_coordinates_per_contrast": EXPECTED_JOBS, "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"analyze_mamba_acf_channel_order_conditioning_screen: error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
