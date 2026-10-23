#!/usr/bin/env python3
"""Strict, read-only readout for the MambaACF memory-gate mode screen.

The registered screen compares two otherwise identical variable-token
``MambaACFVariableAutoCon`` arms:

* ``hidden``: a shared temporal hidden-state gate with shape ``[B,H,1]``;
* ``variable_token``: a channel-specific gate from restored variable tokens,
  with shape ``[B,H,C]``.

This module never trains, evaluates a model, or mutates training artifacts.
It only writes the requested JSON/Markdown readout after all phase validation,
manifest, artifact-hash, sidecar, configuration, and immutable-provenance
checks succeed.  Missing training artifacts are reported as ``NOT_RUN`` and
do not produce a fabricated result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - the reader remains usable without SciPy.
    scipy_stats = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
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
        _inside,
        _live_source_mismatches,
        _phase_from_plan,
        _phase_root_from_plan,
        _validate_provenance,
    )
    from scripts.run_training_plan import _config_hash, _job_specs, _validate_plan
except ModuleNotFoundError:  # Support direct ``python scripts/...`` execution.
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
        _inside,
        _live_source_mismatches,
        _phase_from_plan,
        _phase_root_from_plan,
        _validate_provenance,
    )
    from run_training_plan import _config_hash, _job_specs, _validate_plan  # type: ignore


EXPECTED_PHASES: Tuple[Tuple[str, str], ...] = (
    ("memory_gate_hidden_h96", "hidden"),
    ("memory_gate_variable_token_h96", "variable_token"),
)
PHASES = EXPECTED_PHASES
EXPECTED_DATASETS: Tuple[str, ...] = ("ETTh1", "traffic")
EXPECTED_SEEDS: Tuple[int, ...] = (2021, 2022, 2023)
EXPECTED_FEATURES = "M"
EXPECTED_HORIZON = 96
EXPECTED_JOBS_PER_PHASE = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)
EXPECTED_JOBS = len(EXPECTED_PHASES) * EXPECTED_JOBS_PER_PHASE
EXPECTED_ALIAS = "MambaACFVariableAutoCon"
EXPECTED_EXECUTABLE = "MambaACFAutoCon"
EXPECTED_LAMBDA = 0.03

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

# Configuration fields that define the common trunk.  The treatment variable
# ``mamba_acf_memory_gate_mode`` is intentionally excluded and checked apart.
COMMON_CONFIG_KEYS = (
    "model",
    "features",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "d_ff",
    "e_layers",
    "n_heads",
    "d_layers",
    "factor",
    "dropout",
    "embed",
    "activation",
    "AutoCon",
    "AutoCon_lambda",
    "AutoCon_acf_source",
    "AutoCon_warmup_epochs",
    "AutoCon_multiscales",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "mamba_headdim",
    "mamba_scales",
    "mamba_token_stride",
    "mamba_bidirectional",
    "mamba_causal_embedding",
    "mamba_channel_independent",
    "mamba_channel_gate",
    "mamba_variable_mixer",
    "mamba_variable_mixer_depth",
    "mamba_variable_group_size",
    "mamba_variable_permutation",
    "mamba_acf_channel_order",
    "mamba_acf_channel_conditioning",
    "mamba_acf_channel_gate",
    "mamba_acf_repr_fusion",
    "mamba_patch_mixer",
    "mamba_patch_mixer_depth",
    "mamba_patch_len",
    "mamba_patch_stride",
    "mamba_patch_order",
    "mamba_patch_permutation",
    "mamba_local_mixer",
    "mamba_local_patch_len",
    "mamba_local_patch_stride",
    "mamba_local_width",
    "mamba_local_rank",
    "mamba_local_depth",
    "mamba_low_rank_head",
    "mamba_temporal_rank",
    "mamba_acf_lags",
    "mamba_acf_lag_mode",
    "mamba_acf_max_lag",
    "mamba_acf_top_k",
    "mamba_acf_min_lag",
    "mamba_acf_min_abs",
    "mamba_acf_confidence_z",
    "mamba_acf_min_separation",
    "mamba_acf_detrend",
    "mamba_acf_chunk",
    "mamba_use_acf_memory",
    "mamba_dynamic_acf",
    "mamba_acf_temperature",
    "mamba_acf_step_reliability",
    "mamba_acf_horizon_calibration",
    "mamba_acf_horizon_calibration_rank",
    "mamba_acf_relation_loss",
    "mamba_acf_relation_batching",
    "mamba_acf_relation_lag_balance",
    "mamba_acf_relation_channel_reduce",
    "mamba_acf_relation_scope",
    "mamba_acf_relation_lag_mask",
    "mamba_acf_relation_lag_tolerance",
    "mamba_acf_relation_min_lag",
    "mamba_acf_relation_target_mode",
    "mamba_acf_relation_soft_min_abs",
    "mamba_acf_relation_diagnostic_min_abs",
    "mamba_acf_relation_selected_lag_boost",
    "mamba_acf_relation_active_anchor_normalization",
    "AutoCon_temperature",
    "AutoCon_relation_temperature",
    "AutoCon_sign_weight",
    "AutoCon_sign_margin",
    "batch_size",
    "train_epochs",
    "patience",
    "learning_rate",
    "lradj",
    "min_learning_rate",
    "weight_decay",
    "num_workers",
    "drop_last_train",
    "train_ratio",
    "use_amp",
)


class NotRunError(RuntimeError):
    """Raised when the registered screen has no complete training artifacts."""


def _canonical(value: Any) -> Any:
    """Normalize JSON scalars/containers for exact common-config comparison."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _explicit_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{label} is not an explicit boolean: {value!r}")


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and result != value:
        raise ValueError(f"{label} must be integral")
    return result


def _positive_lags(value: Any, label: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = [_int(item, f"{label} item") for item in value]
    if any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{label} contains invalid or duplicate lags")
    return result


def _command_value(command: Sequence[Any], name: str) -> Optional[str]:
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
        return None
    return tokens[index + 1]


def _command_flag(command: Sequence[Any], name: str, default: bool = False) -> bool:
    """Parse argparse-style flags that may be bare or followed by a boolean."""

    tokens = [str(item) for item in command]
    if name not in tokens:
        return bool(default)
    index = tokens.index(name)
    if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
        try:
            return _explicit_bool(tokens[index + 1], name)
        except ValueError:
            return True
    return True


def _command_hash(command: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _phase_contract(plan: Mapping[str, Any], phase_name: str, mode: str) -> Dict[str, Any]:
    """Validate a phase declaration and return its expected effective flags."""

    phase = _phase_from_plan(plan, phase_name)
    if str(phase.get("name")) != phase_name:
        raise ValueError(f"phase name mismatch: {phase_name}")
    models = phase.get("models")
    if models != [EXPECTED_ALIAS]:
        raise ValueError(f"{phase_name} must use alias {EXPECTED_ALIAS}")
    if str(phase.get("features", plan.get("global", {}).get("features", ""))) != EXPECTED_FEATURES:
        raise ValueError(f"{phase_name} feature contract mismatch")
    if phase.get("datasets") != list(EXPECTED_DATASETS):
        raise ValueError(f"{phase_name} dataset contract mismatch")
    if phase.get("horizons") != [EXPECTED_HORIZON]:
        raise ValueError(f"{phase_name} horizon contract mismatch")
    if phase.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError(f"{phase_name} seed contract mismatch")
    if mode not in {"hidden", "variable_token"}:
        raise ValueError(f"unsupported gate mode: {mode}")
    if str(phase.get("mamba_acf_memory_gate_mode")) != mode:
        raise ValueError(f"{phase_name} gate mode mismatch")
    if phase.get("autocon_lambda") is None or not math.isclose(
        float(phase["autocon_lambda"]), EXPECTED_LAMBDA, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(f"{phase_name} AutoCon lambda mismatch")
    required = {
        "mamba_variable_mixer": True,
        "mamba_variable_group_size": 1,
        "mamba_acf_channel_order": "input",
        "mamba_acf_channel_conditioning": False,
        "mamba_use_acf_memory": True,
        "mamba_dynamic_acf": False,
        "mamba_acf_lag_mode": "train_peaks_available",
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_lag_mask": True,
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_horizon_calibration": False,
        "epochs": 8,
        "patience": 3,
        "stream_metrics": True,
        "stream_sample_windows": 1,
    }
    for key, expected in required.items():
        actual = phase.get(key)
        if isinstance(expected, bool):
            actual = _explicit_bool(actual, f"{phase_name} {key}")
        elif isinstance(expected, float):
            actual = float(actual)
        if actual != expected:
            raise ValueError(f"{phase_name} {key} mismatch: {actual!r} != {expected!r}")
    return {
        "phase": phase,
        "mode": mode,
        "alias": EXPECTED_ALIAS,
        "executable": EXPECTED_EXECUTABLE,
        "lambda": EXPECTED_LAMBDA,
    }


def _plan_contract_errors(plan: Mapping[str, Any]) -> List[str]:
    """Return deterministic plan-contract errors without touching artifacts."""

    errors: List[str] = []
    try:
        generic = _validate_plan(dict(plan))
        if generic:
            errors.extend(str(item) for item in generic)
    except Exception as exc:
        errors.append(f"generic plan validation raised: {exc}")
    if str(plan.get("status")) != "exploratory_mamba_acf_memory_gate_mode_screen_draft":
        errors.append(f"unexpected plan status: {plan.get('status')!r}")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("plan execution metadata is missing")
    else:
        if execution.get("authorized") is not False:
            errors.append("draft plan must have execution.authorized=false")
        if execution.get("training_started") is not False:
            errors.append("draft plan must have execution.training_started=false")
    phases = plan.get("phases")
    if not isinstance(phases, list) or len(phases) != len(EXPECTED_PHASES):
        errors.append("plan must contain exactly two gate-mode phases")
    else:
        names = [str(item.get("name")) for item in phases if isinstance(item, Mapping)]
        if names != [name for name, _ in EXPECTED_PHASES]:
            errors.append(f"phase names/order mismatch: {names!r}")
        for name, mode in EXPECTED_PHASES:
            try:
                _phase_contract(plan, name, mode)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))
    counts = plan.get("registered_job_counts")
    expected_counts = {
        "memory_gate_hidden_h96": 6,
        "memory_gate_variable_token_h96": 6,
        "exploratory_total": 12,
        "training_total": 12,
        "total": 12,
        "training_started": False,
        "resource_runs_not_in_training_matrix": True,
    }
    if not isinstance(counts, Mapping):
        errors.append("registered_job_counts is missing")
    else:
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                errors.append(f"registered_job_counts[{key}] mismatch: {counts.get(key)!r} != {expected!r}")
    global_cfg = plan.get("global")
    if not isinstance(global_cfg, Mapping):
        errors.append("global configuration is missing")
    else:
        for key, expected in {
            "features": "M",
            "seq_len": 96,
            "label_len": 48,
            "d_model": 32,
            "d_ff": 64,
            "mamba_variable_mixer": True,
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
            "mamba_acf_memory_gate_mode": "hidden",
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_min_lag": 96,
        }.items():
            actual = global_cfg.get(key)
            if isinstance(expected, bool):
                try:
                    actual = _explicit_bool(actual, f"global {key}")
                except ValueError:
                    pass
            if actual != expected:
                errors.append(f"global {key} mismatch: {actual!r} != {expected!r}")
    resource = plan.get("resource_contract")
    if not isinstance(resource, Mapping):
        errors.append("resource_contract is missing")
    else:
        if resource.get("gate_parameters_each_mode") != 3168:
            errors.append("resource gate parameter contract mismatch")
        hidden = resource.get("hidden", {})
        variable = resource.get("variable_token", {})
        if not isinstance(hidden, Mapping) or hidden.get("activation_shape") != "B,H,1":
            errors.append("hidden resource shape contract mismatch")
        if not isinstance(variable, Mapping) or variable.get("activation_shape") != "B,H,C":
            errors.append("variable-token resource shape contract mismatch")
    return errors


def _validate_plan_contract(plan: Mapping[str, Any]) -> List[str]:
    """Public testable helper mirroring the runner's list-of-errors contract."""

    return _plan_contract_errors(plan)


def validate_plan_contract(plan: Mapping[str, Any]) -> None:
    errors = _plan_contract_errors(plan)
    if errors:
        raise ValueError("gate-mode plan validation failed: " + "; ".join(errors))


def _missing_training_artifacts(plan: Mapping[str, Any]) -> List[str]:
    """List the first missing completion markers without creating directories."""

    missing: List[str] = []
    for phase_name, _ in EXPECTED_PHASES:
        phase = _phase_from_plan(plan, phase_name)
        root = _phase_root_from_plan(plan, phase)
        for relative in ("phase_validation.json", "logs/job_manifest.jsonl"):
            path = root / relative
            if not path.is_file():
                missing.append(str(path))
        validation_path = root / "phase_validation.json"
        if validation_path.is_file():
            try:
                validation = _read_json(validation_path)
                if validation.get("valid") is not True:
                    missing.append(str(validation_path) + " (valid=false)")
            except (OSError, ValueError, TypeError):
                missing.append(str(validation_path) + " (unreadable)")
    return missing


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    """Load a checkpoint on CPU and strip an optional DataParallel prefix."""

    try:
        import torch

        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch.
            value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    state = value.get("model", value)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint model state is not a mapping: {path}")
    return {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): tensor
        for key, tensor in state.items()
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


def _state_array(value: Any, path: Path, key: str) -> Any:
    try:
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(f"cannot read checkpoint tensor {key}: {path}") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"checkpoint tensor {key} is non-finite: {path}")
    return array


def _gate_state_contract(state: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    """Validate the shared ``Linear(d_model, pred_len)`` gate state."""

    weight_key, bias_key = "acf_gate.weight", "acf_gate.bias"
    if weight_key not in state or bias_key not in state:
        raise ValueError(f"checkpoint is missing ACF gate tensors: {path}")
    weight = _state_array(state[weight_key], path, weight_key)
    bias = _state_array(state[bias_key], path, bias_key)
    if tuple(weight.shape) != (EXPECTED_HORIZON, 32):
        raise ValueError(f"ACF gate weight shape {tuple(weight.shape)} != (96, 32): {path}")
    if tuple(bias.shape) != (EXPECTED_HORIZON,):
        raise ValueError(f"ACF gate bias shape {tuple(bias.shape)} != (96,): {path}")
    residual = state.get("variable_residual_scale")
    residual_value: Optional[float] = None
    if residual is not None:
        residual_array = _state_array(residual, path, "variable_residual_scale")
        if residual_array.size != 1:
            raise ValueError(f"variable_residual_scale is not scalar: {path}")
        residual_value = float(residual_array.reshape(-1)[0])
    return {
        "gate_weight_shape": list(weight.shape),
        "gate_bias_shape": list(bias.shape),
        "gate_parameter_count": int(weight.size + bias.size),
        "gate_weight_mean": float(weight.mean()),
        "gate_weight_std": float(weight.std()),
        "gate_weight_l2": float((weight * weight).sum() ** 0.5),
        "gate_bias_mean": float(bias.mean()),
        "gate_bias_std": float(bias.std()),
        "variable_residual_scale": residual_value,
        "state_numel": _state_numel(state),
    }


def _check_config(
    config: Mapping[str, Any], mode: str, dataset: str, seed: int, job_id: str
) -> Dict[str, Any]:
    """Validate persisted config and return sidecar/common-signature metadata."""

    channels = 7 if dataset == "ETTh1" else 862
    expected: Dict[str, Any] = {
        "model": EXPECTED_EXECUTABLE,
        "data": dataset,
        "features": EXPECTED_FEATURES,
        "seq_len": 96,
        "label_len": 48,
        "pred_len": EXPECTED_HORIZON,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "AutoCon": True,
        "AutoCon_lambda": EXPECTED_LAMBDA,
        "AutoCon_acf_source": "raw",
        "mamba_causal_embedding": True,
        "mamba_variable_mixer": True,
        "mamba_variable_group_size": 1,
        "mamba_variable_permutation": False,
        "mamba_acf_channel_order": "input",
        "mamba_acf_channel_conditioning": False,
        "mamba_use_acf_memory": True,
        "mamba_dynamic_acf": False,
        "mamba_acf_memory_gate_mode": mode,
        "mamba_acf_lag_mode": "train_peaks_available",
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_lag_mask": True,
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_horizon_calibration": False,
        "mamba_acf_horizon_calibration_rank": 2,
        "train_epochs": 8,
        "patience": 3,
        "batch_size": 32,
        "skip_test": False,
        "stream_metrics": True,
        "stream_sample_windows": 1,
        "run_seed": seed,
        "seed": seed,
    }
    for key, wanted in expected.items():
        if key not in config:
            raise ValueError(f"config {key} is missing for {job_id}")
        actual = config[key]
        if isinstance(wanted, bool):
            actual = _explicit_bool(actual, f"config {key} for {job_id}")
        elif isinstance(wanted, float):
            try:
                if not math.isclose(float(actual), wanted, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} mismatch for {job_id}") from exc
            continue
        elif key in {
            "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out",
            "mamba_variable_group_size", "mamba_acf_relation_min_lag",
            "mamba_acf_horizon_calibration_rank", "train_epochs", "patience",
            "batch_size", "run_seed", "seed", "stream_sample_windows",
        }:
            actual = _int(actual, f"config {key} for {job_id}")
        if actual != wanted:
            raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {wanted!r}")

    requested_lags = _positive_lags(config.get("mamba_acf_lags"), f"config lags {job_id}")
    effective_lags = _positive_lags(
        config.get("mamba_acf_effective_lags"), f"effective lags {job_id}"
    )
    prior_lags = _positive_lags(config.get("mamba_acf_prior_lags"), f"prior lags {job_id}")
    if effective_lags != prior_lags or len(effective_lags) != 3:
        raise ValueError(f"effective/prior lag contract failed for {job_id}")
    if str(config.get("mamba_acf_effective_lag_source")) not in {
        "train_peaks_available", "persisted_sidecar"
    }:
        raise ValueError(f"effective lag source mismatch for {job_id}")
    if _explicit_bool(config.get("mamba_acf_effective_lag_fallback"), "effective lag fallback"):
        raise ValueError(f"ACF lag fallback used for {job_id}")
    relation_lags = _positive_lags(
        config.get("mamba_acf_relation_effective_lags"), f"relation lags {job_id}"
    )
    if any(lag < 96 for lag in relation_lags):
        raise ValueError(f"relation lag below 96 for {job_id}")
    if str(config.get("mamba_acf_relation_lag_source")) not in {
        "train_peak_subset", "persisted_sidecar"
    }:
        raise ValueError(f"relation lag source mismatch for {job_id}")
    if _explicit_bool(config.get("mamba_acf_relation_lag_fallback"), "relation lag fallback"):
        raise ValueError(f"relation lag fallback used for {job_id}")
    selected = _positive_lags(
        config.get("mamba_acf_relation_selected_lags"), f"selected relation lags {job_id}"
    )
    if not set(relation_lags).issubset(set(selected)):
        raise ValueError(f"relation effective lags are not selected lags for {job_id}")
    prior_shape = config.get("mamba_acf_prior_shape")
    if not isinstance(prior_shape, (list, tuple)) or len(prior_shape) != 2:
        raise ValueError(f"prior shape is invalid for {job_id}")
    prior_shape = [_int(item, f"prior shape {job_id}") for item in prior_shape]
    if prior_shape[0] != channels or prior_shape[1] != len(effective_lags):
        raise ValueError(f"prior shape does not match dataset/lags for {job_id}")
    if config.get("mamba_acf_prior_artifact") != "acf_prior.npz":
        raise ValueError(f"prior artifact is not acf_prior.npz for {job_id}")
    prior_sha = str(config.get("mamba_acf_prior_sha256", ""))
    if len(prior_sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in prior_sha):
        raise ValueError(f"invalid prior sha256 for {job_id}")

    signature: Dict[str, Any] = {}
    for key in COMMON_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"common config field {key} is missing for {job_id}")
        signature[key] = _canonical(config[key])
    return {
        "requested_lags": requested_lags,
        "effective_lags": effective_lags,
        "relation_lags": relation_lags,
        "selected_relation_lags": selected,
        "prior_sha256": prior_sha.lower(),
        "prior_shape": prior_shape,
        "signature": signature,
        "mode": mode,
    }


def _validate_command(
    record: Mapping[str, Any], mode: str, job_id: str, checkpoint_root: Optional[Path] = None
) -> None:
    command = record.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"missing command for {job_id}")
    if record.get("command_sha256") != _command_hash(command):
        raise ValueError(f"command hash mismatch for {job_id}")
    if _command_value(command, "--model") != EXPECTED_EXECUTABLE:
        raise ValueError(f"executable model command mismatch for {job_id}")
    if not _command_flag(command, "--mamba_variable_mixer"):
        raise ValueError(f"variable mixer command flag missing for {job_id}")
    if not _command_flag(command, "--mamba_use_acf_memory"):
        raise ValueError(f"ACF memory command flag missing for {job_id}")
    if _command_flag(command, "--mamba_dynamic_acf"):
        raise ValueError(f"dynamic ACF command flag unexpectedly enabled for {job_id}")
    if _command_value(command, "--mamba_acf_memory_gate_mode") != mode:
        raise ValueError(f"gate mode command mismatch for {job_id}")
    if "--AutoCon" not in command or "--mamba_acf_relation_loss" not in command:
        raise ValueError(f"relation command flags missing for {job_id}")
    try:
        command_lambda = float(_command_value(command, "--AutoCon_lambda"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"command lambda missing for {job_id}") from exc
    if not math.isclose(command_lambda, EXPECTED_LAMBDA, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"command lambda mismatch for {job_id}")
    if _command_value(command, "--mamba_acf_relation_min_lag") != "96":
        raise ValueError(f"relation min-lag command mismatch for {job_id}")
    if (_command_value(command, "--AutoCon_acf_source") or "").lower() != "raw":
        raise ValueError(f"command ACF source is not raw for {job_id}")
    if "--mamba_acf_horizon_calibration" in command:
        raise ValueError(f"horizon calibration unexpectedly enabled for {job_id}")
    if checkpoint_root is not None:
        declared = _command_value(command, "--checkpoints")
        if declared is not None and Path(declared).expanduser().resolve() != checkpoint_root.resolve():
            raise ValueError(f"command checkpoint root mismatch for {job_id}")


def _load_sidecar(path: Path, config: Mapping[str, Any], job_id: str) -> Dict[str, Any]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as payload:
            lags = np.asarray(payload["lags"]).reshape(-1)
            source = str(np.asarray(payload["acf_source"]).item()).lower()
            values = np.asarray(payload["acf_values"])
            full = np.asarray(payload["acf_full_values"])
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"invalid ACF sidecar for {job_id}: {path}: {exc}") from exc
    if lags.dtype.kind not in "iu":
        raise ValueError(f"sidecar lags are not integer for {job_id}")
    sidecar_lags = [int(item) for item in lags.tolist()]
    expected_lags = _positive_lags(config.get("mamba_acf_prior_lags"), f"prior lags {job_id}")
    if sidecar_lags != expected_lags:
        raise ValueError(f"sidecar lag order differs from config for {job_id}")
    if source != "raw":
        raise ValueError(f"sidecar source is not raw for {job_id}")
    if values.ndim != 2 or values.shape[1] != len(sidecar_lags):
        raise ValueError(f"sidecar compact shape is invalid for {job_id}")
    if full.ndim != 2 or full.shape[0] != values.shape[0]:
        raise ValueError(f"sidecar full shape is invalid for {job_id}")
    if not bool(__import__("numpy").isfinite(values).all()) or not bool(__import__("numpy").isfinite(full).all()):
        raise ValueError(f"sidecar contains non-finite values for {job_id}")
    return {
        "lags": sidecar_lags,
        "source": source,
        "compact_shape": list(values.shape),
        "full_shape": list(full.shape),
    }


def _runtime_diagnostics(logs: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    return {key: _summary_series(logs, key, path) for key in RUNTIME_LOG_KEYS}


def _relation_diagnostics(logs: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    if logs.get("acf_relation_objective_version") != 3:
        raise ValueError(f"relation objective is not v3: {path}")
    if str(logs.get("acf_relation_target_mode")) != "hard_selected":
        raise ValueError(f"relation target mode mismatch: {path}")
    return {key: _summary_series(logs, key, path) for key in RELATION_LOG_KEYS}


def _state_scalar(state: Mapping[str, Any], key: str, path: Path) -> Optional[float]:
    value = state.get(key)
    if value is None:
        return None
    array = _state_array(value, path, key)
    if array.size != 1:
        raise ValueError(f"checkpoint scalar {key} is not scalar: {path}")
    return float(array.reshape(-1)[0])


def _load_phase(
    plan: Mapping[str, Any], plan_path: Path, phase_name: str, mode: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase = _phase_from_plan(plan, phase_name)
    phase_root = _phase_root_from_plan(plan, phase)
    validation_path = phase_root / "phase_validation.json"
    if not validation_path.is_file():
        raise NotRunError(f"training not run: missing {validation_path}")
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ValueError(f"invalid phase validation: {validation_path}")
    if validation.get("plan") and Path(str(validation["plan"])).resolve() != plan_path.resolve():
        raise ValueError(f"phase validation points to another plan: {phase_name}")
    plan_hash = _sha256(plan_path)
    if validation.get("plan_sha256") != plan_hash:
        raise ValueError(f"phase validation plan hash mismatch: {phase_name}")
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file():
        raise NotRunError(f"training not run: missing {manifest_path}")
    if validation.get("manifest") and Path(str(validation["manifest"])).resolve() != manifest_path.resolve():
        raise ValueError(f"phase validation points to another manifest: {phase_name}")
    if validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"manifest changed after validation: {phase_name}")
    records = _manifest_records(phase_root, validation)
    expected_jobs = _int(validation.get("expected_jobs", -1), f"expected jobs {phase_name}")
    if expected_jobs != EXPECTED_JOBS_PER_PHASE or len(records) != expected_jobs:
        raise ValueError(f"unexpected manifest coverage for {phase_name}")
    if _int(validation.get("manifest_unique_jobs", -1), f"unique jobs {phase_name}") != expected_jobs:
        raise ValueError(f"manifest unique-job count mismatch: {phase_name}")
    if validation.get("malformed_manifest_lines", 0) != 0 or validation.get("errors") not in ([], None):
        raise ValueError(f"phase validation contains errors: {phase_name}")
    if validation.get("status_counts") != {"ok": expected_jobs}:
        raise ValueError(f"phase status counts are not all ok: {phase_name}")
    ids = [str(item.get("job_id", "")) for item in records]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"manifest job IDs are missing/duplicated: {phase_name}")
    provenance_meta = _validate_provenance(
        phase_root, phase, plan_path, plan, validation, records
    )
    hash_map = validation.get("artifact_hashes_by_job")
    if not isinstance(hash_map, Mapping) or set(hash_map) != set(ids):
        raise ValueError(f"phase artifact hash map does not cover manifest: {phase_name}")
    expected_phase = _phase_contract(plan, phase_name, mode)
    # Re-expand the frozen plan so each manifest row is checked against the
    # registered coordinate and configuration hash, not just its label.
    registered_jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
    registered_by_coord = {
        (str(item["dataset"]), int(item["pred_len"]), int(item["seed"])): item
        for item in registered_jobs
    }
    if len(registered_by_coord) != EXPECTED_JOBS_PER_PHASE:
        raise ValueError(f"registered coordinate expansion is incomplete: {phase_name}")

    result_root = (phase_root / "results").resolve()
    checkpoint_root = (phase_root / "checkpoints").resolve()
    rows: List[Dict[str, Any]] = []
    observed: set[Tuple[str, int]] = set()
    for record in records:
        job_id = str(record.get("job_id"))
        if record.get("phase") != phase_name or record.get("status") != "ok":
            raise ValueError(f"manifest record is not successful: {job_id}")
        dataset = str(record.get("dataset"))
        seed = _int(record.get("seed"), f"{job_id} seed")
        horizon = _int(record.get("pred_len"), f"{job_id} pred_len")
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS or horizon != EXPECTED_HORIZON:
            raise ValueError(f"unexpected coordinate: {job_id}")
        coordinate = (dataset, seed)
        if coordinate in observed:
            raise ValueError(f"duplicate coordinate: {phase_name} {coordinate}")
        observed.add(coordinate)
        registered = registered_by_coord.get((dataset, horizon, seed))
        if registered is None:
            raise ValueError(f"manifest coordinate is not registered: {job_id}")
        if record.get("model") != EXPECTED_ALIAS:
            raise ValueError(f"manifest alias mismatch for {job_id}")
        if record.get("config_hash") != _config_hash(registered, record.get("provenance_id")):
            raise ValueError(f"manifest config hash mismatch for {job_id}")
        _validate_command(record, mode, job_id, checkpoint_root)
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected_artifact for {job_id}")
        metrics_path = Path(str(expected_artifact)).expanduser().resolve()
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not _inside(metrics_path, result_root):
            raise ValueError(f"metrics artifact missing/outside result root for {job_id}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        record_hashes = record.get("artifact_hashes")
        phase_hashes = hash_map.get(job_id)
        if not isinstance(record_hashes, Mapping) or not isinstance(phase_hashes, Mapping):
            raise ValueError(f"missing artifact hash map for {job_id}")
        if dict(record_hashes) != dict(phase_hashes):
            raise ValueError(f"manifest/validation artifact hashes differ for {job_id}")
        paths = _artifact_paths(result_dir, checkpoint_dir, record_hashes)
        actual_hashes = {name: _sha256(path) for name, path in paths.items()}
        if actual_hashes != dict(record_hashes):
            raise ValueError(f"artifact hash mismatch after validation for {job_id}")
        config = _read_json(result_dir / "config.json")
        config_info = _check_config(config, mode, dataset, seed, job_id)
        if config_info["prior_sha256"] != actual_hashes.get("acf_prior.npz", "").lower():
            raise ValueError(f"config/sidecar hash mismatch for {job_id}")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section for {job_id}")
        mse = _finite(scaled.get("mse"), f"{job_id} scaled mse")
        mae = _finite(scaled.get("mae"), f"{job_id} scaled mae")
        sidecar = _load_sidecar(result_dir / "acf_prior.npz", config, job_id)
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        relation = _relation_diagnostics(logs, logs_path)
        runtime = _runtime_diagnostics(logs, logs_path)
        effective_lambda = _summary_series(logs, "effective_lambda", logs_path)
        if not math.isclose(effective_lambda["last"], EXPECTED_LAMBDA, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"effective lambda mismatch for {job_id}")
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        state = _checkpoint_state(checkpoint_path)
        gate_state = _gate_state_contract(state, checkpoint_path)
        mode_diag = logs.get("acf_memory_gate_variable_token")
        if mode_diag is not None:
            mode_values = mode_diag if isinstance(mode_diag, list) else [mode_diag]
            expected_value = 1.0 if mode == "variable_token" else 0.0
            for value in mode_values:
                if not math.isclose(_finite(value, f"{job_id} gate mode diagnostic"), expected_value, abs_tol=1e-9):
                    raise ValueError(f"runtime gate mode diagnostic mismatch for {job_id}")
        rows.append({
            "phase": phase_name,
            "arm": mode,
            "dataset": dataset,
            "seed": seed,
            "horizon": horizon,
            "mse": mse,
            "mae": mae,
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "epochs_ran": _int(record.get("epochs_ran"), f"{job_id} epochs_ran"),
            "best_epoch": _int(record.get("best_epoch"), f"{job_id} best_epoch"),
            "job_id": job_id,
            "sidecar_sha256": actual_hashes["acf_prior.npz"],
            "sidecar": sidecar,
            "config_signature": config_info["signature"],
            "effective_lags": config_info["effective_lags"],
            "relation_lags": config_info["relation_lags"],
            "selected_relation_lags": config_info["selected_relation_lags"],
            "relation_diagnostics": relation,
            "runtime_diagnostics": runtime,
            "effective_lambda": effective_lambda,
            "gate_state": gate_state,
            "autocon_loss": _summary_series(logs, "autocon_loss", logs_path),
            "config_sha256": actual_hashes["config.json"],
            "metrics_sha256": actual_hashes["metrics.npy"],
        })
    expected_coordinates = {
        (dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS
    }
    if observed != expected_coordinates:
        raise ValueError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows, {
        "phase": phase_name,
        "arm": mode,
        "root": str(phase_root),
        "validation": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "provenance": provenance_meta,
        "expected_jobs": EXPECTED_JOBS_PER_PHASE,
        "observed_jobs": len(rows),
        "valid": True,
    }


def _paired_summary(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("empty paired delta vector")
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    t_stat: float
    p_value: Optional[float] = None
    if math.isfinite(sd) and sd > 0:
        t_stat = mean / (sd / math.sqrt(len(numbers)))
    elif mean == 0:
        t_stat = 0.0
    else:
        t_stat = math.copysign(math.inf, mean)
    if scipy_stats is not None and len(numbers) > 1:
        try:
            test = scipy_stats.ttest_1samp(numbers, 0.0)
            t_stat, p_value = float(test.statistic), float(test.pvalue)
        except Exception:
            p_value = None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
        "delta_values": numbers,
    }


def _relative(delta: float, reference: float) -> float:
    return delta / reference if reference else math.nan


def _contrast(
    pairs: Sequence[Mapping[str, Any]],
    metric: str,
    reference_key: str = "hidden",
    treatment_key: str = "variable_token",
) -> Dict[str, Any]:
    deltas = [float(item[f"{treatment_key}_{metric}"] - item[f"{reference_key}_{metric}"]) for item in pairs]
    reference = [float(item[f"{reference_key}_{metric}"]) for item in pairs]
    treatment = [float(item[f"{treatment_key}_{metric}"]) for item in pairs]
    summary = _paired_summary(deltas)
    summary.update({
        "reference": reference_key,
        "treatment": treatment_key,
        "reference_mean": statistics.fmean(reference),
        "treatment_mean": statistics.fmean(treatment),
        "relative_delta_mean": _relative(summary["mean"], statistics.fmean(reference)),
    })
    return summary


def _make_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    indexed = {(str(row["arm"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    expected = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    pairs: List[Dict[str, Any]] = []
    for dataset, seed in sorted(expected):
        hidden = indexed.get(("hidden", dataset, seed))
        variable = indexed.get(("variable_token", dataset, seed))
        if hidden is None or variable is None:
            raise ValueError(f"incomplete paired coordinate: {dataset} seed {seed}")
        if hidden["sidecar_sha256"] != variable["sidecar_sha256"]:
            raise ValueError(f"sidecar hash differs at {dataset} seed {seed}")
        if hidden["effective_lags"] != variable["effective_lags"]:
            raise ValueError(f"effective lags differ at {dataset} seed {seed}")
        if hidden["relation_lags"] != variable["relation_lags"]:
            raise ValueError(f"relation lags differ at {dataset} seed {seed}")
        if hidden["config_signature"] != variable["config_signature"]:
            raise ValueError(f"common config differs at {dataset} seed {seed}")
        pairs.append({
            "dataset": dataset,
            "seed": seed,
            "sidecar_sha256": hidden["sidecar_sha256"],
            "hidden_mse": hidden["mse"],
            "variable_token_mse": variable["mse"],
            "delta_mse": variable["mse"] - hidden["mse"],
            "relative_delta_mse": _relative(variable["mse"] - hidden["mse"], hidden["mse"]),
            "hidden_mae": hidden["mae"],
            "variable_token_mae": variable["mae"],
            "delta_mae": variable["mae"] - hidden["mae"],
            "relative_delta_mae": _relative(variable["mae"] - hidden["mae"], hidden["mae"]),
            "hidden_elapsed_sec": hidden["elapsed_sec"],
            "variable_token_elapsed_sec": variable["elapsed_sec"],
            "delta_elapsed_sec": variable["elapsed_sec"] - hidden["elapsed_sec"],
            "hidden_job_id": hidden["job_id"],
            "variable_token_job_id": variable["job_id"],
        })
    return pairs


def _validate_invariants(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected = {
        (mode, dataset, seed)
        for _, mode in EXPECTED_PHASES
        for dataset in EXPECTED_DATASETS
        for seed in EXPECTED_SEEDS
    }
    observed = {(str(row.get("arm")), str(row.get("dataset")), int(row.get("seed"))) for row in rows}
    if observed != expected or len(rows) != EXPECTED_JOBS:
        raise ValueError("row matrix does not contain exactly the registered 12 coordinates")
    if any(row.get("gate_state", {}).get("gate_parameter_count") != 3168 for row in rows):
        raise ValueError("gate parameter state contract failed")
    _make_pairs(rows)
    return {
        "paired_coordinates": EXPECTED_JOBS_PER_PHASE,
        "sidecar_hashes_equal_per_coordinate": True,
        "effective_lags_equal_per_coordinate": True,
        "relation_lags_equal_per_coordinate": True,
        "common_config_equal_per_coordinate": True,
        "hidden_shape": "B,H,1",
        "variable_token_shape": "B,H,C",
        "gate_parameter_count_equal": True,
        "hidden_mode_observed": True,
        "variable_token_mode_observed": True,
    }


def _stopping_decision(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    favorable = sum(float(item["delta_mse"]) < 0 for item in pairs)
    by_dataset = {
        dataset: sum(float(item["delta_mse"]) < 0 for item in pairs if item["dataset"] == dataset)
        for dataset in EXPECTED_DATASETS
    }
    mean_rel_by_dataset = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        ref = statistics.fmean(item["hidden_mse"] for item in subset)
        delta = statistics.fmean(item["delta_mse"] for item in subset)
        mean_rel_by_dataset[dataset] = _relative(delta, ref)
    meets = (
        favorable >= 4
        and all(value >= 2 for value in by_dataset.values())
        and all(value <= 0.01 for value in mean_rel_by_dataset.values())
    )
    return {
        "criterion": "variable-token mode favorable on >=4/6 paired MSE coordinates, >=2/3 seeds in each dataset, and no dataset mean degradation >1% (descriptive only)",
        "favorable_coordinates": favorable,
        "favorable_coordinates_by_dataset": by_dataset,
        "mean_relative_delta_by_dataset": mean_rel_by_dataset,
        "meets_descriptive_screen": meets,
        "action": "continue only to a separately reviewed frozen replication" if meets else "stop exploratory screen; do not add arms, seeds, horizons, or interactions",
    }


def build_readout(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    validation: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = [dict(row) for row in rows]
    invariants = _validate_invariants(normalized)
    pairs = _make_pairs(normalized)
    contrasts: Dict[str, Any] = {
        "variable_token_minus_hidden": {
            "mse": _contrast(pairs, "mse"),
            "mae": _contrast(pairs, "mae"),
            "elapsed_sec": _contrast(pairs, "elapsed_sec"),
            "by_dataset": {},
        }
    }
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        contrasts["variable_token_minus_hidden"]["by_dataset"][dataset] = {
            "n": len(subset),
            "mse": _contrast(subset, "mse"),
            "mae": _contrast(subset, "mae"),
            "elapsed_sec": _contrast(subset, "elapsed_sec"),
        }
    diagnostics: Dict[str, Any] = {}
    for arm in ("hidden", "variable_token"):
        subset = [row for row in normalized if row["arm"] == arm]
        diagnostics[arm] = {
            "n": len(subset),
            "state_numel_mean": statistics.fmean(row["gate_state"]["state_numel"] for row in subset),
            "gate_parameter_count_values": sorted({row["gate_state"]["gate_parameter_count"] for row in subset}),
            "gate_bias_mean": statistics.fmean(row["gate_state"]["gate_bias_mean"] for row in subset),
            "gate_weight_l2_mean": statistics.fmean(row["gate_state"]["gate_weight_l2"] for row in subset),
            "variable_residual_scale_mean": statistics.fmean(
                row["gate_state"]["variable_residual_scale"]
                for row in subset
                if row["gate_state"]["variable_residual_scale"] is not None
            ),
            "runtime_gate_mean": statistics.fmean(
                row["runtime_diagnostics"]["acf_runtime_gate_mean"]["mean"] for row in subset
            ),
            "runtime_gate_std": statistics.fmean(
                row["runtime_diagnostics"]["acf_runtime_gate_std"]["mean"] for row in subset
            ),
            "effective_memory_rms": statistics.fmean(
                row["runtime_diagnostics"]["acf_effective_memory_rms"]["mean"] for row in subset
            ),
            "relation_sign_accuracy": statistics.fmean(
                row["relation_diagnostics"]["acf_relation_global_sign_accuracy"]["mean"] for row in subset
            ),
        }
    plan_path = Path(str(plan.get("_path", ""))).resolve()
    validation_list = list(validation or [])
    source_drift = {
        item["phase"]: list(item.get("provenance", {}).get("live_source_mismatches", []))
        for item in validation_list
    }
    snapshot_valid = all(
        bool(item.get("provenance", {}).get("source_tree_sha256"))
        and not item.get("provenance", {}).get("snapshot_hash_error")
        for item in validation_list
    )
    return {
        "schema_version": 1,
        "status": "exploratory_mamba_acf_memory_gate_mode_screen_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "phases": [name for name, _ in EXPECTED_PHASES],
        "rows": normalized,
        "pairs": pairs,
        "validation": validation_list,
        "replay": {
            "phase_validations_all_valid": all(item.get("valid") is True for item in validation_list),
            "artifact_hashes_replayed": True,
            "immutable_snapshot_hashes_valid": snapshot_valid,
            "live_source_mismatches_by_phase": source_drift,
            "live_source_matches_snapshot": all(not values for values in source_drift.values()),
        },
        "invariants": invariants,
        "contrasts": contrasts,
        "stopping_decision": _stopping_decision(pairs),
        "resource_contract": plan.get("resource_contract"),
        "evidence_boundary": "Descriptive exploratory gate-source evidence only; no test-based selection, pooled cross-plan inference, or publication claim.",
    }


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def markdown(result: Mapping[str, Any]) -> str:
    contrast = result["contrasts"]["variable_token_minus_hidden"]
    lines = [
        "# MambaACF Memory-Gate Mode Screen",
        "",
        "> Exploratory, non-confirmatory readout. The two arms share the variable-token MambaACF AutoCon trunk, train-only raw ACF sidecar, relation objective, data, seeds, and budget; only the memory-gate source changes.",
        "",
        "## Validation",
        "",
        f"- Plan: `{result['plan']}`; SHA-256 `{result['plan_sha256']}`.",
        f"- Completed rows: `{len(result['rows'])}/{EXPECTED_JOBS}`; pairing unit: dataset x seed at H96.",
        "- Negative accuracy deltas favor `variable_token`; all p-values are descriptive and are not used for selection.",
        "",
        "| Phase | Jobs | Manifest SHA-256 | Provenance ID | Source-tree SHA-256 |",
        "|---|---:|---|---|---|",
    ]
    for item in result.get("validation", []):
        provenance = item.get("provenance", {})
        lines.append(
            f"| `{item.get('phase')}` | {item.get('observed_jobs')}/{item.get('expected_jobs')} | `{item.get('manifest_sha256')}` | `{provenance.get('provenance_id')}` | `{provenance.get('source_tree_sha256')}` |"
        )
    lines += [
        "",
        "## Per-seed paired metrics",
        "",
        "| Dataset | Seed | Hidden MSE | Variable-token MSE | Delta | Hidden MAE | Variable-token MAE | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["pairs"]:
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {_fmt(row['hidden_mse'])} | {_fmt(row['variable_token_mse'])} | {_fmt(row['delta_mse'])} | {_fmt(row['hidden_mae'])} | {_fmt(row['variable_token_mae'])} | {_fmt(row['delta_mae'])} |"
        )
    lines += [
        "",
        "## Paired summaries",
        "",
        "| Dataset | Metric | n | Hidden mean | Variable-token mean | Delta | Relative | Wins | p (descriptive) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ["pooled descriptive", *EXPECTED_DATASETS]:
        values = contrast if dataset.startswith("pooled") else contrast["by_dataset"][dataset]
        for metric in ("mse", "mae", "elapsed_sec"):
            summary = values[metric]
            lines.append(
                f"| {dataset} | {metric} | {summary['n']} | {_fmt(summary['reference_mean'])} | {_fmt(summary['treatment_mean'])} | {_fmt(summary['mean'])} | {_fmt(100 * summary['relative_delta_mean'], 3)}% | {summary['wins_lower']}/{summary['n']} | {_fmt(summary.get('two_sided_p'), 4)} |"
            )
    lines += [
        "",
        "## Gate diagnostics",
        "",
        "| Arm | Gate shape | Gate params | State numel | Runtime gate mean | Runtime gate std | Effective memory RMS | Relation sign accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    shapes = {"hidden": "B,H,1", "variable_token": "B,H,C"}
    for arm in ("hidden", "variable_token"):
        diag = result["diagnostics"][arm]
        lines.append(
            f"| {arm} | {shapes[arm]} | {diag['gate_parameter_count_values']} | {_fmt(diag['state_numel_mean'], 1)} | {_fmt(diag['runtime_gate_mean'])} | {_fmt(diag['runtime_gate_std'])} | {_fmt(diag['effective_memory_rms'])} | {_fmt(diag['relation_sign_accuracy'])} |"
        )
    resource = result.get("resource_contract") or {}
    lines += [
        "",
        "## Resource boundary",
        "",
        f"- Both modes use `{resource.get('gate_parameters_each_mode', 'n/a')}` gate parameters. Hidden output is `[B,H,1]`; variable-token output is `[B,H,C]` and scales with channel count.",
        "- Gate-only MAC/activation estimates exclude backend overhead and must not be converted directly into end-to-end training speed claims.",
        "- The resource benchmark and its 1.25x latency/peak-memory budget are separate from the 12 training jobs.",
        "",
        "## Screening decision",
        "",
        f"- Criterion: `{result['stopping_decision']['criterion']}`.",
        f"- Favorable coordinates: `{result['stopping_decision']['favorable_coordinates']}/6`; by dataset `{result['stopping_decision']['favorable_coordinates_by_dataset']}`.",
        f"- Action: **{result['stopping_decision']['action']}**.",
        "- Any result remains exploratory and cannot support a publication claim without an independently reviewed frozen replication.",
        "",
        "## Provenance boundary",
        "",
        "- Artifact hashes and immutable snapshots are authoritative for historical results. Any post-launch live-source mismatch is reported in the JSON and does not become a current-tree rerun.",
        "- This analyzer never creates a result file when the registered training artifacts are absent; it reports `NOT_RUN` instead.",
        "",
    ]
    return "\n".join(lines)


def _not_run_message(plan: Mapping[str, Any]) -> str:
    missing = _missing_training_artifacts(plan)
    if not missing:
        return "NOT_RUN: no complete training artifacts found for the registered gate-mode screen"
    preview = "; ".join(missing[:6])
    suffix = " ..." if len(missing) > 6 else ""
    return f"NOT_RUN: registered gate-mode screen has no complete training artifacts ({preview}{suffix})"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/research/mamba_acf_memory_gate_mode_screen_plan_20260902.json"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", "--output-markdown", dest="output_md", type=Path, default=None)
    args = parser.parse_args(argv)
    plan_path = _resolve(args.plan)
    try:
        plan = dict(_read_json(plan_path))
        plan["_path"] = str(plan_path)
        validate_plan_contract(plan)
        missing = _missing_training_artifacts(plan)
        if missing:
            print(_not_run_message(plan), file=sys.stderr)
            return 3
        rows: List[Dict[str, Any]] = []
        validation: List[Dict[str, Any]] = []
        for phase_name, mode in EXPECTED_PHASES:
            phase_rows, phase_meta = _load_phase(plan, plan_path, phase_name, mode)
            rows.extend(phase_rows)
            validation.append(phase_meta)
        result = build_readout(plan, rows, validation)
        output_json = _resolve(
            args.output_json
            or Path("artifacts/paper_final/mamba_acf_memory_gate_mode_screen_20260902/readout.json")
        )
        output_md = _resolve(
            args.output_md
            or Path("docs/research/mamba_acf_memory_gate_mode_screen_readout_20260902.md")
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        output_md.write_text(markdown(result), encoding="utf-8")
        print(json.dumps({
            "rows": len(rows),
            "paired_coordinates": len(result["pairs"]),
            "output_json": str(output_json),
            "output_md": str(output_md),
        }, ensure_ascii=True))
        return 0
    except NotRunError as exc:
        print(f"NOT_RUN: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"analyze_mamba_acf_memory_gate_mode_screen: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
