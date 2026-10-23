#!/usr/bin/env python3
"""Audit and read out the paired horizon-amplitude calibration screen.

The screen has two registered phases that share the complete MambaACF
AutoCon/relation pipeline and differ only in the static rank-2 horizon
amplitude adapter.  This reader is deliberately strict about phase
validation, artifact hashes, train-only sidecar identity, and checkpoint
state layout before it computes paired descriptive contrasts.

It is an exploratory reader: it never selects a model from test metrics and
does not make a publication or universal-accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional analysis dependency
    scipy_stats = None

from scripts.analyze_mamba_relation_null_control import (
    _finite,
    _hash_map_for_record,
    _manifest_records,
    _read_json,
    _resolve,
    _sha256,
    _summary_series,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PHASES = (
    ("horizon_calibration_off_raw_h96", "off", False),
    ("horizon_calibration_on_raw_h96", "on", True),
)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
EXPECTED_RANK = 2

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

# These fields form the one-factor contract.  Run-specific paths, setting
# names, and plan job ids are intentionally excluded because the two phases
# must write to different output roots.
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
    "mamba_variable_group_size",
    "mamba_acf_channel_order",
    "mamba_acf_channel_conditioning",
    "mamba_acf_channel_gate",
    "mamba_acf_repr_fusion",
    "mamba_patch_mixer",
    "mamba_local_mixer",
    "mamba_low_rank_head",
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
    "mamba_acf_effective_lags",
    "mamba_acf_effective_lag_source",
    "mamba_acf_effective_lag_fallback",
    "mamba_acf_relation_effective_lags",
    "mamba_acf_relation_lag_source",
    "mamba_acf_relation_lag_fallback",
    "mamba_acf_prior_sha256",
    "mamba_acf_prior_shape",
    "mamba_acf_prior_lags",
    "mamba_acf_prior_full_shape",
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _explicit_bool(value: Any, label: str) -> bool:
    """Parse persisted booleans without treating ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{label} is not an explicit boolean: {value!r}")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    return value


def _positive_lags(value: Any, label: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    output: List[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} contains a boolean")
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-integer") from exc
        if parsed <= 0 or parsed != item:
            raise ValueError(f"{label} contains a non-positive/non-integral lag")
        output.append(parsed)
    if len(set(output)) != len(output):
        raise ValueError(f"{label} contains duplicate lags")
    return output


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    """Load a checkpoint on CPU and normalize an optional DataParallel prefix."""

    try:
        import torch

        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility
            value = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - surfaced as an audit error
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


def _tensor_shape(state: Mapping[str, Any], key: str, path: Path) -> Tuple[int, ...]:
    value = state.get(key)
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"checkpoint is missing tensor {key}: {path}")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid tensor shape for {key}: {path}") from exc


def _tensor_array(value: Any, path: Path) -> Any:
    try:
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(f"cannot read checkpoint tensor at {path}: {exc}") from exc
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"calibration coefficients are not finite rank-2 data: {path}")
    return array


def _calibration_contract(
    state: Mapping[str, Any], path: Path, enabled: bool, lag_count: int
) -> Dict[str, Any]:
    key = "acf_horizon_calibration.coefficients"
    calibration_keys = sorted(
        name for name in state if str(name).startswith("acf_horizon_calibration.")
    )
    if not enabled:
        if calibration_keys:
            raise ValueError(
                f"calibration-off checkpoint contains calibration tensors {calibration_keys}: {path}"
            )
        return {
            "enabled": False,
            "rank": EXPECTED_RANK,
            "lag_count": lag_count,
            "parameter_count": 0,
            "coefficient_shape": None,
            "amplitude": None,
        }
    if calibration_keys != [key]:
        raise ValueError(
            f"calibration-on checkpoint tensors {calibration_keys} != [{key!r}]: {path}"
        )
    expected_shape = (lag_count, EXPECTED_RANK)
    actual_shape = _tensor_shape(state, key, path)
    if actual_shape != expected_shape:
        raise ValueError(
            f"calibration coefficient shape {actual_shape} != {expected_shape}: {path}"
        )
    coefficients = _tensor_array(state[key], path)
    if coefficients.shape != expected_shape:
        raise ValueError(f"calibration coefficient array shape mismatch: {path}")
    try:
        import numpy as np

        steps = np.arange(1, EXPECTED_HORIZON + 1, dtype=float)
        basis = np.column_stack(
            (np.ones_like(steps), 2.0 * steps / (EXPECTED_HORIZON + 1.0) - 1.0)
        )
        logits = basis @ coefficients.T
        clipped = np.clip(logits, -math.log(2.0), math.log(2.0))
        amplitudes = np.exp(clipped)
        if not np.all(np.isfinite(amplitudes)):
            raise ValueError(f"calibration amplitudes are not finite: {path}")
        if float(amplitudes.min()) < 0.5 - 1e-7 or float(amplitudes.max()) > 2.0 + 1e-7:
            raise ValueError(f"calibration amplitudes exceed the registered [0.5, 2] bound: {path}")
        amplitude_summary = {
            "mean": float(amplitudes.mean()),
            "std": float(amplitudes.std()),
            "min": float(amplitudes.min()),
            "max": float(amplitudes.max()),
        }
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency
        raise ValueError("numpy is required for calibration state auditing") from exc
    return {
        "enabled": True,
        "rank": EXPECTED_RANK,
        "lag_count": lag_count,
        "parameter_count": int(coefficients.size),
        "coefficient_shape": list(actual_shape),
        "coefficient_mean": float(coefficients.mean()),
        "coefficient_std": float(coefficients.std()),
        "amplitude": amplitude_summary,
    }


def _optional_summary(
    logs: Mapping[str, Any], key: str, path: Path, *, expect_numeric: bool
) -> Dict[str, Any]:
    """Summarize a diagnostic that is numeric for ``on`` and null for ``off``."""

    values = logs.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"missing calibration diagnostic {key}: {path}")
    numbers: List[float] = []
    null_count = 0
    for value in values:
        if value is None:
            null_count += 1
        else:
            numbers.append(_finite(value, f"{path.name}:{key}"))
    if expect_numeric and (not numbers or null_count):
        raise ValueError(
            f"calibration-on diagnostic {key} has null/non-numeric epochs: {path}"
        )
    if not expect_numeric and numbers:
        raise ValueError(
            f"calibration-off diagnostic {key} unexpectedly has numeric epochs: {path}"
        )
    result: Dict[str, Any] = {
        "n": len(values),
        "finite_n": len(numbers),
        "null_n": null_count,
        "mean": None,
        "min": None,
        "max": None,
        "last": None,
    }
    if numbers:
        result.update(
            mean=statistics.fmean(numbers),
            min=min(numbers),
            max=max(numbers),
            last=numbers[-1],
        )
    return result


def _check_config(
    config: Mapping[str, Any], enabled: bool, job_id: str
) -> Dict[str, Any]:
    expected: Dict[str, Any] = {
        "model": "MambaACFAutoCon",
        "features": "M",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": EXPECTED_HORIZON,
        "AutoCon": True,
        "AutoCon_lambda": 0.03,
        "AutoCon_acf_source": "raw",
        "mamba_causal_embedding": True,
        "mamba_use_acf_memory": True,
        "mamba_dynamic_acf": False,
        "mamba_acf_lag_mode": "train_peaks_available",
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_lag_mask": True,
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_variable_mixer": False,
        "mamba_variable_group_size": 1,
        "mamba_acf_horizon_calibration_rank": EXPECTED_RANK,
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
        elif key in {"seq_len", "label_len", "pred_len", "mamba_acf_relation_min_lag", "mamba_variable_group_size", "mamba_acf_horizon_calibration_rank"}:
            try:
                actual = int(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} is not an integer for {job_id}") from exc
        if actual != wanted:
            raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {wanted!r}")
    actual_enabled = _explicit_bool(
        config.get("mamba_acf_horizon_calibration"),
        f"config mamba_acf_horizon_calibration for {job_id}",
    )
    if actual_enabled != enabled:
        raise ValueError(
            f"config calibration flag mismatch for {job_id}: {actual_enabled!r} != {enabled!r}"
        )
    requested_lags = _positive_lags(config.get("mamba_acf_lags"), "config mamba_acf_lags")
    effective_lags = _positive_lags(
        config.get("mamba_acf_effective_lags"), "config mamba_acf_effective_lags"
    )
    prior_lags = _positive_lags(
        config.get("mamba_acf_prior_lags"), "config mamba_acf_prior_lags"
    )
    if sorted(effective_lags) != sorted(prior_lags):
        raise ValueError(f"config effective/prior lag mismatch for {job_id}")
    if len(effective_lags) != 3:
        raise ValueError(
            f"expected three resolved ACF memory lag slots, got {len(effective_lags)} for {job_id}"
        )
    if str(config.get("mamba_acf_effective_lag_source")) != "train_peaks_available":
        raise ValueError(f"config effective lag source is not train_peaks_available for {job_id}")
    if _explicit_bool(config.get("mamba_acf_effective_lag_fallback"), "effective lag fallback"):
        raise ValueError(f"config uses an ACF lag fallback for {job_id}")
    relation_lags = _positive_lags(
        config.get("mamba_acf_relation_effective_lags"),
        "config mamba_acf_relation_effective_lags",
    )
    if str(config.get("mamba_acf_relation_lag_source")) != "train_peak_subset":
        raise ValueError(f"config relation lag source is not train_peak_subset for {job_id}")
    if _explicit_bool(config.get("mamba_acf_relation_lag_fallback"), "relation lag fallback"):
        raise ValueError(f"config uses a relation lag fallback for {job_id}")
    if any(lag < 96 for lag in relation_lags):
        raise ValueError(f"relation lag below the registered 96-step floor for {job_id}")
    prior_shape = config.get("mamba_acf_prior_shape")
    if not isinstance(prior_shape, list) or len(prior_shape) != 2:
        raise ValueError(f"config prior shape is invalid for {job_id}")
    try:
        prior_shape = [int(item) for item in prior_shape]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config prior shape is invalid for {job_id}") from exc
    if prior_shape[1] != len(effective_lags):
        raise ValueError(f"config prior width does not match resolved lag count for {job_id}")
    if config.get("mamba_acf_prior_artifact") != "acf_prior.npz":
        raise ValueError(f"config prior artifact is not acf_prior.npz for {job_id}")
    prior_sha = str(config.get("mamba_acf_prior_sha256", ""))
    if len(prior_sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in prior_sha):
        raise ValueError(f"config prior sha256 is invalid for {job_id}")

    signature: Dict[str, Any] = {}
    for key in COMMON_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"one-factor config field {key} is missing for {job_id}")
        signature[key] = _canonical(config[key])
    # The adapter flag itself is the registered treatment variable and is
    # compared separately.  Its rank remains in the common signature.
    return {
        "requested_lags": requested_lags,
        "effective_lags": effective_lags,
        "relation_lags": relation_lags,
        "prior_sha256": prior_sha,
        "prior_shape": prior_shape,
        "signature": signature,
    }


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
        raise ValueError(f"ACF sidecar lags are not integer for {job_id}")
    sidecar_lags = [int(item) for item in lags.tolist()]
    expected_lags = _positive_lags(config.get("mamba_acf_prior_lags"), "config prior lags")
    if sidecar_lags != expected_lags:
        raise ValueError(f"sidecar lag order differs from config for {job_id}")
    if source != "raw":
        raise ValueError(f"sidecar source is not raw for {job_id}")
    if values.ndim != 2 or values.shape[1] != len(sidecar_lags):
        raise ValueError(f"sidecar compact shape is invalid for {job_id}")
    if full.ndim != 2 or full.shape[0] != values.shape[0]:
        raise ValueError(f"sidecar complete table shape is invalid for {job_id}")
    return {
        "lags": sidecar_lags,
        "source": source,
        "compact_shape": list(values.shape),
        "full_shape": list(full.shape),
    }


def _load_phase(
    plan: Mapping[str, Any], phase_name: str, arm: str, enabled: bool
) -> List[Dict[str, Any]]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    phase = next(
        (item for item in phases if isinstance(item, Mapping) and item.get("name") == phase_name),
        None,
    )
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
    seen: set[Tuple[str, int]] = set()
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
        job_id = str(record.get("job_id"))
        if record.get("model") != "MambaACFAutoCon":
            raise ValueError(f"manifest model mismatch: {job_id}")
        if not math.isclose(float(record.get("autocon_lambda")), 0.03, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"manifest AutoCon lambda mismatch: {job_id}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected artifact: {job_id}")
        metrics_path = _resolve(str(expected_artifact))
        if (
            metrics_path.name != "metrics.npy"
            or not metrics_path.is_file()
            or not _inside(metrics_path, result_root)
        ):
            raise ValueError(f"metrics artifact outside result root: {job_id}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        config = _read_json(result_dir / "config.json")
        config_info = _check_config(config, enabled, job_id)
        if config_info["prior_sha256"] != hashes.get("acf_prior.npz"):
            raise ValueError(f"config/manifest sidecar hash mismatch: {job_id}")
        sidecar_info = _load_sidecar(result_dir / "acf_prior.npz", config, job_id)
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {job_id}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        relation_diagnostics = {
            key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS
        }
        runtime_diagnostics = {
            key: _summary_series(logs, key, logs_path) for key in RUNTIME_LOG_KEYS
        }
        calibration_diagnostics = {
            key: _optional_summary(logs, key, logs_path, expect_numeric=enabled)
            for key in CALIBRATION_LOG_KEYS
        }
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        state = _checkpoint_state(checkpoint_path)
        state_info = _calibration_contract(
            state, checkpoint_path, enabled, len(config_info["effective_lags"])
        )
        rows.append(
            {
                "phase": phase_name,
                "arm": arm,
                "dataset": dataset,
                "seed": seed,
                "horizon": EXPECTED_HORIZON,
                "mse": _finite(scaled.get("mse"), f"{job_id} scaled mse"),
                "mae": _finite(scaled.get("mae"), f"{job_id} scaled mae"),
                "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed"),
                "epochs_ran": int(record.get("epochs_ran")),
                "best_epoch": int(record.get("best_epoch")),
                "job_id": job_id,
                "sidecar_sha256": hashes.get("acf_prior.npz"),
                "sidecar": sidecar_info,
                "config_signature": config_info["signature"],
                "effective_lags": config_info["effective_lags"],
                "relation_lags": config_info["relation_lags"],
                "state": state_info,
                "relation_diagnostics": relation_diagnostics,
                "runtime_diagnostics": runtime_diagnostics,
                "calibration_diagnostics": calibration_diagnostics,
                "autocon_loss": _summary_series(logs, "autocon_loss", logs_path),
            }
        )
    expected_coordinates = {
        (dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS
    }
    if seen != expected_coordinates:
        raise ValueError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows


def _paired_test(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    if scipy_stats is not None and len(numbers) > 1:
        result = scipy_stats.ttest_1samp(numbers, 0.0)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat = mean / (sd / math.sqrt(len(numbers)))
        p_value = None
    else:
        t_stat = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p_value = None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
    }


def _relative(delta: float, reference: float) -> float:
    return delta / reference if reference else math.nan


def _contrast(pairs: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    deltas = [float(item[f"delta_{metric}"]) for item in pairs]
    off_values = [float(item[f"off_{metric}"]) for item in pairs]
    on_values = [float(item[f"on_{metric}"]) for item in pairs]
    result = _paired_test(deltas)
    result.update(
        {
            "reference_mean": statistics.fmean(off_values),
            "treatment_mean": statistics.fmean(on_values),
            "relative_delta_mean": _relative(result["mean"], statistics.fmean(off_values)),
        }
    )
    return result


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {
        (str(row["arm"]), str(row["dataset"]), int(row["seed"])): row for row in rows
    }
    coordinates = sorted({(str(row["dataset"]), int(row["seed"])) for row in rows})
    expected_coordinates = {
        (dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS
    }
    if set(coordinates) != expected_coordinates:
        raise ValueError("paired readout coordinate matrix is incomplete")
    pairs: List[Dict[str, Any]] = []
    for dataset, seed in coordinates:
        off = indexed[("off", dataset, seed)]
        on = indexed[("on", dataset, seed)]
        if off["sidecar_sha256"] != on["sidecar_sha256"]:
            raise ValueError(f"off/on sidecars differ for {dataset} seed {seed}")
        if off["config_signature"] != on["config_signature"]:
            raise ValueError(f"off/on common configurations differ for {dataset} seed {seed}")
        if off["effective_lags"] != on["effective_lags"]:
            raise ValueError(f"off/on effective lag order differs for {dataset} seed {seed}")
        pairs.append(
            {
                "dataset": dataset,
                "seed": seed,
                "off_mse": off["mse"],
                "on_mse": on["mse"],
                "delta_mse": on["mse"] - off["mse"],
                "relative_delta_mse": _relative(on["mse"] - off["mse"], off["mse"]),
                "off_mae": off["mae"],
                "on_mae": on["mae"],
                "delta_mae": on["mae"] - off["mae"],
                "relative_delta_mae": _relative(on["mae"] - off["mae"], off["mae"]),
                "off_elapsed_sec": off["elapsed_sec"],
                "on_elapsed_sec": on["elapsed_sec"],
                "delta_elapsed_sec": on["elapsed_sec"] - off["elapsed_sec"],
                "sidecar_sha256_equal": True,
                "effective_lags": off["effective_lags"],
                "off_job_id": off["job_id"],
                "on_job_id": on["job_id"],
                "on_calibration_state": on["state"],
            }
        )

    by_dataset: Dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        subset = [pair for pair in pairs if pair["dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(subset),
            "mse": _contrast(subset, "mse"),
            "mae": _contrast(subset, "mae"),
            "elapsed_sec": _contrast(subset, "elapsed_sec"),
        }

    diagnostics: Dict[str, Any] = {}
    for arm in ("off", "on"):
        subset = [row for row in rows if row["arm"] == arm]
        diagnostics[arm] = {
            "autocon_loss_mean_over_runs": statistics.fmean(
                row["autocon_loss"]["mean"] for row in subset
            ),
            "relation": {
                key: statistics.fmean(
                    row["relation_diagnostics"][key]["mean"] for row in subset
                )
                for key in RELATION_LOG_KEYS
            },
            "runtime": {
                key: statistics.fmean(
                    row["runtime_diagnostics"][key]["mean"] for row in subset
                )
                for key in RUNTIME_LOG_KEYS
            },
            "calibration": {
                key: (
                    statistics.fmean(
                        row["calibration_diagnostics"][key]["mean"]
                        for row in subset
                        if row["calibration_diagnostics"][key]["mean"] is not None
                    )
                    if any(
                        row["calibration_diagnostics"][key]["mean"] is not None
                        for row in subset
                    )
                    else None
                )
                for key in CALIBRATION_LOG_KEYS
            },
        }

    return {
        "schema_version": 1,
        "status": "exploratory_horizon_amplitude_calibration_screen_not_confirmatory",
        "plan": str(plan.get("_path", "")),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "phases": [name for name, _, _ in EXPECTED_PHASES],
        "rows": list(rows),
        "pairs": pairs,
        "contrasts": {
            "on_minus_off": {
                "mse": _contrast(pairs, "mse"),
                "mae": _contrast(pairs, "mae"),
                "elapsed_sec": _contrast(pairs, "elapsed_sec"),
            },
            "by_dataset": by_dataset,
        },
        "diagnostics": diagnostics,
        "sidecar_pair_count": len(pairs),
        "evidence_boundary": "Paired exploratory mechanism evidence only; no confirmatory or publication claim.",
    }


def markdown(result: Mapping[str, Any]) -> str:
    contrast = result["contrasts"]["on_minus_off"]
    lines = [
        "# MambaACF Horizon Amplitude Calibration Screen",
        "",
        "Status: **exploratory; not confirmatory**.",
        "",
        "The off/on arms use the same MambaACFAutoCon model, hard global signed ACF relation objective, raw train-only sidecar, data, seeds, and eight-epoch budget. They differ only in the rank-2 horizon amplitude adapter.",
        "",
        "## Paired Readout",
        "",
        "| Contrast (on - off) | n | Mean delta | Relative delta | Wins lower | Two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in (
        ("scaled MSE", contrast["mse"]),
        ("scaled MAE", contrast["mae"]),
        ("elapsed seconds", contrast["elapsed_sec"]),
    ):
        p_value = values.get("two_sided_p")
        p_text = "n/a" if p_value is None else f"{p_value:.4g}"
        lines.append(
            f"| {label} | {values['n']} | {values['mean']:.6g} | {values['relative_delta_mean']:.3%} | {values['wins_lower']} | {p_text} |"
        )
    lines += [
        "",
        "Negative accuracy deltas favor calibration-on. The six paired coordinates are descriptive only.",
        "",
        "## Dataset Splits",
        "",
        "| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in result["contrasts"]["by_dataset"].items():
        mse, mae = values["mse"], values["mae"]
        lines.append(
            f"| {dataset} | {values['n']} | {mse['mean']:.6g} | {mse['relative_delta_mean']:.3%} | {mae['mean']:.6g} | {mae['relative_delta_mean']:.3%} |"
        )
    lines += [
        "",
        "## Adapter Diagnostics",
        "",
        "The checkpoint contract expects six trainable coefficients (`K=3`, rank `2`) on the treatment arm and no calibration tensors on the reference arm. Runtime amplitude logs are null for off and finite for on.",
        "",
        "| Arm | Effective memory RMS | Runtime gate mean | Amplitude mean | Amplitude min | Amplitude max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("off", "on"):
        runtime = result["diagnostics"][arm]["runtime"]
        calibration = result["diagnostics"][arm]["calibration"]
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.6g}"
        lines.append(
            f"| {arm} | {fmt(runtime['acf_effective_memory_rms'])} | {fmt(runtime['acf_runtime_gate_mean'])} | {fmt(calibration['acf_calibration_amp_mean'])} | {fmt(calibration['acf_calibration_amp_min'])} | {fmt(calibration['acf_calibration_amp_max'])} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- The adapter is a bounded functional correction to the existing memory path; its amplitude is partly confounded with the existing horizon-specific ACF gate.",
        "- The paired sample is dataset x seed (`n=6`); p-values are unadjusted descriptive summaries and cannot select a method.",
        "- This screen does not test longer horizons, dynamic ACF, alternative relation targets, or an independent replication.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/research/mamba_acf_horizon_calibration_screen_plan.json"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", "--output-markdown", dest="output_md", type=Path, default=None)
    args = parser.parse_args()
    plan_path = _resolve(args.plan)
    plan = dict(_read_json(plan_path))
    plan["_path"] = str(plan_path)
    rows: List[Dict[str, Any]] = []
    for phase_name, arm, enabled in EXPECTED_PHASES:
        rows.extend(_load_phase(plan, phase_name, arm, enabled))
    result = build_readout(plan, rows)
    output_json = args.output_json or Path(
        "artifacts/paper_final/mamba_acf_horizon_calibration_screen_20260901/readout.json"
    )
    output_md = args.output_md or Path(
        "docs/research/mamba_acf_horizon_calibration_screen_readout_20260901.md"
    )
    output_json = _resolve(output_json)
    output_md = _resolve(output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {"rows": len(rows), "pairs": len(result["pairs"]), "output_json": str(output_json), "output_md": str(output_md)},
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
