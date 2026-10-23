#!/usr/bin/env python
"""Validate that a registered phase is complete before it enters a report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_training_plan import (
    DATASETS, MODEL_SPECS, _command_hash, _feature_dimensions, _job_specs, _load_json,
    _canonical_hash, _selected_lambdas, _validate_confirmation_launch_binding,
    _validate_selection_binding,
)  # noqa: E402
from utils.acf_relation_contract import (  # noqa: E402
    resolve_acf_relation_batching,
    resolve_acf_source,
)
from utils.acf_guidance import compile_channel_lag_support  # noqa: E402
from utils.mamba_causality_contract import (  # noqa: E402
    strict_representation_causality_errors,
)


# Keep absent plan fields aligned with ``run_training_plan._command``. Mamba
# scale [1] is the executable default; [1, 2] is only used when a plan asks
# for the multiscale MambaACF path explicitly.
_DEFAULT_MAMBA_SCALES = [1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, str]:
    paths = [result_dir / "config.json", result_dir / "metrics.npy",
             result_dir / "original_metrics.npy", result_dir / "channel_metrics.npy",
             result_dir / "stream_metrics.json", result_dir / "stream_samples.npz",
             result_dir / "pred.npy", result_dir / "true.npy",
             result_dir / "training_complete.json", result_dir / "acf_prior.npz"]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _validate_stream_metrics(result_dir: Path, config: dict, job: dict) -> list[str]:
    """Validate compact streaming artifacts without loading full predictions."""
    errors = []
    stream_path = result_dir / "stream_metrics.json"
    sample_path = result_dir / "stream_samples.npz"
    if not stream_path.is_file():
        return ["stream_metrics_missing"]
    if not sample_path.is_file():
        errors.append("stream_samples_missing")
    try:
        with stream_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return errors + ["stream_metrics_invalid"]
    if not isinstance(payload, dict):
        return errors + ["stream_metrics_schema_invalid"]
    if payload.get("schema_version") != 1 or payload.get("aggregation") != "streaming_window_weighted":
        errors.append("stream_metrics_schema_invalid")
    expected_channels = int(job["config"]["c_out"])
    expected_horizon = int(job["pred_len"])
    configured_stride = job["config"].get("origin_stride")
    try:
        # The experiment code treats an omitted/zero stride as one horizon.
        expected_stride = int(configured_stride or expected_horizon)
    except (TypeError, ValueError):
        expected_stride = -1
    if expected_stride <= 0:
        errors.append("configured_origin_stride_invalid")
    if payload.get("output_channels") != expected_channels:
        errors.append("stream_output_channels_mismatch")
    if payload.get("pred_len") != expected_horizon:
        errors.append("stream_pred_len_mismatch")
    payload_stride = payload.get("origin_stride")
    if (isinstance(payload_stride, bool) or not isinstance(payload_stride, int)
            or payload_stride <= 0 or payload_stride != expected_stride):
        errors.append("stream_origin_stride_mismatch")
    windows = payload.get("window_count")
    elements = payload.get("element_count")
    origins = payload.get("origin_count")
    if not isinstance(windows, int) or windows <= 0:
        errors.append("stream_window_count_invalid")
    if (not isinstance(elements, int) or not isinstance(windows, int)
            or elements != windows * expected_horizon * expected_channels):
        errors.append("stream_element_count_invalid")
    if not isinstance(origins, int) or origins <= 0 or origins > windows:
        errors.append("stream_origin_count_invalid")
    elif isinstance(windows, int) and expected_stride > 0:
        expected_origins = (windows + expected_stride - 1) // expected_stride
        if origins != expected_origins:
            errors.append("stream_origin_count_mismatch")
    metric_names = ["mae", "mse", "rmse", "mape", "mspe"]
    for section in ("scaled", "original"):
        values = payload.get(section, {})
        for name in metric_names:
            value = values.get(name)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                errors.append(f"stream_{section}_{name}_invalid")
    origin_payload = payload.get("origin", {})
    if not isinstance(origin_payload, dict):
        errors.append("stream_origin_invalid")
        origin_payload = {}
    for name in metric_names:
        origin_scaled = origin_payload.get("scaled", {})
        origin_original = origin_payload.get("original", {})
        if not isinstance(origin_scaled, dict):
            origin_scaled = {}
        if not isinstance(origin_original, dict):
            origin_original = {}
        value = origin_scaled.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            errors.append(f"stream_origin_scaled_{name}_invalid")
        value = origin_original.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            errors.append(f"stream_origin_original_{name}_invalid")
    channels = np.asarray(payload.get("channel_metrics", []), dtype=float)
    if channels.shape != (expected_channels, 6) or not np.all(np.isfinite(channels)):
        errors.append("stream_channel_metrics_invalid")
    try:
        metrics = np.asarray(np.load(result_dir / "metrics.npy", allow_pickle=False), dtype=float).reshape(-1)
        original = np.asarray(np.load(result_dir / "original_metrics.npy", allow_pickle=False), dtype=float).reshape(-1)
    except (OSError, ValueError, EOFError):
        errors.append("stream_metric_arrays_missing_or_invalid")
    else:
        expected_scaled = np.asarray([payload["scaled"][name] for name in metric_names], dtype=float)
        expected_original = np.asarray([payload["original"][name] for name in metric_names], dtype=float)
        if metrics.size < 5 or not np.allclose(metrics[:5], expected_scaled, rtol=1e-6, atol=1e-8):
            errors.append("stream_metrics_array_mismatch")
        if original.size < 5 or not np.allclose(original[:5], expected_original, rtol=1e-6, atol=1e-8):
            errors.append("stream_original_metrics_array_mismatch")
    if sample_path.is_file():
        try:
            with np.load(sample_path, allow_pickle=False) as samples:
                pred, true = np.asarray(samples["pred"]), np.asarray(samples["true"])
            if (pred.shape != true.shape or pred.ndim != 3
                    or pred.shape[1:] != (expected_horizon, expected_channels)
                    or pred.shape[0] > int(config.get("stream_sample_windows", 4))
                    or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true))):
                errors.append("stream_samples_invalid")
        except (OSError, ValueError, EOFError, KeyError):
            errors.append("stream_samples_invalid")
    sample_meta = payload.get("sample_artifact", {})
    if sample_meta.get("sha256") and sample_path.is_file() and _sha256(sample_path) != sample_meta["sha256"]:
        errors.append("stream_sample_hash_mismatch")
    return errors


def _float_equal(left, right) -> bool:
    try:
        return bool(np.isclose(float(left), float(right)))
    except (TypeError, ValueError):
        return False


def _bool_value(value, default=False) -> bool:
    """Parse persisted CLI booleans without Python's ``bool('false')`` trap."""

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


def _acf_source_value(config: dict, default: str = "residual") -> str:
    """Resolve the persisted ACF-source spelling used by plans and results.

    Training plans use the lower-case ``autocon_acf_source`` key, while the
    runtime CLI serializes ``AutoCon_acf_source``.  The runner gives the plan
    spelling precedence when both are present, so validation follows that
    same order and accepts either spelling when only one is persisted.
    """

    return resolve_acf_source(config, default=default)


def _validate_mamba_acf_wnorm_contract(config: dict, model: str) -> list[str]:
    """Reject the unsupported decomposition normalization for MambaACF records."""

    if (
        "MambaACF" in str(model)
        and str(config.get("AutoCon_wnorm", "LastVal")).strip().lower() == "decomp"
    ):
        return ["config_AutoCon_wnorm_decomp_unsupported"]
    return []


def _int_list(value, default) -> list[int]:
    """Canonicalize scalar/list integer settings for strict config checks."""

    if value is None:
        value = default
    if isinstance(value, (str, bytes)):
        value = [item for item in str(value).replace(",", " ").split() if item]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return sorted({int(item) for item in value})
    except (TypeError, ValueError):
        return []


def _coordinate(item: dict) -> tuple:
    config = item.get("config", {})
    features = item.get("features", config.get("features", "S"))
    enc_in = item.get("enc_in", config.get("enc_in", 1))
    dec_in = item.get("dec_in", config.get("dec_in", 1))
    c_out = item.get("c_out", config.get("c_out", 1))
    return (item.get("phase"), item.get("dataset"), item.get("model"),
            features, int(enc_in), int(dec_in), int(c_out),
            int(item.get("pred_len")), int(item.get("seed")),
            None if item.get("autocon_lambda") is None else float(item["autocon_lambda"]))


def _read_manifest(path: Path) -> tuple[dict[str, dict], int]:
    records = {}
    malformed = 0
    if not path.is_file():
        return records, malformed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if "job_id" in record:
                records[record["job_id"]] = record
    return records, malformed


def _arg(command: list[str], name: str) -> str | None:
    if name not in command:
        return None
    index = command.index(name)
    return command[index + 1] if index + 1 < len(command) else None


def _strict_positive_lags(value, label: str) -> tuple[list[int] | None, list[str]]:
    """Normalize a persisted lag list and report machine-checkable errors."""

    errors: list[str] = []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None, [f"{label}_missing_or_not_a_list"]
    if not value:
        return None, [f"{label}_empty"]
    normalized: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            errors.append(f"{label}_contains_nonpositive_or_nonfinite")
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            errors.append(f"{label}_contains_nonpositive_or_nonfinite")
            continue
        if not math.isfinite(number) or number <= 0.0 or not number.is_integer():
            errors.append(f"{label}_contains_nonpositive_or_nonfinite")
            continue
        normalized.append(int(number))
    if errors:
        return None, sorted(set(errors))
    if len(set(normalized)) != len(normalized):
        errors.append(f"{label}_contains_duplicates")
    return sorted(normalized), errors


def _strict_positive_int(value, label: str) -> tuple[int | None, list[str]]:
    """Normalize a persisted positive integer without truncating input."""

    if isinstance(value, (bool, np.bool_)):
        return None, [f"{label}_invalid"]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None, [f"{label}_invalid"]
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, [f"{label}_invalid"]
    if not isinstance(value, (str, bytes)) and parsed != value:
        return None, [f"{label}_invalid"]
    if parsed <= 0:
        return None, [f"{label}_invalid"]
    return parsed, []


def _strict_nonnegative_int(value, label: str) -> tuple[int | None, list[str]]:
    """Validate a persisted non-negative integer without truncation."""

    if isinstance(value, (bool, np.bool_)):
        return None, [f"{label}_invalid"]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None, [f"{label}_invalid"]
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, [f"{label}_invalid"]
    if not isinstance(value, (str, bytes)) and parsed != value:
        return None, [f"{label}_invalid"]
    if parsed < 0:
        return None, [f"{label}_invalid"]
    return parsed, []


def _strict_positive_int_list(
    value, label: str, *, default=None, allow_empty: bool = False
) -> tuple[list[int] | None, list[str]]:
    """Validate a scalar/list integer setting while preserving duplicates."""

    if value is None:
        value = default
    if isinstance(value, (str, bytes)):
        value = [item for item in str(value).replace(",", " ").split() if item]
    elif not isinstance(value, (list, tuple)):
        return None, [f"{label}_invalid"]
    if not value and not allow_empty:
        return None, [f"{label}_empty"]
    normalized: list[int] = []
    errors: list[str] = []
    for item in value:
        parsed, item_errors = _strict_positive_int(item, label)
        if item_errors:
            errors.extend(item_errors)
        else:
            normalized.append(parsed)
    if errors:
        return None, sorted(set(errors))
    return normalized, []


def _strict_float_value(
    value,
    label: str,
    *,
    default=None,
    minimum=None,
    minimum_inclusive: bool = True,
    maximum=None,
) -> tuple[float | None, list[str]]:
    """Validate a persisted finite float and optional inclusive bounds."""

    if value is None:
        value = default
    if isinstance(value, (bool, np.bool_)):
        return None, [f"{label}_invalid"]
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, [f"{label}_invalid"]
    if not math.isfinite(parsed):
        return None, [f"{label}_invalid"]
    if minimum is not None:
        if minimum_inclusive and parsed < minimum:
            return None, [f"{label}_invalid"]
        if not minimum_inclusive and parsed <= minimum:
            return None, [f"{label}_invalid"]
    if maximum is not None and parsed > maximum:
        return None, [f"{label}_invalid"]
    return parsed, []


def _validate_mamba_numeric_contract(config: dict, model: str) -> list[str]:
    """Reject malformed Mamba numeric settings during phase replay."""

    if not str(model).startswith("Mamba"):
        return []
    errors: list[str] = []

    def positive(key: str, default):
        _, item_errors = _strict_positive_int(
            config.get(key, default), f"config_{key}"
        )
        errors.extend(item_errors)

    def nonnegative(key: str, default):
        _, item_errors = _strict_nonnegative_int(
            config.get(key, default), f"config_{key}"
        )
        errors.extend(item_errors)

    def finite_float(key: str, default, **kwargs):
        _, item_errors = _strict_float_value(
            config.get(key, default), f"config_{key}", **kwargs
        )
        errors.extend(item_errors)

    positive("mamba_token_stride", 1)
    scales, scale_errors = _strict_positive_int_list(
        config.get("mamba_scales"), "config_mamba_scales",
        default=_DEFAULT_MAMBA_SCALES
    )
    errors.extend(scale_errors)
    if "MambaACF" not in str(model):
        return sorted(set(errors))

    lags, lag_errors = _strict_positive_int_list(
        config.get("mamba_acf_lags"), "config_mamba_acf_lags",
        default=[24, 168, 336], allow_empty=True,
    )
    errors.extend(lag_errors)
    positive("mamba_acf_max_lag", 720)
    positive("mamba_acf_top_k", 8)
    positive("mamba_acf_min_lag", 2)
    positive("mamba_acf_min_separation", 2)
    positive("mamba_acf_chunk", 64)
    positive("mamba_acf_horizon_calibration_rank", 2)
    positive("mamba_temporal_rank", 16)
    positive("mamba_acf_channel_lag_top_k", 2)
    positive("mamba_data_dependent_channel_mixer_rank", 8)
    nonnegative("mamba_acf_relation_lag_tolerance", 0)
    nonnegative("mamba_acf_relation_min_lag", 0)
    finite_float("mamba_acf_min_abs", 0.1, minimum=0.0, maximum=1.0)
    finite_float("mamba_acf_confidence_z", 1.96, minimum=0.0)
    finite_float(
        "mamba_acf_temperature", 4.0, minimum=0.0,
        minimum_inclusive=False,
    )
    finite_float("mamba_acf_relation_soft_min_abs", 0.1, minimum=0.0, maximum=1.0)
    finite_float(
        "mamba_acf_relation_diagnostic_min_abs", 0.1,
        minimum=0.0, maximum=1.0,
    )
    finite_float(
        "mamba_acf_relation_selected_lag_boost", 0.0, minimum=0.0
    )
    finite_float("AutoCon_temperature", 0.2, minimum=0.0, minimum_inclusive=False)
    finite_float(
        "AutoCon_relation_temperature", 0.25,
        minimum=0.0, minimum_inclusive=False,
    )
    finite_float("AutoCon_sign_weight", 0.25, minimum=0.0)
    finite_float("AutoCon_sign_margin", 0.1, minimum=0.0)
    # The values above are validated independently of the relation switch;
    # they are part of the persisted MambaACF contract and must not become
    # valid merely because a malformed record disables the objective.
    return sorted(set(errors))


def _strict_scalar_positive_int(value, label: str) -> tuple[int | None, list[str]]:
    """Validate a scalar integer field without ``int(4.9)`` truncation."""

    raw = np.asarray(value)
    if raw.ndim != 0:
        return None, [f"{label}_invalid"]
    return _strict_positive_int(raw.item(), label)


_RELATION_BATCHING_CONFIG_FIELDS = (
    "mamba_acf_relation_batching_requested",
    "mamba_acf_relation_batching_effective",
    "mamba_acf_relation_batching_reason",
    "mamba_acf_relation_selected_lag_support",
)
_RELATION_BATCHING_LOG_FIELDS = (
    "acf_relation_batching_requested",
    "acf_relation_batching_effective",
    "acf_relation_batching_reason",
    "acf_relation_selected_lag_support",
)
_RELATION_PAIR_TELEMETRY_FIELDS = (
    "acf_relation_lag_pair_counts",
    "acf_relation_lag_pair_exposure",
)


def _sampler_text(value) -> str | None:
    """Return a persisted sampler enum or ``None`` for a blank value."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _sampler_bool(value) -> bool | None:
    """Parse a persisted boolean without accepting arbitrary truthy strings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _has_sampler_metadata(payload: dict, fields: tuple[str, ...]) -> bool:
    """Treat a key as metadata even when a writer serialized it as null."""

    return any(field in payload for field in fields)


def _zero_only_pair_telemetry(value, label: str) -> list[str]:
    """Require inactive sampler telemetry to be empty or numerically zero."""

    errors: list[str] = []

    def visit(item) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if isinstance(item, bool):
            if item:
                errors.append(f"{label}_nonzero_for_shuffle")
            return
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            errors.append(f"{label}_invalid_for_shuffle")
            return
        if not math.isfinite(numeric):
            errors.append(f"{label}_invalid_for_shuffle")
        elif numeric != 0.0:
            errors.append(f"{label}_nonzero_for_shuffle")

    visit(value)
    return sorted(set(errors))


def _validate_active_lag_pair_telemetry(
    logs: dict,
    effective_lags: list[int] | None,
    epoch_count: int | None,
) -> list[str]:
    """Validate that an enabled lag-pair sampler actually supplied pairs."""

    errors: list[str] = []
    count_epochs = logs.get("acf_relation_lag_pair_counts")
    exposure_epochs = logs.get("acf_relation_lag_pair_exposure")
    if not isinstance(count_epochs, list) or not count_epochs:
        return ["acf_relation_lag_pair_counts_missing_or_empty_for_lag_pairs"]
    if not isinstance(exposure_epochs, list) or len(exposure_epochs) != len(count_epochs):
        return ["acf_relation_lag_pair_exposure_missing_or_misaligned_for_lag_pairs"]
    if epoch_count is not None and len(count_epochs) != epoch_count:
        errors.append("acf_relation_lag_pair_counts_length_mismatch")

    expected_keys = (
        {str(int(lag)) for lag in effective_lags}
        if effective_lags is not None else None
    )
    for epoch, (raw_counts, raw_exposure) in enumerate(
        zip(count_epochs, exposure_epochs), start=1
    ):
        if not isinstance(raw_counts, dict) or not raw_counts:
            errors.append("acf_relation_lag_pair_counts_invalid_for_lag_pairs")
            continue
        if not isinstance(raw_exposure, dict):
            errors.append("acf_relation_lag_pair_exposure_invalid_for_lag_pairs")
            continue
        normalized_raw_counts = {str(key): value for key, value in raw_counts.items()}
        normalized_raw_exposure = {
            str(key): value for key, value in raw_exposure.items()
        }
        count_keys = set(normalized_raw_counts)
        exposure_keys = set(normalized_raw_exposure)
        if count_keys != exposure_keys:
            errors.append("acf_relation_lag_pair_exposure_keys_mismatch")
            continue
        if expected_keys is not None and count_keys != expected_keys:
            errors.append("acf_relation_lag_pair_counts_lags_mismatch")

        counts: dict[str, int] = {}
        for lag in count_keys:
            try:
                numeric = float(normalized_raw_counts[lag])
            except (TypeError, ValueError):
                errors.append("acf_relation_lag_pair_counts_invalid_for_lag_pairs")
                continue
            if (
                not math.isfinite(numeric)
                or numeric < 0
                or not numeric.is_integer()
            ):
                errors.append("acf_relation_lag_pair_counts_invalid_for_lag_pairs")
                continue
            counts[lag] = int(numeric)
        if len(counts) != len(count_keys):
            continue
        total = sum(counts.values())
        if total <= 0:
            errors.append("acf_relation_lag_pair_counts_zero_for_lag_pairs")
            continue
        for lag, count in counts.items():
            try:
                exposure = float(normalized_raw_exposure[lag])
            except (TypeError, ValueError):
                errors.append("acf_relation_lag_pair_exposure_invalid_for_lag_pairs")
                continue
            expected_exposure = count / total
            if (
                not math.isfinite(exposure)
                or exposure < 0.0
                or exposure > 1.0
                or not math.isclose(
                    exposure, expected_exposure, rel_tol=1e-7, abs_tol=1e-9
                )
            ):
                errors.append("acf_relation_lag_pair_exposure_invalid_for_lag_pairs")
    return sorted(set(errors))


def _validate_relation_batching_contract(
    config: dict,
    logs: dict,
    *,
    strict: bool,
    effective_lags: list[int] | None = None,
) -> list[str]:
    """Bind persisted sampler metadata to the live relation-loss semantics.

    The resolver is data-independent, so reusing it here proves that a stored
    ``lag_pairs`` claim was meaningful for the target contract at run time.
    Historical records that predate all sampler metadata remain readable when
    ``strict`` is false; a strict/provenance-bearing record must attest every
    field and its realised pair telemetry.
    """

    errors: list[str] = []
    relation_loss = _bool_value(config.get("mamba_acf_relation_loss", False))
    config_metadata = _has_sampler_metadata(config, _RELATION_BATCHING_CONFIG_FIELDS)
    log_metadata = _has_sampler_metadata(logs, _RELATION_BATCHING_LOG_FIELDS)
    telemetry_metadata = _has_sampler_metadata(logs, _RELATION_PAIR_TELEMETRY_FIELDS)

    if not relation_loss:
        # ``persist_run_config`` deliberately does not add the resolved
        # fields for an inactive objective.  Accept either blank config
        # values or a complete neutral decision for forward compatibility.
        expected = resolve_acf_relation_batching(SimpleNamespace(**config))
        config_values = [config.get(field) for field in _RELATION_BATCHING_CONFIG_FIELDS]
        config_blank = not config_metadata or all(
            value is None or (isinstance(value, str) and not value.strip())
            for value in config_values
        )
        config_neutral = (
            _sampler_text(config.get("mamba_acf_relation_batching_requested"))
            == expected["requested"]
            and _sampler_text(config.get("mamba_acf_relation_batching_effective"))
            == "shuffle"
            and _sampler_text(config.get("mamba_acf_relation_batching_reason"))
            == expected["reason"]
            and _sampler_bool(config.get("mamba_acf_relation_selected_lag_support"))
            is expected["selected_lag_support"]
        )
        if not config_blank and not config_neutral:
            errors.append("config_acf_relation_batching_present_while_loss_disabled")

        # Non-relation AutoCon logs may still contain their constructor
        # defaults.  They are acceptable only when they describe shuffle/no
        # selected-lag support and never claim accepted lag pairs.
        if log_metadata:
            requested = _sampler_text(logs.get("acf_relation_batching_requested"))
            effective = _sampler_text(logs.get("acf_relation_batching_effective"))
            support = _sampler_bool(logs.get("acf_relation_selected_lag_support"))
            reason = _sampler_text(logs.get("acf_relation_batching_reason"))
            if requested not in {None, expected["requested"]}:
                errors.append("acf_relation_batching_requested_mismatch_while_loss_disabled")
            if effective not in {None, "shuffle"}:
                errors.append("acf_relation_batching_effective_active_while_loss_disabled")
            if support not in {None, False, expected["selected_lag_support"]}:
                errors.append("acf_relation_selected_lag_support_active_while_loss_disabled")
            if reason not in {None, "not_resolved", "relation_loss_disabled"}:
                errors.append("acf_relation_batching_reason_active_while_loss_disabled")
        for field in _RELATION_PAIR_TELEMETRY_FIELDS:
            errors.extend(_zero_only_pair_telemetry(logs.get(field), field))
        return sorted(set(errors))

    enforce = strict or config_metadata or log_metadata or telemetry_metadata
    if not enforce:
        return []

    expected = resolve_acf_relation_batching(SimpleNamespace(**config))
    config_expected = {
        "mamba_acf_relation_batching_requested": expected["requested"],
        "mamba_acf_relation_batching_effective": expected["effective"],
        "mamba_acf_relation_batching_reason": expected["reason"],
        "mamba_acf_relation_selected_lag_support": expected["selected_lag_support"],
    }
    log_expected = {
        "acf_relation_batching_requested": expected["requested"],
        "acf_relation_batching_effective": expected["effective"],
        "acf_relation_batching_reason": expected["reason"],
        "acf_relation_selected_lag_support": expected["selected_lag_support"],
    }
    for field, expected_value in config_expected.items():
        if field not in config:
            errors.append(f"{field}_missing")
            continue
        actual = (
            _sampler_bool(config.get(field))
            if isinstance(expected_value, bool) else _sampler_text(config.get(field))
        )
        if actual is None:
            errors.append(f"{field}_invalid")
        elif actual != expected_value:
            errors.append(f"{field}_mismatch")
    for field, expected_value in log_expected.items():
        if field not in logs:
            errors.append(f"{field}_missing")
            continue
        actual = (
            _sampler_bool(logs.get(field))
            if isinstance(expected_value, bool) else _sampler_text(logs.get(field))
        )
        if actual is None:
            errors.append(f"{field}_invalid")
        elif actual != expected_value:
            errors.append(f"{field}_mismatch")
        config_field = "mamba_" + field
        if config_field in config:
            config_value = (
                _sampler_bool(config.get(config_field))
                if isinstance(expected_value, bool) else _sampler_text(config.get(config_field))
            )
            if actual is not None and config_value is not None and actual != config_value:
                errors.append(f"{field}_config_log_mismatch")

    epoch_count = len(logs.get("val_loss")) if isinstance(logs.get("val_loss"), list) else None
    if expected["effective"] == "shuffle":
        for field in _RELATION_PAIR_TELEMETRY_FIELDS:
            if field not in logs:
                errors.append(f"{field}_missing_for_shuffle")
            errors.extend(_zero_only_pair_telemetry(logs.get(field), field))
    else:
        if not expected["selected_lag_support"]:
            errors.append("acf_relation_lag_pairs_without_selected_lag_support")
        if effective_lags is None:
            errors.append("config_mamba_acf_relation_effective_lags_missing_for_lag_pairs")
        errors.extend(
            _validate_active_lag_pair_telemetry(logs, effective_lags, epoch_count)
        )
    return sorted(set(errors))


def _validate_relation_contract(
    config: dict, logs: dict, *, strict: bool = True
) -> list[str]:
    """Validate train-peak relation metadata and epoch-level coverage.

    ``strict`` is used only for legacy ad-hoc records that predate the
    provenance-bearing confirmation runner.  Any real phase record is strict;
    missing metadata therefore cannot be promoted into a confirmation result.
    """

    errors: list[str] = []
    relation_loss = _bool_value(config.get("mamba_acf_relation_loss", False))
    if not relation_loss:
        return _validate_relation_batching_contract(config, logs, strict=strict)

    lag_keys_present = any(
        key in config
        for key in (
            "mamba_acf_effective_lags",
            "mamba_acf_relation_effective_lags",
            "mamba_acf_train_peak_lags",
            "mamba_acf_relation_lag_source",
            "mamba_acf_relation_lag_fallback",
        )
    )
    enforce_metadata = bool(strict or lag_keys_present)
    effective, lag_errors = _strict_positive_lags(
        config.get("mamba_acf_relation_effective_lags"),
        "config_mamba_acf_relation_effective_lags",
    )
    errors.extend(lag_errors if enforce_metadata else [])
    selected, selected_errors = _strict_positive_lags(
        config.get("mamba_acf_relation_selected_lags"),
        "config_mamba_acf_relation_selected_lags",
    )
    errors.extend(selected_errors if enforce_metadata else [])
    train_peaks, peak_errors = _strict_positive_lags(
        config.get("mamba_acf_train_peak_lags"),
        "config_mamba_acf_train_peak_lags",
    )
    errors.extend(peak_errors if enforce_metadata else [])
    errors.extend(
        _validate_relation_batching_contract(
            config,
            logs,
            strict=strict,
            effective_lags=effective,
        )
    )

    try:
        max_lag = float(config.get("mamba_acf_prior_max_lag", config.get("mamba_acf_max_lag", 720)))
    except (TypeError, ValueError):
        max_lag = float("nan")
    try:
        min_lag = float(config.get("mamba_acf_relation_min_lag", 0))
    except (TypeError, ValueError):
        min_lag = float("nan")
    if enforce_metadata:
        if not math.isfinite(max_lag) or max_lag <= 0:
            errors.append("config_mamba_acf_prior_max_lag_invalid")
        if not math.isfinite(min_lag) or min_lag < 0:
            errors.append("config_mamba_acf_relation_min_lag_invalid")
        if effective is not None:
            if any(float(lag) > max_lag for lag in effective):
                errors.append("config_mamba_acf_relation_effective_lags_out_of_range")
            if any(float(lag) < min_lag for lag in effective):
                errors.append("config_mamba_acf_relation_effective_lags_below_min_lag")
        if selected is not None and effective is not None:
            if not set(effective).issubset(set(selected)):
                errors.append("config_mamba_acf_relation_effective_lags_not_selected")
        if train_peaks is not None and effective is not None:
            if not set(effective).issubset(set(train_peaks)):
                errors.append("config_mamba_acf_relation_effective_lags_not_train_peaks")
        lag_mode = str(config.get("mamba_acf_lag_mode", "")).lower()
        if lag_mode not in {"train_peaks", "train_peaks_available"}:
            errors.append("config_mamba_acf_relation_lag_mode_not_train_peak")
        source = str(config.get("mamba_acf_relation_lag_source", "")).strip().lower()
        if source != "train_peak_subset":
            errors.append("config_mamba_acf_relation_lag_source_invalid")
        if _bool_value(config.get("mamba_acf_relation_lag_fallback", False)):
            errors.append("config_mamba_acf_relation_lag_fallback_forbidden")

    # Relation diagnostics are epoch trajectories, not a single scalar.  They
    # must be non-empty, finite, and aligned with validation epochs.  Global
    # coverage is required to be strictly positive for a usable relation loss.
    val_loss = logs.get("val_loss")
    epoch_count = len(val_loss) if isinstance(val_loss, list) else None
    scope = str(config.get("mamba_acf_relation_scope", "both")).lower()
    if scope not in {"global", "local", "both"}:
        errors.append("acf_relation_scope_invalid")
        scope = "both"
    required: list[tuple[str, str, bool]] = []
    if scope in {"global", "both"}:
        required.extend([
            ("acf_relation_global_valid_fraction", "fraction", True),
            ("acf_relation_global_active_anchor_fraction", "fraction", True),
            ("acf_relation_global_candidate_fraction", "fraction", True),
            ("acf_relation_global_candidate_active_anchor_fraction", "fraction", True),
            ("acf_relation_global_positive_count_mean", "positive", True),
            ("acf_relation_global_high_acf_excluded_fraction", "fraction", False),
        ])
    if scope in {"local", "both"}:
        required.extend([
            ("acf_relation_local_target_fraction", "fraction", True),
            ("acf_relation_local_active_position_fraction", "fraction", True),
        ])
    if enforce_metadata:
        log_effective, log_effective_errors = _strict_positive_lags(
            logs.get("acf_relation_effective_lags"),
            "acf_relation_effective_lags",
        )
        log_selected, log_selected_errors = _strict_positive_lags(
            logs.get("acf_relation_selected_lags"),
            "acf_relation_selected_lags",
        )
        log_peaks, log_peak_errors = _strict_positive_lags(
            logs.get("acf_relation_train_peak_lags"),
            "acf_relation_train_peak_lags",
        )
        errors.extend(log_effective_errors)
        errors.extend(log_selected_errors)
        errors.extend(log_peak_errors)
        if effective is not None and log_effective is not None and effective != log_effective:
            errors.append("acf_relation_effective_lags_config_log_mismatch")
        if selected is not None and log_selected is not None and selected != log_selected:
            errors.append("acf_relation_selected_lags_config_log_mismatch")
        if train_peaks is not None and log_peaks is not None and train_peaks != log_peaks:
            errors.append("acf_relation_train_peak_lags_config_log_mismatch")
        if str(logs.get("acf_relation_lag_source", "")).strip().lower() != "train_peak_subset":
            errors.append("acf_relation_lag_source_missing_or_invalid")
        if _bool_value(logs.get("acf_relation_lag_fallback", False)):
            errors.append("acf_relation_lag_fallback_forbidden")
        for key, kind, positive in required:
            values = logs.get(key)
            if not isinstance(values, list) or not values:
                errors.append(f"{key}_missing_or_empty")
                continue
            if epoch_count is not None and len(values) != epoch_count:
                errors.append(f"{key}_coverage_length_mismatch")
            numbers: list[float] = []
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    numbers.append(float("nan"))
                    continue
                numbers.append(number)
            if not numbers or not np.all(np.isfinite(numbers)):
                errors.append(f"{key}_nonfinite")
                continue
            if kind == "fraction" and (np.any(np.asarray(numbers) < 0.0) or np.any(np.asarray(numbers) > 1.0)):
                errors.append(f"{key}_outside_unit_interval")
            if positive and np.any(np.asarray(numbers) <= 0.0):
                errors.append(f"{key}_not_strictly_positive")
    return sorted(set(errors))


def _validate_memory_lag_contract(
    config: dict,
    job: dict,
    plan: dict | None = None,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate resolved ACF-memory lags for strict confirmation records.

    Relation-loss metadata is not present in the ACF-only ablation arms, so
    checking it only inside ``_validate_relation_contract`` leaves the peak
    memory arm open to a configured-lag fallback.  Confirmation records must
    expose the train-only peak provenance independently of whether relation
    supervision is enabled.  Legacy exploratory records retain their older
    validation behavior by default.
    """

    if not strict or "MambaACF" not in str(job.get("model", "")):
        return []

    # The confirmation claim is about a causal encoder plus causal ACF
    # memory.  Circular value embedding or a reverse Mamba scan would widen
    # the information set and invalidate that interpretation.
    errors: list[str] = []
    if not _bool_value(config.get("mamba_causal_embedding", False)):
        errors.append("config_mamba_causal_embedding_required")
    if _bool_value(config.get("mamba_bidirectional", False)):
        errors.append("config_mamba_bidirectional_forbidden")
    # The current fitter estimates residual/trend priors on the complete
    # training series, while inference decomposes each encoder window
    # independently.  Only the raw source is numerically identical at both
    # points; a strict confirmation run must therefore use it until a
    # window-consistent decomposition is implemented and separately reviewed.
    if _acf_source_value(config) != "raw":
        errors.append("config_acf_source_raw_required_for_strict_causality")
    if str(config.get("mamba_acf_detrend", "mean")).strip().lower() == "linear":
        errors.append("config_mamba_acf_linear_detrend_unsupported")
    anchor_copy = _bool_value(config.get("mamba_acf_anchor_copy", False))
    if not _bool_value(config.get("mamba_use_acf_memory", False)) and not anchor_copy:
        return sorted(set(errors))
    effective, effective_errors = _strict_positive_lags(
        config.get("mamba_acf_effective_lags"),
        "config_mamba_acf_effective_lags",
    )
    errors.extend(effective_errors)
    if _bool_value(config.get("mamba_acf_effective_lag_fallback", False)):
        errors.append("config_mamba_acf_effective_lag_fallback_forbidden")

    try:
        max_lag = float(config.get("mamba_acf_prior_max_lag", config.get("mamba_acf_max_lag", 720)))
    except (TypeError, ValueError):
        max_lag = float("nan")
    if not math.isfinite(max_lag) or max_lag <= 0:
        errors.append("config_mamba_acf_prior_max_lag_invalid")
    if effective is not None and max_lag > 0 and any(float(lag) > max_lag for lag in effective):
        errors.append("config_mamba_acf_effective_lags_out_of_range")

    falsification = (plan or {}).get("acf_falsification_contract", {})
    placebo_phase = (
        str(falsification.get("fixed_placebo_arm", ""))
        if isinstance(falsification, dict) else ""
    )
    phase_name = str(job.get("phase", ""))
    if phase_name == placebo_phase:
        if str(config.get("mamba_acf_lag_mode", "")).strip().lower() != "fixed":
            errors.append("config_mamba_acf_placebo_lag_mode_not_fixed")
        if str(config.get("mamba_acf_effective_lag_source", "")).strip().lower() != "fixed_configured":
            errors.append("config_mamba_acf_placebo_lag_source_invalid")
        registered = (
            falsification.get("placebo_lags")
            if isinstance(falsification, dict) else None
        )
        placebo_lags, placebo_errors = _strict_positive_lags(
            registered, "registered_placebo_lags"
        )
        errors.extend(placebo_errors)
        if effective is not None and placebo_lags is not None and set(effective) != set(placebo_lags):
            errors.append("config_mamba_acf_placebo_effective_lags_mismatch")
    else:
        train_peaks, peak_errors = _strict_positive_lags(
            config.get("mamba_acf_train_peak_lags"),
            "config_mamba_acf_train_peak_lags",
        )
        errors.extend(peak_errors)
        if train_peaks is None:
            # _strict_positive_lags already emitted a useful shape/value
            # error; keep the subset check conditional to avoid a secondary
            # misleading diagnostic.
            return sorted(set(errors))
        if effective is not None and not set(effective).issubset(set(train_peaks)):
            errors.append("config_mamba_acf_effective_lags_not_train_peaks")
        lag_mode = str(config.get("mamba_acf_lag_mode", "")).strip().lower()
        if lag_mode not in {"train_peaks", "train_peaks_available"}:
            errors.append("config_mamba_acf_memory_lag_mode_not_train_peak")
        source = str(config.get("mamba_acf_effective_lag_source", "")).strip().lower()
        if source not in {"train_peaks", "train_peaks_available"}:
            errors.append("config_mamba_acf_memory_lag_source_invalid")
    return sorted(set(errors))


def _validate_strict_representation_causality_contract(
    config: dict,
    model: str,
) -> list[str]:
    """Validate an opt-in full representation-grid causality claim.

    This is deliberately independent of the confirmation memory contract.
    Historical artifacts can be replayed with their original settings, while a
    record that explicitly requests a strict representation claim must expose
    the complete conservative configuration at result-validation time.
    """

    if not _bool_value(
        config.get("mamba_require_strict_representation_causality", False)
    ):
        return []
    if "MambaACF" not in str(model):
        return ["config_strict_representation_causality_model_unsupported"]
    return [
        "config_strict_representation_causality_" + reason
        for reason in strict_representation_causality_errors(config, model=model)
    ]


def _validate_acf_prior_artifact(
    config: dict,
    result_dir: Path,
    job: dict,
    *,
    strict: bool = False,
) -> list[str]:
    """Require and verify train-only ACF sidecars for strict ACF jobs."""

    anchor_required = _bool_value(config.get("mamba_acf_anchor_copy", False))
    # Anchor retrieval is itself a train-prior-dependent forecast branch.  It
    # therefore opts into sidecar validation even when callers do not set the
    # broader strict-memory switch (for example, an exploratory record with
    # no provenance id yet).
    if (not strict and not anchor_required) or "MambaACF" not in str(job.get("model", "")):
        return []
    errors: list[str] = []
    memory_required = (
        _bool_value(config.get("mamba_use_acf_memory", False))
        and not _bool_value(config.get("mamba_dynamic_acf", False))
    )
    # Anchor-copy is an independent forecast branch, but it still consumes
    # the same static, train-only ACF prior.  Include it in the complete-table
    # contract even when the soft memory residual and relation objective are
    # disabled.
    memory_required = memory_required or anchor_required
    relation_required = _bool_value(config.get("mamba_acf_relation_loss", False))
    # A strict/provenance-bearing MambaACF record must carry the sidecar even
    # when the forecast and relation branches are disabled.  The runner fits
    # the train-only prior before every MambaACF training run, and retaining
    # that artifact is what makes an ablation auditable (same prior, same
    # effective lags, and no silent refit).  Non-strict legacy records keep the
    # historical optional behavior through the ``strict`` guard above.
    artifact_name = str(config.get("mamba_acf_prior_artifact", "")).strip()
    if artifact_name != "acf_prior.npz" or Path(artifact_name).name != artifact_name:
        return ["config_mamba_acf_prior_artifact_invalid"]
    artifact_path = result_dir / artifact_name
    if not artifact_path.is_file():
        return ["mamba_acf_prior_artifact_missing"]

    expected_sha = str(config.get("mamba_acf_prior_sha256", "")).strip().lower()
    if len(expected_sha) != 64:
        errors.append("config_mamba_acf_prior_sha256_invalid")
    else:
        actual_sha = _sha256(artifact_path)
        if actual_sha != expected_sha:
            errors.append("mamba_acf_prior_artifact_sha256_mismatch")

    try:
        with np.load(str(artifact_path), allow_pickle=False) as payload:
            required = {
                "schema_version", "acf_values", "lags", "seq_len", "pred_len",
                "channels", "acf_source", "acf_temperature",
            }
            if not required.issubset(set(payload.files)):
                errors.append("mamba_acf_prior_artifact_schema_missing")
                return sorted(set(errors))
            schema_version, schema_errors = _strict_scalar_positive_int(
                payload["schema_version"], "mamba_acf_prior_artifact_schema"
            )
            if schema_errors:
                errors.extend(schema_errors)
            elif schema_version != 1:
                errors.append("mamba_acf_prior_artifact_schema_invalid")
            values = np.asarray(payload["acf_values"], dtype=np.float32)
            full_values = (
                np.asarray(payload["acf_full_values"], dtype=np.float32)
                if "acf_full_values" in payload.files else None
            )
            raw_lags = np.asarray(payload["lags"])
            lag_dtype_valid = raw_lags.dtype.kind in {"i", "u"}
            if raw_lags.dtype.kind == "u" and np.any(
                raw_lags > np.iinfo(np.int64).max
            ):
                lag_dtype_valid = False
            if not lag_dtype_valid:
                errors.append("mamba_acf_prior_artifact_lags_dtype_invalid")
                lags = np.asarray([], dtype=np.int64)
            else:
                lags = raw_lags.astype(np.int64, copy=False).reshape(-1)
            if (
                lags.size == 0
                or np.any(lags <= 0)
                or len(set(int(item) for item in lags.tolist())) != int(lags.size)
            ):
                errors.append("mamba_acf_prior_artifact_lags_invalid")
            artifact_seq_len, seq_errors = _strict_scalar_positive_int(
                payload["seq_len"], "mamba_acf_prior_artifact_seq_len"
            )
            artifact_pred_len, pred_errors = _strict_scalar_positive_int(
                payload["pred_len"], "mamba_acf_prior_artifact_pred_len"
            )
            artifact_channels, channel_errors = _strict_scalar_positive_int(
                payload["channels"], "mamba_acf_prior_artifact_channels"
            )
            errors.extend(seq_errors)
            errors.extend(pred_errors)
            errors.extend(channel_errors)
            if artifact_seq_len is not None and artifact_seq_len != int(config.get("seq_len", -1)):
                errors.append("mamba_acf_prior_artifact_seq_len_mismatch")
            if artifact_pred_len is not None and artifact_pred_len != int(config.get("pred_len", -1)):
                errors.append("mamba_acf_prior_artifact_pred_len_mismatch")
            if artifact_channels is not None and artifact_channels != int(config.get("enc_in", -1)):
                errors.append("mamba_acf_prior_artifact_channels_mismatch")
            if str(np.asarray(payload["acf_source"]).item()).strip().lower() != _acf_source_value(config):
                errors.append("mamba_acf_prior_artifact_source_mismatch")
            try:
                artifact_temperature = float(np.asarray(payload["acf_temperature"]).item())
                config_temperature = float(config.get("mamba_acf_temperature", 4.0))
                if (
                    not np.isfinite(artifact_temperature)
                    or artifact_temperature <= 0.0
                    or not np.isfinite(config_temperature)
                    or config_temperature <= 0.0
                ):
                    errors.append("mamba_acf_prior_artifact_temperature_invalid")
                elif not np.isclose(artifact_temperature, config_temperature, rtol=0.0, atol=1e-12):
                    errors.append("mamba_acf_prior_artifact_temperature_mismatch")
            except (TypeError, ValueError):
                errors.append("mamba_acf_prior_artifact_temperature_invalid")
            if values.ndim != 2 or values.shape[0] != int(config.get("enc_in", -1)):
                errors.append("mamba_acf_prior_artifact_shape_invalid")
            if values.ndim == 2 and lags.size and values.shape[1] != int(lags.size):
                errors.append("mamba_acf_prior_compact_width_mismatch")
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0 + 1e-6):
                errors.append("mamba_acf_prior_artifact_values_invalid")
            effective, effective_errors = _strict_positive_lags(
                config.get("mamba_acf_effective_lags"),
                "config_mamba_acf_effective_lags",
            )
            errors.extend(effective_errors)
            if effective is not None and sorted(int(item) for item in lags.tolist()) != effective:
                errors.append("mamba_acf_prior_artifact_lags_mismatch")
            channel_lag_bank_enabled = _bool_value(
                config.get("mamba_acf_channel_lag_bank", False)
            )
            if channel_lag_bank_enabled:
                if config.get("mamba_acf_channel_lag_bank_selection") != (
                    "abs_acf_times_legal_step_fraction"
                ):
                    errors.append("config_mamba_acf_channel_lag_selection_invalid")
                recorded_resolved_lags, recorded_resolved_errors = _strict_positive_lags(
                    config.get("mamba_acf_channel_lag_bank_resolved_lags"),
                    "config_mamba_acf_channel_lag_bank_resolved_lags",
                )
                errors.extend(recorded_resolved_errors)
                if effective is not None and recorded_resolved_lags != effective:
                    errors.append("config_mamba_acf_channel_lag_resolved_lags_invalid")
            # The channel lag-bank metadata is a derived claim, not merely a
            # free-form annotation.  Recompile it from the sidecar values and
            # forecast geometry so a forged 64-character hash or density
            # cannot pass phase validation.
            if (
                channel_lag_bank_enabled
                and values.ndim == 2
                and lags.size
                and values.shape[1] == int(lags.size)
                and effective is not None
            ):
                top_k, top_k_errors = _strict_positive_int(
                    config.get("mamba_acf_channel_lag_top_k", 2),
                    "config_mamba_acf_channel_lag_top_k",
                )
                errors.extend(top_k_errors)
                if top_k is not None:
                    try:
                        support, legal_fraction, canonical_lags = (
                            compile_channel_lag_support(
                                values,
                                lags.tolist(),
                                seq_len=config.get("seq_len"),
                                horizon=config.get("pred_len"),
                                top_k=top_k,
                            )
                        )
                    except (TypeError, ValueError):
                        errors.append("mamba_acf_prior_channel_lag_support_recompute_invalid")
                    else:
                        expected_hash = hashlib.sha256(
                            np.asarray(support, dtype=np.uint8).tobytes(order="C")
                        ).hexdigest()
                        recorded_hash = config.get(
                            "mamba_acf_channel_lag_bank_support_sha256"
                        )
                        if (
                            not isinstance(recorded_hash, str)
                            or len(recorded_hash) != 64
                            or any(char not in "0123456789abcdefABCDEF" for char in recorded_hash)
                        ):
                            errors.append("config_mamba_acf_channel_lag_support_hash_invalid")
                        elif recorded_hash.lower() != expected_hash:
                            errors.append("config_mamba_acf_channel_lag_support_hash_mismatch")
                        if list(canonical_lags) != list(effective):
                            errors.append("config_mamba_acf_channel_lag_support_lags_mismatch")
                        expected_legal_lags = [
                            int(lag)
                            for lag, fraction in zip(canonical_lags, legal_fraction.tolist())
                            if fraction > 0.0
                        ]
                        recorded_legal_lags, recorded_legal_errors = _strict_positive_lags(
                            config.get("mamba_acf_channel_lag_bank_legal_lags"),
                            "config_mamba_acf_channel_lag_bank_legal_lags",
                        )
                        errors.extend(recorded_legal_errors)
                        if recorded_legal_lags != expected_legal_lags:
                            errors.append("config_mamba_acf_channel_lag_support_legal_lags_mismatch")
                        recorded_resolved_lags, recorded_resolved_errors = _strict_positive_lags(
                            config.get("mamba_acf_channel_lag_bank_resolved_lags"),
                            "config_mamba_acf_channel_lag_bank_resolved_lags",
                        )
                        errors.extend(recorded_resolved_errors)
                        if recorded_resolved_lags != list(canonical_lags):
                            errors.append("config_mamba_acf_channel_lag_support_lags_mismatch")
                        recorded_fractions = config.get(
                            "mamba_acf_channel_lag_bank_legal_step_fractions"
                        )
                        try:
                            fraction_match = np.array_equal(
                                np.asarray(recorded_fractions, dtype=np.float32),
                                np.asarray(legal_fraction, dtype=np.float32),
                            )
                        except (TypeError, ValueError):
                            fraction_match = False
                        if not fraction_match:
                            errors.append("config_mamba_acf_channel_lag_support_fractions_mismatch")
                        expected_shape = [int(values.shape[0]), int(values.shape[1])]
                        if config.get("mamba_acf_channel_lag_bank_support_shape") != expected_shape:
                            errors.append("config_mamba_acf_channel_lag_support_shape_mismatch")
                        expected_selected = int(support.sum())
                        if config.get("mamba_acf_channel_lag_bank_selected_slots") != expected_selected:
                            errors.append("config_mamba_acf_channel_lag_support_selected_slots_mismatch")
                        expected_total = int(support.size)
                        if config.get("mamba_acf_channel_lag_bank_total_slots") != expected_total:
                            errors.append("config_mamba_acf_channel_lag_support_total_slots_mismatch")
                        expected_density = expected_selected / max(1, expected_total)
                        try:
                            density_match = bool(np.isclose(
                                float(config.get("mamba_acf_channel_lag_bank_density")),
                                expected_density,
                                rtol=0.0,
                                atol=1e-12,
                            ))
                        except (TypeError, ValueError):
                            density_match = False
                        if not density_match:
                            errors.append("config_mamba_acf_channel_lag_support_density_mismatch")
            shape_metadata = config.get("mamba_acf_prior_shape")
            if isinstance(shape_metadata, list):
                try:
                    if [int(item) for item in shape_metadata] != list(values.shape):
                        errors.append("config_mamba_acf_prior_shape_mismatch")
                except (TypeError, ValueError):
                    errors.append("config_mamba_acf_prior_shape_invalid")
            if (memory_required or relation_required) and full_values is None:
                errors.append("mamba_acf_prior_complete_table_missing")
            if full_values is not None:
                if (
                    full_values.ndim != 2
                    or full_values.shape[0] != int(config.get("enc_in", -1))
                    or full_values.shape[1] < 2
                ):
                    errors.append("mamba_acf_prior_complete_table_shape_invalid")
                elif effective is not None and max(effective) >= full_values.shape[1]:
                    errors.append("mamba_acf_prior_complete_table_lag_missing")
                prior_max_lag = config.get(
                    "mamba_acf_prior_max_lag", config.get("mamba_acf_max_lag", 720)
                )
                try:
                    prior_max_lag = int(prior_max_lag)
                except (TypeError, ValueError):
                    prior_max_lag = -1
                if prior_max_lag > 0 and full_values.shape[1] != prior_max_lag + 1:
                    errors.append("mamba_acf_prior_complete_table_width_mismatch")
                if (
                    values.ndim == 2
                    and full_values.ndim == 2
                    and effective is not None
                    and all(0 < int(lag) < full_values.shape[1] for lag in effective)
                    and full_values.shape[0] == values.shape[0]
                ):
                    selected_full = full_values[:, np.asarray(effective, dtype=np.int64)]
                    # The model accepts an unsorted sidecar lag vector as long
                    # as compact columns remain aligned with that vector;
                    # compare after applying the same canonical lag ordering.
                    compact_for_compare = values
                    if (
                        lags.size == values.shape[1]
                        and np.all(lags > 0)
                        and len(set(int(item) for item in lags.tolist())) == int(lags.size)
                    ):
                        lag_order = np.argsort(lags, kind="stable")
                        compact_for_compare = values[:, lag_order]
                    if selected_full.shape != compact_for_compare.shape:
                        errors.append(
                            "mamba_acf_prior_complete_table_compact_shape_mismatch"
                        )
                    elif not np.allclose(
                        selected_full,
                        compact_for_compare,
                        rtol=0.0,
                        atol=1e-6,
                    ):
                        errors.append("mamba_acf_prior_complete_table_compact_values_mismatch")
                if not np.all(np.isfinite(full_values)) or np.any(
                    np.abs(full_values) > 1.0 + 1e-6
                ):
                    errors.append("mamba_acf_prior_complete_table_values_invalid")
                full_shape_metadata = config.get("mamba_acf_prior_full_shape")
                if isinstance(full_shape_metadata, list):
                    try:
                        if [int(item) for item in full_shape_metadata] != list(full_values.shape):
                            errors.append("config_mamba_acf_prior_full_shape_mismatch")
                    except (TypeError, ValueError):
                        errors.append("config_mamba_acf_prior_full_shape_invalid")
    except (OSError, ValueError, EOFError, TypeError):
        errors.append("mamba_acf_prior_artifact_invalid")
    return sorted(set(errors))


def _validate_record(
    record: dict,
    job: dict,
    *,
    plan: dict | None = None,
    strict_memory: bool = False,
    require_acf_sidecar: bool = False,
) -> list[str]:
    errors = []
    if record.get("status") != "ok":
        errors.append(f"status={record.get('status')}")
    expected_artifact = record.get("expected_artifact")
    if not expected_artifact or not Path(expected_artifact).is_file():
        errors.append("expected_artifact_missing")
        return errors
    result_dir = Path(expected_artifact).parent
    config_path = result_dir / "config.json"
    if not config_path.is_file():
        errors.append("config_missing")
        return errors
    try:
        config = _load_json(config_path)
    except (OSError, ValueError):
        errors.append("config_invalid")
        return errors
    if record.get("provenance_id") is not None:
        if config.get("plan_job_id") != record.get("job_id"):
            errors.append("config_plan_job_id_mismatch")
        hash_payload = {
            "phase": job["phase"], "dataset": job["dataset"],
            "model": job["model"], "pred_len": job["pred_len"],
            "seed": job["seed"], "autocon_lambda": job["autocon_lambda"],
            "config": job["config"], "provenance_id": record.get("provenance_id"),
        }
        expected_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()[:12]
        if record.get("config_hash") != expected_hash:
            errors.append("manifest_config_hash_mismatch")
    model_spec = MODEL_SPECS[job["model"]]
    dataset_spec = DATASETS[job["dataset"]]
    cfg = job["config"]
    errors.extend(
        _validate_mamba_numeric_contract(config, model_spec["model"])
    )
    errors.extend(
        _validate_strict_representation_causality_contract(
            config, model_spec["model"]
        )
    )
    # This check is deliberately independent of the relation objective.  The
    # peak-memory ablation must prove its lag provenance even when the
    # contrastive relation loss is disabled.
    errors.extend(
        _validate_memory_lag_contract(
            config,
            job,
            plan,
            strict=(
                strict_memory
                or _bool_value(config.get("mamba_acf_anchor_copy", False))
                or _bool_value(config.get("mamba_acf_channel_lag_bank", False))
            ),
        )
    )
    # Provenance-bearing records are the only records eligible for a strict
    # confirmation aggregate.  Require the sidecar there so legacy hand-built
    # unit fixtures remain useful without weakening the launch gate.
    anchor_copy = _bool_value(config.get("mamba_acf_anchor_copy", False))
    channel_lag_bank = _bool_value(
        config.get("mamba_acf_channel_lag_bank", False)
    )
    if record.get("provenance_id") is not None or anchor_copy or channel_lag_bank:
        errors.extend(
            _validate_acf_prior_artifact(
                config,
                result_dir,
                job,
                strict=(
                    strict_memory
                    or require_acf_sidecar
                    or anchor_copy
                    or channel_lag_bank
                ),
            )
        )
    features = cfg["features"]
    enc_in, dec_in, c_out = _feature_dimensions(job["dataset"], features)
    checks = {
        "data": job["dataset"],
        "root_path": dataset_spec["root"],
        "data_path": dataset_spec["path"],
        "target": dataset_spec["target"],
        "freq": dataset_spec["freq"],
        "data_format": str(dataset_spec.get("format", "auto")),
        "data_key": str(dataset_spec.get("data_key") or "data"),
        "data_feature_index": int(dataset_spec.get("data_feature_index", 0)),
        "target_index": int(dataset_spec.get("target_index", -1)),
        "pred_len": int(job["pred_len"]),
        "run_seed": int(job["seed"]),
        "model": model_spec["model"],
        "AutoCon": _bool_value(model_spec["autocon"]),
        "features": features,
        "seq_len": int(cfg.get("seq_len", 96)),
        "label_len": int(cfg.get("label_len", 48)),
        "enc_in": enc_in,
        "dec_in": dec_in,
        "c_out": c_out,
        "d_model": int(cfg.get("d_model", 32)),
        "d_ff": int(cfg.get("d_ff", 32)),
        "e_layers": int(cfg.get("e_layers", 1)),
        "n_heads": int(cfg.get("n_heads", 8)),
        "d_layers": int(cfg.get("d_layers", 1)),
        "factor": int(cfg.get("factor", 1)),
        "top_k": int(cfg.get("top_k", 5)),
        "num_kernels": int(cfg.get("num_kernels", 6)),
        "moving_avg": int(cfg.get("moving_avg", 25)),
        "patch_len": int(cfg.get("patch_len", 16)),
        "patch_stride": int(cfg.get("patch_stride", 8)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "embed": str(cfg.get("embed", "timeF")),
        "activation": str(cfg.get("activation", "gelu")),
        "wo_freq": str(cfg.get("wo_freq", "")),
        "individual": _bool_value(cfg.get("individual", False)),
        "use_amp": _bool_value(cfg.get("use_amp", False)),
        "mamba_d_state": int(cfg.get("mamba_d_state", 16)),
        "mamba_d_conv": int(cfg.get("mamba_d_conv", 4)),
        "mamba_expand": int(cfg.get("mamba_expand", 2)),
        "mamba_headdim": int(cfg.get("mamba_headdim", 16)),
        "mamba_scales": _int_list(
            cfg.get("mamba_scales"), _DEFAULT_MAMBA_SCALES
        ),
        "mamba2_repr_forecast_fusion": _bool_value(
            cfg.get("mamba2_repr_forecast_fusion", False)
        ),
        "mamba_token_stride": int(cfg.get("mamba_token_stride", 1)),
        "mamba_bidirectional": _bool_value(cfg.get("mamba_bidirectional", False)),
        "mamba_channel_independent": _bool_value(cfg.get("mamba_channel_independent", False)),
        "mamba_channel_gate": _bool_value(cfg.get("mamba_channel_gate", False)),
        "mamba_channel_gate_rank": int(cfg.get("mamba_channel_gate_rank", 8)),
        "mamba_data_dependent_channel_mixer": _bool_value(
            cfg.get("mamba_data_dependent_channel_mixer", False)
        ),
        "mamba_data_dependent_channel_mixer_rank": int(
            cfg.get("mamba_data_dependent_channel_mixer_rank", 8)
        ),
        "mamba_acf_channel_gate": _bool_value(
            cfg.get("mamba_acf_channel_gate", False)
        ),
        "mamba_causal_embedding": _bool_value(cfg.get("mamba_causal_embedding", False)),
        "mamba_require_strict_representation_causality": _bool_value(
            cfg.get("mamba_require_strict_representation_causality", False)
        ),
        "mamba_variable_mixer": _bool_value(cfg.get("mamba_variable_mixer", False)),
        "mamba_variable_mixer_depth": int(cfg.get("mamba_variable_mixer_depth", 1)),
        "mamba_variable_group_size": int(cfg.get("mamba_variable_group_size", 1)),
        "mamba_variable_repr_fusion": _bool_value(
            cfg.get("mamba_variable_repr_fusion", False)
        ),
        "mamba_variable_acf_memory_selector": _bool_value(
            cfg.get("mamba_variable_acf_memory_selector", False)
        ),
        "mamba_variable_permutation": _bool_value(cfg.get("mamba_variable_permutation", False)),
        "mamba_acf_channel_order": str(cfg.get("mamba_acf_channel_order", "input")),
        "mamba_acf_channel_conditioning": _bool_value(
            cfg.get("mamba_acf_channel_conditioning", False)
        ),
        "mamba_acf_repr_fusion": _bool_value(
            cfg.get("mamba_acf_repr_fusion", False)
        ),
        "mamba_patch_mixer": _bool_value(cfg.get("mamba_patch_mixer", False)),
        "mamba_patch_mixer_depth": int(cfg.get("mamba_patch_mixer_depth", 1)),
        "mamba_patch_len": int(cfg.get("mamba_patch_len", 16)),
        "mamba_patch_stride": int(cfg.get("mamba_patch_stride", 8)),
        "mamba_patch_order": str(cfg.get("mamba_patch_order", "time_major")),
        "mamba_patch_permutation": _bool_value(cfg.get("mamba_patch_permutation", False)),
        "mamba_local_mixer": _bool_value(cfg.get("mamba_local_mixer", False)),
        "mamba_local_patch_len": int(cfg.get("mamba_local_patch_len", 16)),
        "mamba_local_patch_stride": int(cfg.get("mamba_local_patch_stride", 8)),
        "mamba_local_width": int(
            cfg.get("mamba_local_width")
            if cfg.get("mamba_local_width") is not None
            else cfg.get("d_model", 32)
        ),
        "mamba_local_rank": int(cfg.get("mamba_local_rank", 16)),
        "mamba_local_depth": int(cfg.get("mamba_local_depth", 1)),
        "mamba_low_rank_head": _bool_value(cfg.get("mamba_low_rank_head", False)),
        "mamba_temporal_rank": int(cfg.get("mamba_temporal_rank", 16)),
        "mamba_acf_lags": _int_list(cfg.get("mamba_acf_lags"), [24, 168, 336]),
        "mamba_acf_channel_lag_bank": _bool_value(
            cfg.get("mamba_acf_channel_lag_bank", False)
        ),
        "mamba_acf_channel_lag_top_k": int(
            cfg.get("mamba_acf_channel_lag_top_k", 2)
        ),
        "mamba_acf_lag_mode": str(cfg.get("mamba_acf_lag_mode", "fixed")),
        "mamba_acf_max_lag": int(cfg.get("mamba_acf_max_lag", 720)),
        "mamba_acf_top_k": int(cfg.get("mamba_acf_top_k", 8)),
        "mamba_acf_min_lag": int(cfg.get("mamba_acf_min_lag", 2)),
        "mamba_acf_min_abs": float(cfg.get("mamba_acf_min_abs", 0.1)),
        "mamba_acf_confidence_z": float(cfg.get("mamba_acf_confidence_z", 1.96)),
        "mamba_acf_min_separation": int(cfg.get("mamba_acf_min_separation", 2)),
        "mamba_acf_detrend": str(cfg.get("mamba_acf_detrend", "mean")),
        "mamba_acf_chunk": int(cfg.get("mamba_acf_chunk", 64)),
        "mamba_use_acf_memory": _bool_value(cfg.get("mamba_use_acf_memory", False)),
        "mamba_dynamic_acf": _bool_value(cfg.get("mamba_dynamic_acf", False)),
        "mamba_acf_temperature": float(cfg.get("mamba_acf_temperature", 4.0)),
        "mamba_acf_step_reliability": _bool_value(cfg.get("mamba_acf_step_reliability", False)),
        "mamba_acf_reliability_interpolation": _bool_value(
            cfg.get("mamba_acf_reliability_interpolation", False)
        ),
        "mamba_acf_memory_gate_mode": str(
            cfg.get("mamba_acf_memory_gate_mode", "hidden")
        ).lower(),
        "mamba_acf_anchor_copy": _bool_value(
            cfg.get("mamba_acf_anchor_copy", False)
        ),
        "mamba_acf_horizon_calibration": _bool_value(
            cfg.get("mamba_acf_horizon_calibration", False)
        ),
        "mamba_acf_horizon_calibration_rank": int(
            cfg.get("mamba_acf_horizon_calibration_rank", 2)
        ),
        "mamba_acf_relation_loss": _bool_value(cfg.get("mamba_acf_relation_loss", False)),
        "mamba_acf_relation_batching": str(cfg.get("mamba_acf_relation_batching", "lag_pairs")),
        "mamba_acf_relation_channel_reduce": str(cfg.get("mamba_acf_relation_channel_reduce", "mean")),
        "mamba_acf_relation_scope": str(cfg.get("mamba_acf_relation_scope", "both")),
        "mamba_acf_relation_representation": str(
            cfg.get("mamba_acf_relation_representation", "encoder")
        ).lower(),
        "mamba_acf_relation_lag_mask": _bool_value(cfg.get("mamba_acf_relation_lag_mask", False)),
        "mamba_acf_relation_lag_tolerance": int(cfg.get("mamba_acf_relation_lag_tolerance", 0)),
        "mamba_acf_relation_min_lag": int(cfg.get("mamba_acf_relation_min_lag", 0)),
        "mamba_acf_relation_target_mode": str(
            cfg.get("mamba_acf_relation_target_mode", "hard_selected")
        ).lower(),
        "mamba_acf_relation_soft_min_abs": float(
            cfg.get("mamba_acf_relation_soft_min_abs", 0.1)
        ),
        "mamba_acf_relation_diagnostic_min_abs": float(
            cfg.get("mamba_acf_relation_diagnostic_min_abs", 0.1)
        ),
        "mamba_acf_relation_selected_lag_boost": float(
            cfg.get("mamba_acf_relation_selected_lag_boost", 0.0)
        ),
        "AutoCon_multiscales": list(cfg.get("autocon_scale", [96]) if isinstance(cfg.get("autocon_scale", [96]), list) else [cfg.get("autocon_scale", 96)]),
        "AutoCon_warmup_epochs": int(cfg.get("autocon_warmup", 1)),
        "AutoCon_acf_source": _acf_source_value(cfg),
        "AutoCon_temperature": float(cfg.get("AutoCon_temperature", 0.2)),
        "AutoCon_relation_temperature": float(cfg.get("AutoCon_relation_temperature", 0.25)),
        "AutoCon_sign_weight": float(cfg.get("AutoCon_sign_weight", 0.25)),
        "AutoCon_sign_margin": float(cfg.get("AutoCon_sign_margin", 0.1)),
        "batch_size": int(cfg.get("batch_size", 256)),
        "train_epochs": int(cfg.get("epochs", 20)),
        "patience": int(cfg.get("patience", 5)),
        "learning_rate": float(cfg.get("learning_rate", 1e-3)),
        "lradj": str(cfg.get("lradj", "type1")),
        "min_learning_rate": float(cfg.get("min_learning_rate", 0.0)),
        "weight_decay": float(cfg.get("weight_decay", 1e-5)),
        "num_workers": int(cfg.get("num_workers", 0)),
        "drop_last_train": _bool_value(cfg.get("drop_last_train", False)),
        "train_ratio": float(cfg.get("train_ratio", 0.6)),
        "save": _bool_value(cfg.get("save_predictions", False)),
        "skip_test": _bool_value(cfg.get("skip_test", False)),
        "stream_metrics": _bool_value(cfg.get("stream_metrics", False)),
        "origin_stride": cfg.get("origin_stride"),
        # The runner only serializes ``stream_sample_windows`` to the CLI
        # when streaming metrics are enabled.  With streaming disabled the
        # runtime parser keeps its default (4), so a plan-local sample limit
        # is not an executable contract for ordinary validation-only jobs.
        "stream_sample_windows": (
            int(cfg.get("stream_sample_windows", 4))
            if _bool_value(cfg.get("stream_metrics", False))
            else 4
        ),
    }
    if dataset_spec.get("data_interval_minutes") is not None:
        checks["data_interval_minutes"] = int(
            dataset_spec["data_interval_minutes"]
        )
    # These options were added after the first frozen phases were run.  Their
    # absence in an older config is unambiguous because the original CLI
    # defaults are the same values used by the plan validator.  Preserve strict
    # checking for explicitly recorded values while allowing legacy configs to
    # be revalidated after the schema grew.
    legacy_defaults = {
        "data_format": "auto",
        "data_key": "data",
        "data_feature_index": 0,
        "target_index": -1,
        "data_interval_minutes": None,
        "min_learning_rate": 0.0,
        "drop_last_train": False,
        "stream_metrics": False,
        "stream_sample_windows": 4,
        # Mamba/ACF options were added after the frozen legacy phases.  A
        # missing key therefore means the exact parser default, not an
        # unknown or implicitly truthy value.
        "mamba_scales": _DEFAULT_MAMBA_SCALES,
        "mamba_token_stride": 1,
        "mamba_bidirectional": False,
        "mamba_channel_independent": False,
        "mamba_channel_gate": False,
        "mamba_channel_gate_rank": 8,
        "mamba_data_dependent_channel_mixer": False,
        "mamba_data_dependent_channel_mixer_rank": 8,
        "mamba_acf_channel_gate": False,
        "mamba_causal_embedding": False,
        "mamba_require_strict_representation_causality": False,
        "mamba_variable_mixer": False,
        "mamba_variable_mixer_depth": 1,
        "mamba_variable_group_size": 1,
        "mamba_variable_repr_fusion": False,
        "mamba_variable_permutation": False,
        "mamba_acf_channel_order": "input",
        "mamba_acf_channel_conditioning": False,
        "mamba_acf_repr_fusion": False,
        "mamba_patch_mixer": False,
        "mamba_patch_mixer_depth": 1,
        "mamba_patch_len": 16,
        "mamba_patch_stride": 8,
        "mamba_patch_order": "time_major",
        "mamba_patch_permutation": False,
        "mamba_local_mixer": False,
        "mamba_local_patch_len": 16,
        "mamba_local_patch_stride": 8,
        "mamba_local_width": None,
        "mamba_local_rank": 16,
        "mamba_local_depth": 1,
        "mamba_low_rank_head": False,
        "mamba_temporal_rank": 16,
        "mamba_acf_lags": [24, 168, 336],
        "mamba_acf_channel_lag_bank": False,
        "mamba_acf_channel_lag_top_k": 2,
        "mamba_acf_lag_mode": "fixed",
        "mamba_acf_max_lag": 720,
        "mamba_acf_top_k": 8,
        "mamba_acf_min_lag": 2,
        "mamba_acf_min_abs": 0.1,
        "mamba_acf_confidence_z": 1.96,
        "mamba_acf_min_separation": 2,
        "mamba_acf_detrend": "mean",
        "mamba_acf_chunk": 64,
        "mamba_use_acf_memory": False,
        "mamba_dynamic_acf": False,
        "mamba_acf_temperature": 4.0,
        "mamba_acf_step_reliability": False,
        "mamba_acf_reliability_interpolation": False,
        "mamba_acf_memory_gate_mode": "hidden",
        "mamba_acf_anchor_copy": False,
        "mamba_acf_horizon_calibration": False,
        "mamba_acf_horizon_calibration_rank": 2,
        "mamba_acf_relation_loss": False,
        "mamba_acf_relation_batching": "lag_pairs",
        "mamba_acf_relation_channel_reduce": "mean",
        "mamba_acf_relation_scope": "both",
        "mamba_acf_relation_representation": "encoder",
        "mamba_acf_relation_lag_mask": False,
        "mamba_acf_relation_lag_tolerance": 0,
        "mamba_acf_relation_min_lag": 0,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_relation_soft_min_abs": 0.1,
        "mamba_acf_relation_diagnostic_min_abs": 0.1,
        "mamba_acf_relation_selected_lag_boost": 0.0,
        "AutoCon_acf_source": "residual",
        "AutoCon_temperature": 0.2,
        "AutoCon_relation_temperature": 0.25,
        "AutoCon_sign_weight": 0.25,
        "AutoCon_sign_margin": 0.1,
    }
    if "mamba_local_width" not in config:
        # The CLI default is ``d_model``; resolve that value for older
        # artifacts whose config predates the local adapter fields.
        legacy_defaults["mamba_local_width"] = int(config.get("d_model", 32))
    for key, expected in checks.items():
        actual = config.get(key, legacy_defaults.get(key))
        if key == "mamba_local_width" and actual is None:
            actual = int(config.get("d_model", 32))
        if key in {
            "mamba_bidirectional", "mamba_channel_independent", "mamba_channel_gate",
            "mamba_data_dependent_channel_mixer",
            "mamba_acf_channel_gate",
            "mamba_acf_channel_lag_bank",
            "mamba_causal_embedding",
            "mamba_require_strict_representation_causality",
            "mamba_variable_mixer", "mamba_variable_repr_fusion",
            "mamba_variable_acf_memory_selector",
            "mamba2_repr_forecast_fusion",
            "mamba_variable_permutation",
            "mamba_acf_channel_conditioning",
            "mamba_acf_repr_fusion",
            "mamba_patch_mixer", "mamba_patch_permutation", "mamba_low_rank_head",
            "mamba_local_mixer",
            "mamba_use_acf_memory", "mamba_dynamic_acf", "mamba_acf_relation_loss",
            "mamba_acf_step_reliability",
            "mamba_acf_reliability_interpolation",
            "mamba_acf_anchor_copy",
            "mamba_acf_horizon_calibration",
            "mamba_acf_relation_lag_mask",
            "individual", "use_amp", "drop_last_train", "save", "skip_test", "stream_metrics",
        }:
            actual = _bool_value(actual)
        elif key in {
            "mamba_acf_channel_lag_top_k",
            "mamba_data_dependent_channel_mixer_rank",
        }:
            try:
                actual = int(actual)
            except (TypeError, ValueError):
                actual = None
        elif key in {"mamba_scales", "mamba_acf_lags", "AutoCon_multiscales"}:
            actual = _int_list(actual, [])
        # ``--mamba_acf_relation_lag_mask`` is emitted by the runner only
        # when the relation objective is active.  Treat the flag as
        # relation-scoped for plain/memory arms so a global plan default does
        # not create a false config mismatch in otherwise valid sidecar runs.
        expected_value = expected
        if (
            key == "mamba_acf_relation_lag_mask"
            and not _bool_value(config.get("mamba_acf_relation_loss", False))
        ):
            expected_value = False
        if isinstance(expected, float):
            okay = _float_equal(actual, expected_value)
        else:
            okay = actual == expected_value
        if not okay:
            errors.append(f"config_{key}={actual!r}, expected={expected!r}")
    if model_spec["autocon"] and not _float_equal(config.get("AutoCon_lambda"), job["autocon_lambda"]):
        errors.append("config_AutoCon_lambda_mismatch")
    if config.get("AutoCon_wnorm", "LastVal") != cfg.get("autocon_wnorm", "LastVal"):
        errors.append("config_AutoCon_wnorm_mismatch")
    errors.extend(
        _validate_mamba_acf_wnorm_contract(config, model_spec["model"])
    )
    if _acf_source_value(config) != _acf_source_value(cfg):
        errors.append("config_acf_source_mismatch")

    # Keep the resolved lag policy closed under the live CLI contract.  The
    # equality checks above catch a plan/config mismatch, but would otherwise
    # accept the same unknown string on both sides of a hand-edited record.
    model_name = str(model_spec["model"])
    supports_variable_repr_fusion = (
        model_name == "NLinearVariableMambaAutoCon"
        or ("MambaACF" in model_name and model_name.endswith("AutoCon"))
    )
    if "MambaACF" in model_name:
        lag_mode = str(config.get("mamba_acf_lag_mode", "fixed")).lower()
        if lag_mode not in {"fixed", "train_peaks", "train_peaks_available"}:
            errors.append("config_mamba_acf_lag_mode_invalid")
        try:
            relation_min_lag = int(config.get("mamba_acf_relation_min_lag", 0))
        except (TypeError, ValueError):
            relation_min_lag = -1
        if relation_min_lag < 0:
            errors.append("config_mamba_acf_relation_min_lag_invalid")
        try:
            relation_max_lag = int(config.get("mamba_acf_max_lag", 720))
        except (TypeError, ValueError):
            relation_max_lag = -1
        if relation_max_lag <= 0:
            errors.append("config_mamba_acf_max_lag_invalid")
        if (
            _bool_value(config.get("mamba_acf_relation_loss", False))
            and relation_min_lag >= 0
            and relation_max_lag > 0
            and relation_min_lag > relation_max_lag
        ):
            errors.append("config_mamba_acf_relation_min_lag_exceeds_max_lag")
        relation_target_mode = str(
            config.get("mamba_acf_relation_target_mode", "hard_selected")
        ).lower()
        if relation_target_mode not in {
            "hard_selected", "soft_multi_peak", "soft_full"
        }:
            errors.append("config_mamba_acf_relation_target_mode_invalid")
        relation_representation = str(
            config.get("mamba_acf_relation_representation", "encoder")
        ).lower()
        if relation_representation not in {"encoder", "forecast"}:
            errors.append("config_mamba_acf_relation_representation_invalid")
        elif relation_representation == "forecast":
            if str(model_spec["model"]) not in {
                "MambaACFAutoCon", "MambaACF1AutoCon"
            }:
                errors.append(
                    "config_mamba_acf_relation_representation_requires_mamba_acf_autocon"
                )
            if not _bool_value(config.get("mamba_acf_relation_loss", False)):
                errors.append(
                    "config_mamba_acf_relation_representation_requires_relation_loss"
                )
            if str(config.get("mamba_acf_relation_scope", "both")).lower() != "global":
                errors.append(
                    "config_mamba_acf_relation_representation_requires_global_scope"
                )
            if _acf_source_value(config) != "raw":
                errors.append(
                    "config_mamba_acf_relation_representation_requires_raw_source"
                )
            if str(config.get("features", "")).upper() != "M":
                errors.append(
                    "config_mamba_acf_relation_representation_requires_multivariate_features"
                )
            if config.get("c_out") != config.get("enc_in"):
                errors.append(
                    "config_mamba_acf_relation_representation_requires_matching_output_channels"
                )
            try:
                forecast_relation_channels = int(config.get("enc_in"))
            except (TypeError, ValueError):
                forecast_relation_channels = 0
            if forecast_relation_channels < 2:
                errors.append(
                    "config_mamba_acf_relation_representation_requires_at_least_two_channels"
                )
            if _bool_value(config.get("mamba_acf_repr_fusion", False)) or _bool_value(
                config.get("mamba_variable_repr_fusion", False)
            ):
                errors.append(
                    "config_mamba_acf_relation_representation_incompatible_with_encoder_fusion"
                )
        try:
            relation_soft_min_abs = float(
                config.get("mamba_acf_relation_soft_min_abs", 0.1)
            )
        except (TypeError, ValueError):
            relation_soft_min_abs = float("nan")
        if not math.isfinite(relation_soft_min_abs) or not 0 <= relation_soft_min_abs <= 1:
            errors.append("config_mamba_acf_relation_soft_min_abs_invalid")
        try:
            relation_diagnostic_min_abs = float(
                config.get("mamba_acf_relation_diagnostic_min_abs", 0.1)
            )
        except (TypeError, ValueError):
            relation_diagnostic_min_abs = float("nan")
        if (
            not math.isfinite(relation_diagnostic_min_abs)
            or not 0 <= relation_diagnostic_min_abs <= 1
        ):
            errors.append("config_mamba_acf_relation_diagnostic_min_abs_invalid")
        try:
            relation_selected_lag_boost = float(
                config.get("mamba_acf_relation_selected_lag_boost", 0.0)
            )
        except (TypeError, ValueError):
            relation_selected_lag_boost = float("nan")
        if (
            not math.isfinite(relation_selected_lag_boost)
            or relation_selected_lag_boost < 0
        ):
            errors.append("config_mamba_acf_relation_selected_lag_boost_invalid")
        try:
            calibration_rank = int(
                config.get("mamba_acf_horizon_calibration_rank", 2)
            )
        except (TypeError, ValueError):
            calibration_rank = 0
        if calibration_rank not in {1, 2}:
            errors.append("config_mamba_acf_horizon_calibration_rank_invalid")
        memory_gate_mode = str(
            config.get("mamba_acf_memory_gate_mode", "hidden")
        ).lower()
        if memory_gate_mode not in {"hidden", "variable_token"}:
            errors.append("config_mamba_acf_memory_gate_mode_invalid")
        elif memory_gate_mode == "variable_token":
            if (
                model_name != "NLinearVariableMambaAutoCon"
                and not _bool_value(config.get("mamba_variable_mixer", False))
            ):
                errors.append(
                    "config_mamba_acf_memory_gate_mode_requires_variable_mixer"
                )
            if not _bool_value(config.get("mamba_use_acf_memory", False)):
                errors.append(
                    "config_mamba_acf_memory_gate_mode_requires_memory"
                )
            if _bool_value(config.get("mamba_dynamic_acf", False)):
                errors.append(
                    "config_mamba_acf_memory_gate_mode_requires_static_memory"
                )
        anchor_copy = _bool_value(config.get("mamba_acf_anchor_copy", False))
        if anchor_copy:
            if _bool_value(config.get("mamba_dynamic_acf", False)):
                errors.append("config_mamba_acf_anchor_copy_requires_static_memory")
            if _acf_source_value(config) != "raw":
                errors.append("config_mamba_acf_anchor_copy_requires_raw_source")
            anchor_lags, anchor_lag_errors = _strict_positive_lags(
                config.get("mamba_acf_lags"),
                "config_mamba_acf_anchor_lags",
            )
            errors.extend(anchor_lag_errors)
        channel_lag_bank = _bool_value(
            config.get("mamba_acf_channel_lag_bank", False)
        )
        channel_lag_top_k, channel_lag_top_k_errors = _strict_positive_int(
            config.get("mamba_acf_channel_lag_top_k", 2),
            "config_mamba_acf_channel_lag_top_k",
        )
        errors.extend(channel_lag_top_k_errors)
        if channel_lag_top_k is None:
            channel_lag_top_k = 0
        if channel_lag_bank:
            if not _bool_value(config.get("mamba_use_acf_memory", False)):
                errors.append("config_mamba_acf_channel_lag_bank_requires_memory")
            if _bool_value(config.get("mamba_dynamic_acf", False)):
                errors.append(
                    "config_mamba_acf_channel_lag_bank_requires_static_memory"
                )
            if channel_lag_top_k <= 0:
                errors.append("config_mamba_acf_channel_lag_top_k_invalid")
            try:
                configured_lags = _strict_positive_lags(
                    config.get("mamba_acf_lags"),
                    "config_mamba_acf_channel_lag_lags",
                )
                lag_values, lag_errors = configured_lags
                errors.extend(lag_errors)
            except (TypeError, ValueError):
                lag_values = []
            if lag_values and channel_lag_top_k > len(lag_values):
                errors.append(
                    "config_mamba_acf_channel_lag_top_k_exceeds_lag_count"
                )
            effective_lags, effective_lag_errors = _strict_positive_lags(
                config.get("mamba_acf_effective_lags"),
                "config_mamba_acf_channel_lag_effective_lags",
            )
            errors.extend(effective_lag_errors)
            if effective_lags is not None:
                if channel_lag_top_k > len(effective_lags):
                    errors.append(
                        "config_mamba_acf_channel_lag_top_k_exceeds_effective_lag_count"
                    )
                try:
                    seq_len = int(config.get("seq_len", 0))
                    pred_len = int(config.get("pred_len", 0))
                except (TypeError, ValueError):
                    seq_len = 0
                    pred_len = 0
                if seq_len <= 0 or pred_len <= 0:
                    errors.append("config_mamba_acf_channel_lag_geometry_invalid")
                else:
                    steps = np.arange(1, pred_len + 1, dtype=np.int64)[:, None]
                    lag_array = np.asarray(effective_lags, dtype=np.int64)[None, :]
                    valid_fraction = (
                        (
                            (seq_len - 1 + steps - lag_array >= 0)
                            & (seq_len - 1 + steps - lag_array < seq_len)
                        )
                        .astype(np.float32)
                        .mean(axis=0, dtype=np.float32)
                    )
                    if int(np.sum(valid_fraction > 0.0)) < channel_lag_top_k:
                        errors.append(
                            "config_mamba_acf_channel_lag_insufficient_legal_slots"
                        )
                    expected_legal_lags = [
                        int(lag)
                        for lag, fraction in zip(effective_lags, valid_fraction.tolist())
                        if fraction > 0.0
                    ]
                    recorded_fractions = config.get(
                        "mamba_acf_channel_lag_bank_legal_step_fractions"
                    )
                    try:
                        matching_fractions = np.allclose(
                            np.asarray(recorded_fractions, dtype=np.float32),
                            valid_fraction.astype(np.float32),
                            rtol=0.0,
                            atol=0.0,
                        )
                    except (TypeError, ValueError):
                        matching_fractions = False
                    if not matching_fractions:
                        errors.append(
                            "config_mamba_acf_channel_lag_legal_step_fractions_invalid"
                        )
                    recorded_legal_lags, recorded_legal_errors = _strict_positive_lags(
                        config.get("mamba_acf_channel_lag_bank_legal_lags"),
                        "config_mamba_acf_channel_lag_bank_legal_lags",
                    )
                    errors.extend(recorded_legal_errors)
                    if recorded_legal_lags != expected_legal_lags:
                        errors.append("config_mamba_acf_channel_lag_legal_lags_invalid")
                recorded_resolved_lags, recorded_resolved_errors = _strict_positive_lags(
                    config.get("mamba_acf_channel_lag_bank_resolved_lags"),
                    "config_mamba_acf_channel_lag_bank_resolved_lags",
                )
                errors.extend(recorded_resolved_errors)
                if recorded_resolved_lags != effective_lags:
                    errors.append("config_mamba_acf_channel_lag_resolved_lags_invalid")
                expected_selection = "abs_acf_times_legal_step_fraction"
                if config.get("mamba_acf_channel_lag_bank_selection") != expected_selection:
                    errors.append(
                        "config_mamba_acf_channel_lag_selection_invalid"
                    )
                try:
                    expected_channels = int(config.get("enc_in", 0))
                except (TypeError, ValueError):
                    expected_channels = 0
                expected_shape = [expected_channels, len(effective_lags)]
                if config.get("mamba_acf_channel_lag_bank_support_shape") != expected_shape:
                    errors.append(
                        "config_mamba_acf_channel_lag_support_shape_invalid"
                    )
                selected_slots = config.get("mamba_acf_channel_lag_bank_selected_slots")
                if selected_slots != expected_shape[0] * channel_lag_top_k:
                    errors.append(
                        "config_mamba_acf_channel_lag_selected_slots_invalid"
                    )
                total_slots = config.get("mamba_acf_channel_lag_bank_total_slots")
                if total_slots != expected_shape[0] * expected_shape[1]:
                    errors.append(
                        "config_mamba_acf_channel_lag_total_slots_invalid"
                    )
                expected_density = (
                    float(expected_shape[0] * channel_lag_top_k)
                    / max(1, expected_shape[0] * expected_shape[1])
                )
                if not _float_equal(
                    config.get("mamba_acf_channel_lag_bank_density"), expected_density
                ):
                    errors.append("config_mamba_acf_channel_lag_density_invalid")
                support_hash = config.get("mamba_acf_channel_lag_bank_support_sha256")
                if not isinstance(support_hash, str) or len(support_hash) != 64:
                    errors.append("config_mamba_acf_channel_lag_support_hash_invalid")
        group_size = int(config.get("mamba_variable_group_size", 1))
        if group_size <= 0:
            errors.append("config_mamba_variable_group_size_invalid")
        if group_size > 1 and not _bool_value(config.get("mamba_variable_mixer", False)):
            errors.append("config_mamba_variable_group_size_requires_variable_mixer")
        channel_order = str(config.get("mamba_acf_channel_order", "input")).lower()
        if channel_order not in {"input", "strength", "similarity"}:
            errors.append("config_mamba_acf_channel_order_invalid")
        elif channel_order != "input" and not _bool_value(
                config.get("mamba_variable_mixer", False)):
            errors.append("config_mamba_acf_channel_order_requires_variable_mixer")
        if _bool_value(config.get("mamba_acf_channel_gate", False)) and not _bool_value(
            config.get("mamba_channel_gate", False)
        ):
            errors.append("config_mamba_acf_channel_gate_requires_channel_gate")
        if _bool_value(config.get("mamba_variable_repr_fusion", False)):
            if not _bool_value(config.get("AutoCon", False)):
                errors.append("config_mamba_variable_repr_fusion_requires_autocon")
            if not _bool_value(config.get("mamba_variable_mixer", False)):
                errors.append(
                    "config_mamba_variable_repr_fusion_requires_variable_mixer"
                )
            if not str(model_spec.get("model", "")).endswith("AutoCon"):
                errors.append(
                    "config_mamba_variable_repr_fusion_requires_mamba_acf_autocon"
                )
        if _bool_value(config.get("mamba_variable_acf_memory_selector", False)):
            try:
                selector_seq_len = int(config.get("seq_len", 0))
                selector_min_lag = int(
                    config.get("mamba_acf_relation_min_lag", 0)
                )
            except (TypeError, ValueError):
                errors.append("config_mamba_variable_acf_selector_geometry_invalid")
            else:
                if selector_min_lag >= selector_seq_len:
                    errors.append(
                        "config_mamba_variable_acf_selector_requires_legal_lag"
                    )
                resolved_lags = config.get("mamba_acf_prior_lags", [])
                if isinstance(resolved_lags, list) and resolved_lags:
                    try:
                        if not any(int(lag) <= selector_seq_len for lag in resolved_lags):
                            errors.append(
                                "config_mamba_variable_acf_selector_has_no_accessible_lag"
                            )
                    except (TypeError, ValueError):
                        errors.append(
                            "config_mamba_variable_acf_selector_lags_invalid"
                        )
    elif supports_variable_repr_fusion and _bool_value(
        config.get("mamba_variable_repr_fusion", False)
    ):
        if not _bool_value(config.get("AutoCon", False)):
            errors.append("config_mamba_variable_repr_fusion_requires_autocon")
    elif str(config.get("mamba_acf_channel_order", "input")).lower() != "input":
        errors.append("config_mamba_acf_channel_order_requires_mamba_acf")
    elif int(config.get("mamba_variable_group_size", 1)) != 1:
        errors.append("config_mamba_variable_group_size_requires_mamba_acf")
    elif _bool_value(config.get("mamba_variable_repr_fusion", False)):
        errors.append("config_mamba_variable_repr_fusion_requires_mamba_acf")
    if "MambaACF" not in model_spec["model"] and _bool_value(
        config.get("mamba_acf_channel_lag_bank", False)
    ):
        errors.append("config_mamba_acf_channel_lag_bank_requires_mamba_acf")
    if str(config.get("mamba_acf_memory_gate_mode", "hidden")).lower() != "hidden":
        if "MambaACF" not in model_spec["model"]:
            errors.append("config_mamba_acf_memory_gate_mode_requires_mamba_acf")
    if _bool_value(config.get("mamba_acf_anchor_copy", False)) and "MambaACF" not in model_spec["model"]:
        errors.append("config_mamba_acf_anchor_copy_requires_mamba_acf")
    if _bool_value(config.get("mamba_local_mixer", False)) and str(
        model_spec["model"]
    ) not in {"MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"}:
        errors.append("config_mamba_local_mixer_requires_mamba_acf")
    if _bool_value(config.get("mamba_data_dependent_channel_mixer", False)) and str(
        model_spec["model"]
    ) not in {"MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"}:
        errors.append("config_mamba_data_dependent_channel_mixer_requires_mamba_acf")
    if _bool_value(config.get("mamba_data_dependent_channel_mixer", False)) and _bool_value(
        config.get("mamba_channel_independent", False)
    ):
        errors.append(
            "config_mamba_data_dependent_channel_mixer_incompatible_with_channel_independent"
        )
    if _bool_value(config.get("mamba_local_mixer", False)) and _bool_value(
        config.get("mamba_channel_independent", False)
    ):
        errors.append("config_mamba_local_mixer_incompatible_with_channel_independent")
    if "MambaACF" not in model_spec["model"] and _bool_value(
        config.get("mamba_acf_channel_gate", False)
    ):
        errors.append("config_mamba_acf_channel_gate_requires_mamba_acf")
    if _bool_value(config.get("mamba_acf_horizon_calibration", False)):
        if str(model_spec["model"]) not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        }:
            errors.append("config_mamba_acf_horizon_calibration_requires_mamba_acf")
        if not _bool_value(config.get("mamba_use_acf_memory", False)):
            errors.append("config_mamba_acf_horizon_calibration_requires_memory")
        if _bool_value(config.get("mamba_dynamic_acf", False)):
            errors.append("config_mamba_acf_horizon_calibration_requires_static_memory")
    if _bool_value(config.get("mamba_acf_reliability_interpolation", False)):
        if str(model_spec["model"]) not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        }:
            errors.append("config_mamba_acf_reliability_interpolation_requires_mamba_acf")
        if not _bool_value(config.get("mamba_use_acf_memory", False)):
            errors.append("config_mamba_acf_reliability_interpolation_requires_memory")
        if _bool_value(config.get("mamba_dynamic_acf", False)):
            errors.append("config_mamba_acf_reliability_interpolation_requires_static_memory")
        if _acf_source_value(config) != "raw":
            errors.append("config_mamba_acf_reliability_interpolation_requires_raw_source")
        if _bool_value(config.get("mamba_acf_step_reliability", False)):
            errors.append("config_mamba_acf_reliability_interpolation_incompatible_with_step_reliability")

    command = record.get("command", [])
    if record.get("command_sha256") and _command_hash(command) != record["command_sha256"]:
        errors.append("command_hash_mismatch")
    checkpoint_root = _arg(command, "--checkpoints")
    if checkpoint_root:
        checkpoint_dir = Path(checkpoint_root) / result_dir.name
        for name in ("checkpoint.pth", "loss_logs.json"):
            if not (checkpoint_dir / name).is_file():
                errors.append(f"{name}_missing")
        if (checkpoint_dir / "loss_logs.json").is_file():
            try:
                logs = _load_json(checkpoint_dir / "loss_logs.json")
                if not logs.get("val_loss"):
                    errors.append("validation_trajectory_empty")
                relation_loss = _bool_value(config.get("mamba_acf_relation_loss", False))
                if relation_loss:
                    # Version 3 records explicit hard/soft target semantics,
                    # candidate lower bounds, and local coverage.  Earlier
                    # relation logs cannot enter this repaired-phase aggregate.
                    if logs.get("acf_relation_objective_version") != 3:
                        errors.append("acf_relation_objective_version_missing_or_old")
                    for key in (
                        "acf_relation_global_candidate_fraction",
                        "acf_relation_global_candidate_active_anchor_fraction",
                        "acf_relation_global_positive_count_mean",
                        "acf_relation_global_high_acf_excluded_fraction",
                        "acf_relation_local_target_fraction",
                        "acf_relation_local_active_position_fraction",
                    ):
                        if key not in logs:
                            errors.append(f"{key}_missing")
                    # Provenance-bearing records (and any record carrying the
                    # new resolved-lag fields) are held to the full relation
                    # feasibility contract.  Legacy hand-built records keep
                    # the historical v3 presence check above, but are not
                    # eligible for the strict confirmation analyzer.
                # Validate sampler metadata for both active and inactive
                # relation objectives.  Inactive AutoCon logs may contain
                # constructor defaults, but must never contain accepted pairs.
                strict_relation = record.get("provenance_id") is not None
                errors.extend(
                    _validate_relation_contract(
                        config,
                        logs,
                        strict=strict_relation,
                    )
                )
            except (OSError, ValueError):
                errors.append("loss_logs_invalid")
        expected_hashes = record.get("artifact_hashes")
        if expected_hashes:
            actual_hashes = _artifact_hashes(result_dir, checkpoint_dir)
            if actual_hashes != expected_hashes:
                errors.append("artifact_hashes_mismatch")

    expected_skip_test = bool(cfg.get("skip_test", False))
    if expected_skip_test:
        marker = result_dir / "training_complete.json"
        if not marker.is_file():
            errors.append("training_complete_missing")
        else:
            try:
                marker_payload = _load_json(marker)
                if marker_payload.get("status") != "trained_without_test":
                    errors.append("training_complete_status_invalid")
            except (OSError, ValueError):
                errors.append("training_complete_invalid")
    else:
        metrics_path = result_dir / "metrics.npy"
        if metrics_path.is_file():
            try:
                metrics = np.asarray(np.load(metrics_path), dtype=float).reshape(-1)
            except (OSError, ValueError, EOFError, ImportError):
                errors.append("metrics_invalid")
                metrics = np.asarray([], dtype=float)
            if metrics.size < 3 or not np.all(np.isfinite(metrics[:3])):
                errors.append("metrics_invalid")
        else:
            errors.append("metrics_missing")
    expected_stream = bool(cfg.get("stream_metrics", False))
    if expected_stream:
        errors.extend(_validate_stream_metrics(result_dir, config, job))
    expected_save = bool(cfg.get("save_predictions", False))
    if expected_save:
        pred_path, true_path = result_dir / "pred.npy", result_dir / "true.npy"
        original_metrics_path = result_dir / "original_metrics.npy"
        channel_metrics_path = result_dir / "channel_metrics.npy"
        if not pred_path.is_file() or not true_path.is_file():
            errors.append("prediction_arrays_missing")
        else:
            try:
                pred, true = np.load(pred_path), np.load(true_path)
                expected_shape = (int(job["pred_len"]), int(c_out))
                if (pred.shape != true.shape or pred.ndim != 3 or pred.shape[0] == 0
                        or tuple(pred.shape[1:]) != expected_shape
                        or not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true))):
                    errors.append("prediction_arrays_shape_invalid")
                elif not expected_skip_test:
                    expected_metrics = np.asarray(np.load(result_dir / "metrics.npy"), dtype=float).reshape(-1)
                    residual = pred.astype(float) - true.astype(float)
                    recomputed = np.asarray([
                        np.mean(np.abs(residual)), np.mean(residual ** 2),
                        np.sqrt(np.mean(residual ** 2)),
                    ])
                    if expected_metrics.size < 3 or not np.allclose(expected_metrics[:3], recomputed, rtol=1e-5, atol=1e-7):
                        errors.append("metrics_prediction_mismatch")
            except (OSError, ValueError, EOFError, ImportError):
                errors.append("prediction_arrays_invalid")
        if not original_metrics_path.is_file():
            errors.append("original_metrics_missing")
        else:
            try:
                original_metrics = np.asarray(np.load(original_metrics_path), dtype=float).reshape(-1)
                if original_metrics.size < 3 or not np.all(np.isfinite(original_metrics[:3])):
                    errors.append("original_metrics_invalid")
            except (OSError, ValueError, EOFError, ImportError):
                errors.append("original_metrics_invalid")
        if not channel_metrics_path.is_file():
            errors.append("channel_metrics_missing")
        else:
            try:
                channel_metrics = np.asarray(np.load(channel_metrics_path), dtype=float)
                if (channel_metrics.shape != (int(c_out), 6)
                        or not np.all(np.isfinite(channel_metrics))):
                    errors.append("channel_metrics_invalid")
            except (OSError, ValueError, EOFError, ImportError):
                errors.append("channel_metrics_invalid")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/final_training_plan.json"))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection_file", type=Path, default=None)
    parser.add_argument("--selection_provenance", type=Path, default=None,
                        help="verified selection provenance sidecar for selected-lambda phases")
    parser.add_argument("--provenance", type=Path, default=None,
                        help="immutable phase provenance.json; required for strict validation")
    parser.add_argument("--allow_legacy_unknown", action="store_true",
                        help="allow pre-provenance manifests, marked invalid for publication")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    plan = _load_json(plan_path)
    phase = next((item for item in plan["phases"] if item["name"] == args.phase), None)
    if phase is None:
        raise SystemExit(f"unknown phase: {args.phase}")
    selection_path = args.selection_file
    if selection_path is not None and not selection_path.is_absolute():
        selection_path = REPO_ROOT / selection_path
    selection_provenance_path = args.selection_provenance
    if selection_provenance_path is not None and not selection_provenance_path.is_absolute():
        selection_provenance_path = REPO_ROOT / selection_provenance_path
    try:
        selection_binding = _validate_selection_binding(
            selection_path, phase, plan, plan_path, selection_provenance_path
        )
    except ValueError as exc:
        raise SystemExit(f"selection binding invalid: {exc}") from exc
    selection = _selected_lambdas(selection_path) if selection_path else {}
    jobs = _job_specs(plan, phase, selection, allow_unresolved=False)
    # Presence of a confirmation contract is itself the strict boundary.  A
    # partially specified contract must fail its launch-binding check rather
    # than silently falling back to exploratory validation.
    strict_confirmation = isinstance(plan.get("confirmation_contract"), dict)
    # Exploratory plans may opt into the same train-only sidecar contract as a
    # confirmation plan without acquiring confirmation status.  This closes
    # the common gap where a live mechanism screen persists a prior but the
    # validator silently treats it as optional.
    require_acf_sidecar = plan.get("require_acf_sidecar", False)
    if not isinstance(require_acf_sidecar, bool):
        raise SystemExit("plan require_acf_sidecar must be boolean when present")
    if strict_confirmation and not isinstance(plan.get("statistical_gate"), dict):
        raise SystemExit("confirmation plan requires a statistical_gate object")
    expected = {_coordinate(job): job for job in jobs}
    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    records, malformed = _read_manifest(manifest_path)
    actual_by_coord = defaultdict(list)
    record_errors = {}
    artifact_hashes_by_job = {}
    for record in records.values():
        actual_by_coord[_coordinate(record)].append(record)
    errors = []
    warnings = []
    for coordinate, job in expected.items():
        matches = actual_by_coord.get(coordinate, [])
        if len(matches) != 1:
            errors.append({"coordinate": coordinate, "reason": f"manifest_matches={len(matches)}"})
            continue
        item_errors = _validate_record(
            matches[0],
            job,
            plan=plan,
            strict_memory=strict_confirmation,
            require_acf_sidecar=require_acf_sidecar,
        )
        phase_root = (manifest_path.parent.parent).resolve()
        artifact_path = Path(matches[0].get("expected_artifact", "")).resolve()
        if artifact_path and phase_root not in artifact_path.parents:
            item_errors.append("expected_artifact_outside_phase_root")
        checkpoint_root_value = _arg(matches[0].get("command", []), "--checkpoints")
        if checkpoint_root_value:
            checkpoint_root = Path(checkpoint_root_value).resolve()
            if phase_root not in checkpoint_root.parents:
                item_errors.append("checkpoint_root_outside_phase_root")
        if item_errors:
            record_errors[str(coordinate)] = item_errors
            errors.append({"coordinate": coordinate, "reason": item_errors})
        artifact = Path(matches[0].get("expected_artifact", ""))
        if artifact.is_file():
            result_dir = artifact.parent
            checkpoint_root = _arg(matches[0].get("command", []), "--checkpoints")
            checkpoint_dir = Path(checkpoint_root) / result_dir.name if checkpoint_root else None
            artifact_hashes_by_job[matches[0].get("job_id")] = _artifact_hashes(result_dir, checkpoint_dir)
    extra = [coordinate for coordinate in actual_by_coord if coordinate not in expected]
    if extra:
        errors.append({"extra_coordinates": extra[:10], "extra_count": len(extra)})
    phase_root = manifest_path.parent.parent
    provenance_path = args.provenance
    if provenance_path is None:
        provenance_path = phase_root / "provenance.json"
    elif not provenance_path.is_absolute():
        provenance_path = REPO_ROOT / provenance_path
    provenance = None
    expected_selection_hash = None
    expected_selection_provenance_hash = None
    if provenance_path.is_file():
        try:
            provenance = _load_json(provenance_path)
        except (OSError, ValueError):
            errors.append({"reason": "provenance_invalid"})
    elif not args.allow_legacy_unknown:
        errors.append({"reason": "provenance_missing"})
    if provenance is not None:
        if strict_confirmation:
            try:
                if provenance_path.resolve() != (phase_root / "provenance.json").resolve():
                    raise ValueError(
                        "confirmation provenance must be the phase-root provenance.json"
                    )
                _validate_confirmation_launch_binding(
                    plan, plan_path, provenance, phase_root
                )
            except (OSError, ValueError, TypeError) as exc:
                errors.append({
                    "reason": "confirmation_launch_binding_invalid",
                    "detail": str(exc),
                })
        snapshot = Path(provenance.get("source_snapshot", ""))
        for relative in (
            "scripts/run_training_plan.py",
            "scripts/validate_training_phase.py",
            "scripts/select_hyperparameters.py",
            "scripts/attest_selection.py",
            "scripts/audit_dataset_quality.py",
            "scripts/validate_highdim_contract.py",
        ):
            snapshot_file = snapshot / relative
            live_file = REPO_ROOT / relative
            if snapshot_file.is_file() and _sha256(snapshot_file) != _sha256(live_file):
                issue = {
                    "reason": f"live_{relative.replace('/', '_')}_differs_from_provenance_snapshot"
                }
                # Exploratory artifacts remain bound to the immutable runner
                # snapshot.  A later validator-only hardening must be visible
                # but should not invalidate otherwise complete artifacts;
                # confirmation plans retain the strict error boundary.
                if (
                    relative == "scripts/validate_training_phase.py"
                    and not strict_confirmation
                ):
                    warnings.append(issue)
                else:
                    errors.append(issue)
        if selection_path:
            if not selection_path.is_file():
                errors.append({"reason": "selection_file_missing"})
            else:
                expected_selection_hash = _sha256(selection_path)
        if selection_provenance_path:
            if not selection_provenance_path.is_file():
                errors.append({"reason": "selection_provenance_file_missing"})
            else:
                expected_selection_provenance_hash = _sha256(selection_provenance_path)
        provenance_checks = {
            "phase": args.phase,
            "plan_sha256": _sha256(plan_path),
            "selection_sha256": expected_selection_hash,
            "selection_provenance_sha256": expected_selection_provenance_hash,
        }
        for key, expected_value in provenance_checks.items():
            if provenance.get(key) != expected_value:
                errors.append({"reason": f"provenance_{key}_mismatch"})
        if selection_binding is not None:
            if provenance.get("selection_provenance") != selection_binding:
                errors.append({"reason": "provenance_selection_binding_mismatch"})
            if provenance.get("selection_binding_sha256") != _canonical_hash(selection_binding):
                errors.append({"reason": "provenance_selection_binding_sha256_mismatch"})
            selection_copy = Path(str(provenance.get("selection_copy", "")))
            sidecar_copy = Path(str(provenance.get("selection_provenance_copy", "")))
            if not selection_copy.is_file() or _sha256(selection_copy) != expected_selection_hash:
                errors.append({"reason": "selection_copy_mismatch"})
            if (not sidecar_copy.is_file()
                    or _sha256(sidecar_copy) != expected_selection_provenance_hash):
                errors.append({"reason": "selection_provenance_copy_mismatch"})
        if not snapshot.is_dir():
            errors.append({"reason": "source_snapshot_missing"})
        else:
            source_files = {}
            for item in sorted(snapshot.rglob("*")):
                if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}:
                    source_files[item.relative_to(snapshot).as_posix()] = _sha256(item)
            source_tree_hash = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if source_tree_hash != provenance.get("source_tree_sha256") or source_files != provenance.get("source_files"):
                errors.append({"reason": "source_snapshot_changed"})
        expected_data_hashes = {
            name: _sha256(REPO_ROOT / DATASETS[name]["root"] / DATASETS[name]["path"])
            for name in phase.get("datasets", [])
        }
        if provenance.get("dataset_hashes") != expected_data_hashes:
            errors.append({"reason": "dataset_hashes_mismatch"})
        for record in records.values():
            if record.get("provenance_id") != provenance.get("provenance_id"):
                errors.append({"reason": "record_provenance_id_mismatch", "job_id": record.get("job_id")})
            if record.get("source_tree_sha256_before") != provenance.get("source_tree_sha256"):
                errors.append({"reason": "record_source_before_mismatch", "job_id": record.get("job_id")})
            if record.get("source_tree_sha256_after") != provenance.get("source_tree_sha256"):
                errors.append({"reason": "record_source_after_mismatch", "job_id": record.get("job_id")})
            if not record.get("started_at_utc") or not record.get("finished_at_utc"):
                errors.append({"reason": "record_timestamps_missing", "job_id": record.get("job_id")})
    elif args.allow_legacy_unknown:
        errors.append({"reason": "legacy_unknown_provenance"})
    payload = {
        "valid": not errors and not malformed,
        "phase": args.phase,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "provenance": str(provenance_path),
        "provenance_id": provenance.get("provenance_id") if provenance else None,
        "selection_provenance": selection_binding,
        "selection_provenance_sha256": expected_selection_provenance_hash,
        "selection_binding_sha256": (
            _canonical_hash(selection_binding) if selection_binding is not None else None
        ),
        "legacy_unknown_provenance": provenance is None,
        "expected_jobs": len(expected),
        "manifest_unique_jobs": len(records),
        "malformed_manifest_lines": malformed,
        "status_counts": dict(Counter(record.get("status") for record in records.values())),
        "errors": errors,
        "warnings": warnings,
        "record_errors": record_errors,
        "artifact_hashes_by_job": artifact_hashes_by_job,
        "completed_coordinates": [list(coordinate) for coordinate in sorted(expected)],
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(json.dumps({"valid": payload["valid"], "expected_jobs": len(expected),
                      "manifest_unique_jobs": len(records), "errors": len(errors)}, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
