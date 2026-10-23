#!/usr/bin/env python3
"""Read-only readout for the four-arm forecast-relation-view screen.

The registered screen has four paired arms:

* encoder relation view, lambda=0.0 (null)
* encoder relation view, lambda=0.03 (active)
* forecast relation view, lambda=0.0 (null)
* forecast relation view, lambda=0.03 (active)

This reader consumes only completed validation artifacts.  It does not train,
run inference, or alter an artifact directory.  In particular, the screen is
registered with ``skip_test=true``; the primary metric therefore comes from
the pure forecasting ``val_loss`` trajectory in ``loss_logs.json`` rather
than from a test ``metrics.npy`` file.
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
except Exception:  # pragma: no cover - optional for a descriptive readout.
    scipy_stats = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.analyze_mamba_relation_null_control import (
        _hash_map_for_record,
        _manifest_records,
        _read_json,
        _resolve,
        _sha256,
        _summary_series,
    )
    from scripts.run_training_plan import _config_hash, _job_specs, _validate_plan
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from analyze_mamba_relation_null_control import (  # type: ignore
        _hash_map_for_record,
        _manifest_records,
        _read_json,
        _resolve,
        _sha256,
        _summary_series,
    )
    from run_training_plan import _config_hash, _job_specs, _validate_plan  # type: ignore


EXPECTED_PHASES: Tuple[Tuple[str, str, str, float], ...] = (
    ("encoder_relation_null_h96", "encoder", "null", 0.0),
    ("encoder_relation_h96", "encoder", "active", 0.03),
    ("forecast_relation_null_h96", "forecast", "null", 0.0),
    ("forecast_relation_h96", "forecast", "active", 0.03),
)
PHASES = EXPECTED_PHASES
EXPECTED_DATASETS: Tuple[str, ...] = ("ETTh1", "traffic")
EXPECTED_SEEDS: Tuple[int, ...] = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
EXPECTED_ALIAS = "MambaACFAutoCon"
EXPECTED_JOBS_PER_PHASE = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)
EXPECTED_JOBS = len(EXPECTED_PHASES) * EXPECTED_JOBS_PER_PHASE
EXPECTED_PLAN_STATUS = "exploratory_forecast_relation_view_screen_draft"
EXPECTED_RELATION_LOG_KEYS: Tuple[str, ...] = (
    "acf_relation_global_valid_fraction",
    "acf_relation_global_active_anchor_fraction",
    "acf_relation_global_candidate_fraction",
    "acf_relation_global_candidate_active_anchor_fraction",
    "acf_relation_global_candidate_count_mean",
    "acf_relation_global_contrastive_active_anchor_fraction",
    "acf_relation_global_degenerate_positive_anchor_fraction",
    "acf_relation_global_positive_count_mean",
    "acf_relation_global_high_acf_excluded_fraction",
    "acf_relation_global_high_acf_candidate_count",
    "acf_relation_global_high_acf_excluded_count",
    "acf_relation_global_high_acf_conditional_excluded_fraction",
    "acf_relation_global_sign_target_count",
    "acf_relation_global_sign_accuracy",
    "acf_relation_global_sign_hinge",
)

# These are the fields that should be identical within a dataset/seed pair.
# Paths, resolved lag lists, hashes, and the two treatment fields are allowed
# to vary per job and are checked separately.
COMMON_CONFIG_KEYS: Tuple[str, ...] = (
    "model",
    "data",
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
    "dropout",
    "embed",
    "activation",
    "AutoCon",
    "AutoCon_acf_source",
    "AutoCon_warmup_epochs",
    "mamba_use_acf_memory",
    "mamba_dynamic_acf",
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
    "mamba_variable_mixer",
    "mamba_acf_memory_gate_mode",
    "mamba_causal_embedding",
    "mamba_bidirectional",
    "mamba_channel_independent",
    "mamba_acf_repr_fusion",
    "mamba_variable_repr_fusion",
    "mamba_patch_mixer",
    "mamba_local_mixer",
    "mamba_low_rank_head",
    "skip_test",
    "stream_metrics",
)

TREATMENT_FIELDS = {
    "mamba_acf_relation_representation",
    "AutoCon_lambda",
    "allow_zero_autocon_lambda",
}


class NotRunError(RuntimeError):
    """Raised when a registered coordinate has no complete result."""


def _canonical(value: Any) -> Any:
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


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _phase(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    for item in phases:
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    raise ValueError(f"plan has no phase {name!r}")


def _phase_root(plan: Mapping[str, Any], name: str) -> Path:
    phase = _phase(plan, name)
    output_root = phase.get("output_root")
    if not output_root:
        raise ValueError(f"phase {name} has no output_root")
    return _resolve(str(output_root))


def _plan_errors(plan: Mapping[str, Any]) -> List[str]:
    """Return deterministic errors for the frozen four-arm plan contract."""

    errors: List[str] = []
    try:
        errors.extend(str(item) for item in _validate_plan(dict(plan)))
    except Exception as exc:  # pragma: no cover - defensive contract boundary.
        errors.append(f"generic plan validation raised: {exc}")

    if plan.get("status") != EXPECTED_PLAN_STATUS:
        errors.append(f"unexpected plan status: {plan.get('status')!r}")
    if plan.get("require_acf_sidecar") is not True:
        errors.append("plan require_acf_sidecar must be true")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("plan execution metadata is missing")
    else:
        if execution.get("authorized") is not False:
            errors.append("draft plan must have execution.authorized=false")
        if execution.get("training_started") is not False:
            errors.append("draft plan must have execution.training_started=false")

    phases = plan.get("phases")
    expected_names = [item[0] for item in EXPECTED_PHASES]
    if not isinstance(phases, list) or [item.get("name") for item in phases if isinstance(item, Mapping)] != expected_names:
        errors.append(f"phase names/order mismatch: expected {expected_names!r}")
    else:
        roots = set()
        for phase_item, (name, route, arm, lambda_value) in zip(phases, EXPECTED_PHASES):
            if not isinstance(phase_item, Mapping):
                errors.append(f"phase {name} is not an object")
                continue
            if phase_item.get("models") != [EXPECTED_ALIAS]:
                errors.append(f"{name} must use {EXPECTED_ALIAS}")
            if phase_item.get("datasets") != list(EXPECTED_DATASETS):
                errors.append(f"{name} dataset contract mismatch")
            if phase_item.get("horizons") != [EXPECTED_HORIZON]:
                errors.append(f"{name} horizon contract mismatch")
            if phase_item.get("seeds") != list(EXPECTED_SEEDS):
                errors.append(f"{name} seed contract mismatch")
            if phase_item.get("features", "M") != "M":
                errors.append(f"{name} features must be M")
            if str(phase_item.get("mamba_acf_relation_representation")) != route:
                errors.append(f"{name} relation representation mismatch")
            try:
                actual_lambda = float(phase_item.get("autocon_lambda"))
                if not math.isclose(actual_lambda, lambda_value, rel_tol=1e-9, abs_tol=1e-12):
                    errors.append(f"{name} autocon_lambda mismatch")
            except (TypeError, ValueError):
                errors.append(f"{name} autocon_lambda is not numeric")
            try:
                allow_zero = _explicit_bool(phase_item.get("allow_zero_autocon_lambda", False), f"{name} allow_zero_autocon_lambda")
                if allow_zero != (arm == "null"):
                    errors.append(f"{name} allow_zero_autocon_lambda mismatch")
            except ValueError as exc:
                errors.append(str(exc))
            # Relation flags are frozen in ``global`` for this plan.  Only
            # the representation route and lambda are phase-level treatments.
            if "mamba_acf_relation_representation" in phase_item and str(
                phase_item.get("mamba_acf_relation_representation")
            ) != route:
                errors.append(f"{name} relation representation mismatch")
            root = phase_item.get("output_root")
            if root:
                roots.add(str(root))
        if len(roots) != len(EXPECTED_PHASES):
            errors.append("phase output roots must be distinct")

    counts = plan.get("registered_job_counts")
    expected_counts = {name: EXPECTED_JOBS_PER_PHASE for name, *_ in EXPECTED_PHASES}
    expected_counts.update({"training_total": EXPECTED_JOBS, "exploratory_total": EXPECTED_JOBS, "training_started": False})
    if not isinstance(counts, Mapping):
        errors.append("registered_job_counts is missing")
    else:
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                errors.append(f"registered_job_counts[{key}] mismatch")

    global_cfg = plan.get("global")
    if not isinstance(global_cfg, Mapping):
        errors.append("global configuration is missing")
    else:
        for key, expected in {
            "features": "M",
            "seq_len": 96,
            "label_len": 48,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_representation": "encoder",
            "skip_test": True,
            "stream_metrics": False,
        }.items():
            actual = global_cfg.get(key)
            if isinstance(expected, bool):
                try:
                    actual = _explicit_bool(actual, f"global {key}")
                except ValueError:
                    pass
            if actual != expected:
                errors.append(f"global {key} mismatch")

    relation_contract = plan.get("relation_representation_contract")
    if not isinstance(relation_contract, Mapping):
        errors.append("relation_representation_contract is missing")
    else:
        expected_contract = {
            "encoder_null_phase": "encoder_relation_null_h96",
            "encoder_active_phase": "encoder_relation_h96",
            "forecast_null_phase": "forecast_relation_null_h96",
            "forecast_active_phase": "forecast_relation_h96",
        }
        for key, expected in expected_contract.items():
            if relation_contract.get(key) != expected:
                errors.append(f"relation contract {key} mismatch")
    return errors


def validate_plan_contract(plan: Mapping[str, Any]) -> None:
    errors = _plan_errors(plan)
    if errors:
        raise ValueError("forecast-relation screen plan validation failed: " + "; ".join(errors))


def _missing_training_artifacts(plan: Mapping[str, Any]) -> List[str]:
    """Return a short list of missing completion markers, without writes."""

    missing: List[str] = []
    for name, _, _, _ in EXPECTED_PHASES:
        try:
            root = _phase_root(plan, name)
        except (KeyError, TypeError, ValueError) as exc:
            missing.append(f"{name}: {exc}")
            continue
        validation_path = root / "phase_validation.json"
        manifest_path = root / "logs" / "job_manifest.jsonl"
        if not validation_path.is_file():
            missing.append(str(validation_path))
            continue
        try:
            validation = _read_json(validation_path)
        except (OSError, ValueError, TypeError):
            missing.append(str(validation_path) + " (unreadable)")
            continue
        if validation.get("valid") is not True:
            missing.append(str(validation_path) + " (valid=false)")
        if not manifest_path.is_file():
            missing.append(str(manifest_path))
            continue
        try:
            records = _manifest_records(root, validation)
            if len(records) != EXPECTED_JOBS_PER_PHASE:
                missing.append(f"{manifest_path} (incomplete: {len(records)}/{EXPECTED_JOBS_PER_PHASE})")
            if any(record.get("status") != "ok" for record in records):
                missing.append(f"{manifest_path} (not all jobs succeeded)")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # A missing/partial manifest is a not-run state; hash drift is
            # handled later as an integrity error once all files exist.
            text = str(exc).lower()
            if "missing" in text or "empty" in text or "coverage" in text or "status" in text:
                missing.append(f"{manifest_path} ({exc})")
    return missing


def _required_config(config: Mapping[str, Any], route: str, lambda_value: float, dataset: str, seed: int, job_id: str) -> Dict[str, Any]:
    channels = {"ETTh1": 7, "traffic": 862}[dataset]
    expected: Dict[str, Any] = {
        "model": EXPECTED_ALIAS,
        "data": dataset,
        "features": "M",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": EXPECTED_HORIZON,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "AutoCon": True,
        "AutoCon_lambda": lambda_value,
        "AutoCon_acf_source": "raw",
        "mamba_use_acf_memory": True,
        "mamba_dynamic_acf": False,
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_representation": route,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_relation_lag_mask": True,
        "skip_test": True,
        "stream_metrics": False,
    }
    for key, wanted in expected.items():
        if key not in config:
            raise ValueError(f"config {key} is missing for {job_id}")
        actual = config[key]
        if isinstance(wanted, bool):
            actual = _explicit_bool(actual, f"config {key} for {job_id}")
        elif isinstance(wanted, int) and not isinstance(wanted, bool):
            actual = _int(actual, f"config {key} for {job_id}")
        elif isinstance(wanted, float):
            actual = _finite(actual, f"config {key} for {job_id}")
            if not math.isclose(actual, wanted, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"config {key} mismatch for {job_id}")
            continue
        if actual != wanted:
            raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {wanted!r}")

    # Forecast relation routing is defined only for a multivariate window.
    if route == "forecast" and channels < 2:
        raise ValueError(f"forecast relation view requires at least two channels for {job_id}")
    signature: Dict[str, Any] = {}
    for key in COMMON_CONFIG_KEYS:
        if key not in config:
            # Some older run snapshots do not persist optional fields.  They
            # cannot be used for a paired readout if a declared common field
            # is absent, because equality would be ambiguous.
            raise ValueError(f"common config field {key} is missing for {job_id}")
        signature[key] = _canonical(config[key])
    return signature


def _command_value(command: Any, name: str) -> Optional[str]:
    if not isinstance(command, list):
        return None
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _validation_metrics(logs: Mapping[str, Any], path: Path, job_id: str) -> Dict[str, Any]:
    """Extract best pure-forecast validation metrics from the loss log."""

    def series(candidates: Sequence[str], required: bool) -> Tuple[Optional[List[float]], Optional[str]]:
        for key in candidates:
            if key not in logs:
                continue
            raw = logs[key]
            values = raw if isinstance(raw, list) else [raw]
            if not values:
                raise NotRunError(f"empty validation trajectory {key}: {path}")
            numbers = [_finite(item, f"{job_id} {key}") for item in values]
            return numbers, key
        if required:
            raise NotRunError(f"missing validation trajectory ({'/'.join(candidates)}): {path}")
        return None, None

    mse_values, mse_key = series(("val_mse", "validation_mse", "val_loss"), True)
    assert mse_values is not None
    result: Dict[str, Any] = {
        "mse": min(mse_values),
        "mse_source": mse_key,
        "mse_trajectory": {
            "count": len(mse_values),
            "first": mse_values[0],
            "last": mse_values[-1],
            "best": min(mse_values),
        },
    }
    mae_values, mae_key = series(("val_mae", "validation_mae"), False)
    if mae_values is not None:
        result.update({
            "mae": min(mae_values),
            "mae_source": mae_key,
            "mae_trajectory": {
                "count": len(mae_values),
                "first": mae_values[0],
                "last": mae_values[-1],
                "best": min(mae_values),
            },
        })
    return result


def _validate_phase_artifact_bindings(
    validation: Mapping[str, Any],
    manifest_path: Path,
    phase_name: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Bind the phase validation payload to the exact manifest/hash matrix."""

    declared_manifest = validation.get("manifest")
    if not declared_manifest:
        raise ValueError(f"phase validation manifest path is missing: {phase_name}")
    if _resolve(str(declared_manifest)) != manifest_path.resolve():
        raise ValueError(f"phase validation points to another manifest: {phase_name}")
    record_ids = [str(record.get("job_id", "")) for record in records]
    if any(not item for item in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError(f"manifest job IDs are missing/duplicated: {phase_name}")
    validation_hashes = validation.get("artifact_hashes_by_job")
    if not isinstance(validation_hashes, Mapping):
        raise ValueError(f"phase artifact hash map is missing: {phase_name}")
    normalized_hash_keys = {str(key) for key in validation_hashes}
    if normalized_hash_keys != set(record_ids):
        raise ValueError(f"phase artifact hash map does not cover manifest: {phase_name}")
    for record, job_id in zip(records, record_ids):
        record_hashes = record.get("artifact_hashes")
        phase_hashes = validation_hashes.get(job_id)
        if phase_hashes is None and job_id not in validation_hashes:
            # JSON object keys are strings, but accepting a non-string mapping
            # key here would make the binding ambiguous.
            raise ValueError(f"phase artifact hash entry is missing: {job_id}")
        if not isinstance(record_hashes, Mapping) or not isinstance(phase_hashes, Mapping):
            raise ValueError(f"artifact hash map is invalid: {job_id}")
        if dict(record_hashes) != dict(phase_hashes):
            raise ValueError(f"manifest/validation artifact hashes differ for {job_id}")


def _validate_record_config_hash(
    record: Mapping[str, Any], registered_job: Mapping[str, Any], job_id: str
) -> None:
    """Require the runner's registered configuration hash on every record."""

    actual = record.get("config_hash")
    if not isinstance(actual, str) or not actual:
        raise ValueError(f"manifest config hash is missing: {job_id}")
    expected = _config_hash(dict(registered_job), record.get("provenance_id"))
    if actual != expected:
        raise ValueError(f"manifest config hash mismatch: {job_id}")


def _load_phase(plan: Mapping[str, Any], plan_path: Path, phase_name: str, route: str, arm: str, lambda_value: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase_root = _phase_root(plan, phase_name)
    validation_path = phase_root / "phase_validation.json"
    if not validation_path.is_file():
        raise NotRunError(f"missing phase validation: {validation_path}")
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise NotRunError(f"phase validation is incomplete: {validation_path}")
    if validation.get("plan_sha256") and validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError(f"phase validation plan hash mismatch: {phase_name}")
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file():
        raise NotRunError(f"missing manifest: {manifest_path}")
    records = _manifest_records(phase_root, validation)
    if len(records) != EXPECTED_JOBS_PER_PHASE:
        raise NotRunError(f"incomplete manifest for {phase_name}: {len(records)}/{EXPECTED_JOBS_PER_PHASE}")
    _validate_phase_artifact_bindings(validation, manifest_path, phase_name, records)

    phase_decl = _phase(plan, phase_name)
    registered = _job_specs(plan, phase_decl, {}, allow_unresolved=True)
    registered_by_coord = {
        (str(item["dataset"]), int(item["pred_len"]), int(item["seed"])): item
        for item in registered
    }
    if len(registered_by_coord) != EXPECTED_JOBS_PER_PHASE:
        raise ValueError(f"registered coordinate expansion is incomplete: {phase_name}")

    result_root = (phase_root / "results").resolve()
    default_checkpoint_root = (phase_root / "checkpoints").resolve()
    observed: set[Tuple[str, int]] = set()
    rows: List[Dict[str, Any]] = []
    for record in records:
        job_id = str(record.get("job_id", ""))
        if record.get("phase") != phase_name:
            raise ValueError(f"manifest phase mismatch: {job_id}")
        if record.get("status") != "ok":
            raise NotRunError(f"job is incomplete: {job_id}")
        dataset = str(record.get("dataset"))
        seed = _int(record.get("seed"), f"{job_id} seed")
        horizon = _int(record.get("pred_len"), f"{job_id} pred_len")
        coordinate = (dataset, seed)
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS or horizon != EXPECTED_HORIZON:
            raise ValueError(f"unexpected coordinate: {job_id}")
        if coordinate in observed:
            raise ValueError(f"duplicate coordinate: {phase_name} {coordinate}")
        observed.add(coordinate)
        registered_job = registered_by_coord.get((dataset, horizon, seed))
        if registered_job is None:
            raise ValueError(f"unregistered coordinate: {job_id}")
        if record.get("model") != EXPECTED_ALIAS:
            raise ValueError(f"manifest model mismatch: {job_id}")
        _validate_record_config_hash(record, registered_job, job_id)
        record_lambda = record.get("autocon_lambda")
        if record_lambda is None:
            expected_artifact_for_lambda = record.get("expected_artifact")
            if expected_artifact_for_lambda:
                config_for_lambda = _resolve(str(expected_artifact_for_lambda)).parent / "config.json"
                if config_for_lambda.is_file():
                    record_lambda = _read_json(config_for_lambda).get("AutoCon_lambda")
        if not math.isclose(_finite(record_lambda, f"{job_id} lambda"), lambda_value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"manifest lambda mismatch: {job_id}")

        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise NotRunError(f"missing expected artifact: {job_id}")
        marker_path = _resolve(str(expected_artifact))
        if marker_path.name != "training_complete.json" or not marker_path.is_file() or not _inside(marker_path, result_root):
            raise NotRunError(f"missing training completion marker: {job_id}")
        marker = _read_json(marker_path)
        if marker.get("status") != "trained_without_test":
            raise NotRunError(f"training completion marker is incomplete: {job_id}")
        result_dir = marker_path.parent

        checkpoint_root = default_checkpoint_root
        command_checkpoint = _command_value(record.get("command"), "--checkpoints")
        if command_checkpoint:
            checkpoint_root = _resolve(command_checkpoint)
        checkpoint_dir = checkpoint_root / result_dir.name
        required_files = {
            "config.json": result_dir / "config.json",
            "training_complete.json": result_dir / "training_complete.json",
            "acf_prior.npz": result_dir / "acf_prior.npz",
            "checkpoint.pth": checkpoint_dir / "checkpoint.pth",
            "loss_logs.json": checkpoint_dir / "loss_logs.json",
        }
        absent = [str(path) for path in required_files.values() if not path.is_file()]
        if absent:
            raise NotRunError(f"incomplete artifacts for {job_id}: {', '.join(absent)}")
        try:
            hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        except ValueError as exc:
            if "missing artifact" in str(exc).lower():
                raise NotRunError(str(exc)) from exc
            raise
        for name in required_files:
            if name not in hashes:
                raise ValueError(f"manifest hash map is missing {name}: {job_id}")

        config = _read_json(required_files["config.json"])
        signature = _required_config(config, route, lambda_value, dataset, seed, job_id)
        prior_hash = str(config.get("mamba_acf_prior_sha256", "")).lower()
        if prior_hash and prior_hash != hashes["acf_prior.npz"].lower():
            raise ValueError(f"config/sidecar hash mismatch: {job_id}")

        logs_path = required_files["loss_logs.json"]
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {job_id}")
        if str(logs.get("acf_relation_target_mode")) != "hard_selected":
            raise ValueError(f"relation target mode mismatch: {job_id}")
        relation = {key: _summary_series(logs, key, logs_path) for key in EXPECTED_RELATION_LOG_KEYS}
        autocon_loss = _summary_series(logs, "autocon_loss", logs_path)
        effective_lambda = _summary_series(logs, "effective_lambda", logs_path)
        if not math.isclose(effective_lambda["last"], lambda_value, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"effective lambda mismatch: {job_id}")
        metrics = _validation_metrics(logs, logs_path, job_id)
        rows.append({
            "phase": phase_name,
            "route": route,
            "arm": arm,
            "dataset": dataset,
            "seed": seed,
            "horizon": horizon,
            "mse": metrics["mse"],
            "mae": metrics.get("mae"),
            "mse_source": metrics["mse_source"],
            "mae_source": metrics.get("mae_source"),
            "mse_trajectory": metrics["mse_trajectory"],
            "mae_trajectory": metrics.get("mae_trajectory"),
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "epochs_ran": _int(record.get("epochs_ran", logs.get("epochs_ran", len(logs.get("val_loss", [])))), f"{job_id} epochs_ran"),
            "best_epoch": _int(record.get("best_epoch", logs.get("best_epoch", 0)), f"{job_id} best_epoch"),
            "job_id": job_id,
            "sidecar_sha256": hashes["acf_prior.npz"],
            "config_signature": signature,
            "relation_diagnostics": relation,
            "autocon_loss": autocon_loss,
            "effective_lambda": effective_lambda,
            "artifact_hashes": hashes,
        })
    expected_coordinates = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    if observed != expected_coordinates:
        raise NotRunError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows, {
        "phase": phase_name,
        "route": route,
        "arm": arm,
        "root": str(phase_root),
        "validation": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "provenance_id": validation.get("provenance_id"),
        "source_tree_sha256": (
            _read_json(phase_root / "provenance.json").get("source_tree_sha256")
            if (phase_root / "provenance.json").is_file()
            else None
        ),
        "expected_jobs": EXPECTED_JOBS_PER_PHASE,
        "observed_jobs": len(rows),
        "valid": True,
    }


def _paired_summary(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [_finite(value, "paired delta") for value in values]
    if not numbers:
        raise ValueError("empty paired delta vector")
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    t_stat: float
    p_value: Optional[float] = None
    if scipy_stats is not None and len(numbers) > 1:
        test = scipy_stats.ttest_1samp(numbers, 0.0)
        t_stat, p_value = float(test.statistic), float(test.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat = mean / (sd / math.sqrt(len(numbers)))
    elif mean == 0:
        t_stat = 0.0
    else:
        t_stat = math.copysign(math.inf, mean)
    # Keep the emitted readout strict JSON even when a constant paired vector
    # makes scipy return a non-finite statistic.
    if not math.isfinite(t_stat):
        t_stat = None  # type: ignore[assignment]
    if p_value is not None and not math.isfinite(p_value):
        p_value = None
    if not math.isfinite(sd):
        sd = None  # type: ignore[assignment]
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
        "values": numbers,
    }


def _relative(delta: float, reference: float) -> Optional[float]:
    # A zero reference has no meaningful relative change; use JSON null
    # instead of emitting a non-standard NaN token.
    return delta / reference if reference else None


def _metric_summary(pairs: Sequence[Mapping[str, Any]], key: str, reference_key: Optional[str] = None, treatment_key: Optional[str] = None) -> Dict[str, Any]:
    values = [float(item[key]) for item in pairs]
    result = _paired_summary(values)
    if reference_key is not None and treatment_key is not None:
        reference = [float(item[reference_key]) for item in pairs]
        treatment = [float(item[treatment_key]) for item in pairs]
        reference_mean = statistics.fmean(reference)
        result.update({
            "reference": reference_key,
            "treatment": treatment_key,
            "reference_mean": reference_mean,
            "treatment_mean": statistics.fmean(treatment),
            "relative_delta_mean": _relative(result["mean"], reference_mean),
        })
    return result


def _make_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    expected_labels = {"encoder_null", "encoder_active", "forecast_null", "forecast_active"}
    indexed: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset")), int(row.get("seed")), str(row.get("route")) + "_" + str(row.get("arm")))
        if key in indexed:
            raise ValueError(f"duplicate row coordinate: {key}")
        indexed[key] = row
    expected_coordinates = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    observed_coordinates = {(str(row.get("dataset")), int(row.get("seed"))) for row in rows}
    if len(rows) != EXPECTED_JOBS or observed_coordinates != expected_coordinates:
        raise ValueError("row matrix does not contain exactly the registered 24 rows")
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            selected: Dict[str, Mapping[str, Any]] = {}
            for label in expected_labels:
                route, arm = label.split("_")
                row = indexed.get((dataset, seed, label))
                if row is None:
                    raise ValueError(f"incomplete paired coordinate: {dataset} seed {seed} {label}")
                selected[label] = row
            sidecars = {str(row.get("sidecar_sha256")) for row in selected.values()}
            if len(sidecars) != 1:
                raise ValueError(f"sidecar hash differs at {dataset} seed {seed}")
            signatures = {_canonical(row.get("config_signature")) for row in selected.values()}
            if len(signatures) != 1:
                raise ValueError(f"common config differs at {dataset} seed {seed}")
            pair: Dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "horizon": EXPECTED_HORIZON,
                "sidecar_sha256": next(iter(sidecars)),
            }
            for label, row in selected.items():
                pair[f"{label}_mse"] = float(row["mse"])
                pair[f"{label}_mae"] = row.get("mae")
                pair[f"{label}_elapsed_sec"] = float(row.get("elapsed_sec", math.nan))
                pair[f"{label}_job_id"] = row.get("job_id")
            for metric in ("mse", "mae"):
                values = [selected[label].get(metric) for label in expected_labels]
                if all(value is not None for value in values):
                    pair[f"encoder_active_minus_null_{metric}"] = pair[f"encoder_active_{metric}"] - pair[f"encoder_null_{metric}"]
                    pair[f"forecast_active_minus_null_{metric}"] = pair[f"forecast_active_{metric}"] - pair[f"forecast_null_{metric}"]
                    pair[f"difference_in_differences_{metric}"] = pair[f"forecast_active_minus_null_{metric}"] - pair[f"encoder_active_minus_null_{metric}"]
                else:
                    pair[f"encoder_active_minus_null_{metric}"] = None
                    pair[f"forecast_active_minus_null_{metric}"] = None
                    pair[f"difference_in_differences_{metric}"] = None
            pairs.append(pair)
    return pairs


def _contrast_block(pairs: Sequence[Mapping[str, Any]], prefix: str, metric: str, reference_key: Optional[str] = None, treatment_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    key = f"{prefix}_{metric}"
    if any(item.get(key) is None for item in pairs):
        return None
    return _metric_summary(pairs, key, reference_key, treatment_key)


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], validation: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    normalized = [dict(row) for row in rows]
    pairs = _make_pairs(normalized)
    routes = {
        "encoder_active_minus_null": ("encoder_active_minus_null", "encoder_null_mse", "encoder_active_mse"),
        "forecast_active_minus_null": ("forecast_active_minus_null", "forecast_null_mse", "forecast_active_mse"),
        "difference_in_differences": ("difference_in_differences", None, None),
    }
    contrasts: Dict[str, Any] = {}
    for name, (prefix, reference, treatment) in routes.items():
        block: Dict[str, Any] = {}
        for metric in ("mse", "mae"):
            summary = _contrast_block(
                pairs,
                prefix,
                metric,
                reference_key=reference,
                treatment_key=treatment,
            )
            if summary is not None:
                block[metric] = summary
        block["by_dataset"] = {}
        for dataset in EXPECTED_DATASETS:
            subset = [pair for pair in pairs if pair["dataset"] == dataset]
            block["by_dataset"][dataset] = {}
            for metric in ("mse", "mae"):
                summary = _contrast_block(subset, prefix, metric, reference, treatment)
                if summary is not None:
                    block["by_dataset"][dataset][metric] = summary
        contrasts[name] = block

    diagnostics: Dict[str, Any] = {}
    for route, arm in (("encoder", "null"), ("encoder", "active"), ("forecast", "null"), ("forecast", "active")):
        label = f"{route}_{arm}"
        subset = [row for row in normalized if row.get("route") == route and row.get("arm") == arm]
        diagnostics[label] = {
            "n": len(subset),
            "autocon_loss_mean_over_runs": statistics.fmean(row["autocon_loss"]["mean"] for row in subset),
            "effective_lambda_last_values": [row["effective_lambda"]["last"] for row in subset],
            "relation": {
                key: {"mean_over_runs": statistics.fmean(row["relation_diagnostics"][key]["mean"] for row in subset)}
                for key in EXPECTED_RELATION_LOG_KEYS
            },
        }
    plan_path = Path(str(plan.get("_path", ""))).resolve()
    return {
        "schema_version": 1,
        "status": "exploratory_forecast_relation_view_screen_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "phases": [name for name, _, _, _ in EXPECTED_PHASES],
        "rows": normalized,
        "pairs": pairs,
        "validation": list(validation or []),
        "contrasts": contrasts,
        "estimands": {
            "encoder_active_minus_null": contrasts["encoder_active_minus_null"],
            "forecast_active_minus_null": contrasts["forecast_active_minus_null"],
            "difference_in_differences": contrasts["difference_in_differences"],
        },
        "diagnostics": diagnostics,
        "pairing": {
            "unit": ["dataset", "horizon", "seed"],
            "coordinates": len(pairs),
            "rows": len(normalized),
            "sidecar_hashes_equal_per_coordinate": True,
            "common_config_equal_per_coordinate": True,
        },
        "evidence_boundary": "Validation-only paired exploratory evidence; no test metrics, confirmatory inference, speed claim, or publication claim.",
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
    lines = [
        "# MambaACF Forecast Relation-View Screen",
        "",
        "> Exploratory, validation-only, non-confirmatory readout. The four arms are paired by dataset, H96, and seed.",
        "",
        f"- Completed rows: `{len(result.get('rows', []))}/{EXPECTED_JOBS}`; paired coordinates: `{len(result.get('pairs', []))}/{len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)}`.",
        "- The primary metric is the best pure-forecast `val_loss` (reported as validation MSE) from `loss_logs.json`; no test metrics are read.",
        "",
        "## Per-coordinate estimands",
        "",
        "| Dataset | Seed | Encoder null | Encoder active | Encoder active-null | Forecast null | Forecast active | Forecast active-null | Difference-in-differences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in result.get("pairs", []):
        lines.append(
            f"| {pair['dataset']} | {pair['seed']} | {_fmt(pair['encoder_null_mse'])} | {_fmt(pair['encoder_active_mse'])} | {_fmt(pair['encoder_active_minus_null_mse'])} | {_fmt(pair['forecast_null_mse'])} | {_fmt(pair['forecast_active_mse'])} | {_fmt(pair['forecast_active_minus_null_mse'])} | {_fmt(pair['difference_in_differences_mse'])} |"
        )
    lines += [
        "",
        "Negative active-null deltas favor adding the relation objective within that representation view. A negative difference-in-differences favors the forecast route relative to the encoder route; all values are descriptive.",
        "",
        "## Paired summaries",
        "",
        "| Estimand | n | Mean delta | SD | Wins lower | Two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("encoder_active_minus_null", "Encoder active - null"),
        ("forecast_active_minus_null", "Forecast active - null"),
        ("difference_in_differences", "Forecast contrast - encoder contrast"),
    ):
        summary = result["contrasts"][name].get("mse")
        if summary is None:
            continue
        p = summary.get("two_sided_p")
        lines.append(f"| {label} | {summary['n']} | {_fmt(summary['mean'])} | {_fmt(summary['sd'])} | {summary['wins_lower']} | {'n/a' if p is None else _fmt(p, 4)} |")
    lines += [
        "",
        "## Relation diagnostics",
        "",
        "| Arm | n | AutoCon loss | Candidate count | Contrastive-active anchors | Degenerate positives | Sign accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("encoder_null", "encoder_active", "forecast_null", "forecast_active"):
        diag = result["diagnostics"].get(label, {})
        relation = diag.get("relation", {})
        lines.append(
            f"| {label} | {diag.get('n', 0)} | {_fmt(diag.get('autocon_loss_mean_over_runs'))} | {_fmt(relation.get('acf_relation_global_candidate_count_mean', {}).get('mean_over_runs'))} | {_fmt(relation.get('acf_relation_global_contrastive_active_anchor_fraction', {}).get('mean_over_runs'))} | {_fmt(relation.get('acf_relation_global_degenerate_positive_anchor_fraction', {}).get('mean_over_runs'))} | {_fmt(relation.get('acf_relation_global_sign_accuracy', {}).get('mean_over_runs'))} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- This readout uses validation trajectories only; it does not produce or inspect test metrics.",
        "- The forecast relation view is a window-level cross-channel objective because the current global loss averages the forecast steps; it is not a stepwise temporal relation objective.",
        "- Zero-lambda arms still calculate the relation path for telemetry and are not an auxiliary-path throughput control.",
        "- A complete screen remains exploratory and cannot select a route or support a publication claim without a separately reviewed frozen replication.",
    ]
    return "\n".join(lines) + "\n"


def _not_run_message(plan: Mapping[str, Any]) -> str:
    missing = _missing_training_artifacts(plan)
    if not missing:
        return "NOT_RUN: no complete training artifacts found for the registered forecast-relation screen"
    preview = "; ".join(missing[:6])
    suffix = " ..." if len(missing) > 6 else ""
    return f"NOT_RUN: registered forecast-relation screen is incomplete ({preview}{suffix})"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/research/mamba_acf_forecast_relation_view_screen_plan_20260903.json"),
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
        for phase_name, route, arm, lambda_value in EXPECTED_PHASES:
            phase_rows, phase_meta = _load_phase(plan, plan_path, phase_name, route, arm, lambda_value)
            rows.extend(phase_rows)
            validation.append(phase_meta)
        result = build_readout(plan, rows, validation)
        output_json = _resolve(
            args.output_json
            or Path("artifacts/exploratory/mamba_acf_forecast_relation_view_screen_20260903/readout.json")
        )
        output_md = _resolve(
            args.output_md
            or Path("docs/research/mamba_acf_forecast_relation_view_screen_readout_20260903.md")
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
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
        print(f"analyze_mamba_acf_forecast_relation_view_screen: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
