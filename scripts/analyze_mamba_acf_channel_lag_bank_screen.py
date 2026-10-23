#!/usr/bin/env python3
"""Strict, read-only readout for the channel-specific ACF lag-bank screen.

The registered forecast-level screen has two paired arms of the same
``MambaACFAutoCon`` model.  The only treatment is the detached
``mamba_acf_channel_lag_bank`` support mask.  This reader validates the frozen
runner plan, phase-validation payloads, immutable provenance, artifact hashes,
raw train-only ACF sidecars, and the derived support metadata before producing
descriptive paired statistics.  It never trains, evaluates, or edits an
artifact directory.  A partial matrix is reported as ``NOT_RUN`` (exit code
3), and no output files are created in that case.
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
except Exception:  # pragma: no cover - optional reporting dependency.
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
    from scripts.run_training_plan import (
        _artifact_hashes,
        _command_hash,
        _config_hash,
        _job_specs,
        _validate_plan,
    )
    from utils.acf_guidance import compile_channel_lag_support
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from analyze_mamba_relation_null_control import (  # type: ignore
        _hash_map_for_record,
        _manifest_records,
        _read_json,
        _resolve,
        _sha256,
        _summary_series,
    )
    from run_training_plan import (  # type: ignore
        _artifact_hashes,
        _command_hash,
        _config_hash,
        _job_specs,
        _validate_plan,
    )
    from utils.acf_guidance import compile_channel_lag_support  # type: ignore


EXPECTED_PHASES: Tuple[Tuple[str, str, bool], ...] = (
    ("channel_lag_bank_off_h96", "off", False),
    ("channel_lag_bank_on_h96", "on", True),
)
PHASES = EXPECTED_PHASES
EXPECTED_DATASETS: Tuple[str, ...] = ("ETTh1", "traffic")
EXPECTED_SEEDS: Tuple[int, ...] = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
EXPECTED_FEATURES = "M"
EXPECTED_ALIAS = "MambaACFAutoCon"
EXPECTED_TOP_K = 2
EXPECTED_LAMBDA = 0.03
EXPECTED_JOBS_PER_PHASE = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)
EXPECTED_JOBS = len(EXPECTED_PHASES) * EXPECTED_JOBS_PER_PHASE
EXPECTED_PLAN_STATUS = "exploratory_channel_lag_bank_forecast_level_draft"
EXPECTED_SELECTION = "abs_acf_times_legal_step_fraction"
EXPECTED_METRIC_SOURCE = "checkpoint/loss_logs.json:val_loss"
EXPECTED_METRIC_SOURCE_DESCRIPTION = (
    "val_loss is validation scaled MSE; optional val_mse/validation_mse aliases "
    "and optional val_mae/validation_mae aliases may be recorded"
)
# ``val_loss`` is the canonical value emitted by the current training loop and
# is the field named by ``EXPECTED_METRIC_SOURCE``.  The explicit aliases are
# retained only for compatibility checks; they must agree with that canonical
# trajectory and are never allowed to change the reported source.
VALIDATION_MSE_KEYS = ("val_loss", "val_mse", "validation_mse")
VALIDATION_MSE_ALIAS_KEYS = ("val_mse", "validation_mse")
VALIDATION_MAE_KEYS = ("val_mae", "validation_mae")

SUPPORT_METADATA_KEYS = (
    "mamba_acf_channel_lag_bank_selection",
    "mamba_acf_channel_lag_bank_resolved_lags",
    "mamba_acf_channel_lag_bank_legal_lags",
    "mamba_acf_channel_lag_bank_legal_step_fractions",
    "mamba_acf_channel_lag_bank_support_shape",
    "mamba_acf_channel_lag_bank_selected_slots",
    "mamba_acf_channel_lag_bank_total_slots",
    "mamba_acf_channel_lag_bank_density",
    "mamba_acf_channel_lag_bank_support_sha256",
)

# These values are useful diagnostics when present, but a screen result is
# never selected from them.  Requiring objective-v3 itself catches old logs
# whose relation semantics cannot be compared safely.
RELATION_LOG_KEYS = (
    "acf_relation_global_valid_fraction",
    "acf_relation_global_active_anchor_fraction",
    "acf_relation_global_candidate_fraction",
    "acf_relation_global_positive_count_mean",
    "acf_relation_global_sign_accuracy",
    "acf_relation_global_sign_hinge",
)

# Paths and derived lag-bank annotations are intentionally excluded from the
# one-factor signature.  The treatment flag is compared separately.  Runtime
# contract values are retained after removing its corresponding treatment key.
CONFIG_EXCLUDED_KEYS = {
    "name",
    "purpose",
    "output_root",
    "results_dir",
    "test_results_dir",
    "checkpoints",
    "result_log",
    "log",
    "model_id",
    "plan_job_id",
    "setting",
    "des",
    "task_name",
    "base_seed",
    "seed",
    "run_seed",
    "mamba_acf_prior_artifact_path",
}


class NotRunError(RuntimeError):
    """Raised when the registered matrix is not complete."""


def _read_json_safe(path: Path) -> Mapping[str, Any]:
    """Use the shared JSON helper while keeping a stable local name."""

    return _read_json(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and parsed != value:
        raise ValueError(f"{label} must be integral")
    return parsed


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _canonical(value: Any) -> Any:
    """Canonicalize persisted JSON for exact paired-config comparison."""

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


def _strip_runtime_treatment(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_runtime_treatment(item)
            for key, item in value.items()
            # The runtime contract stores the *effective* top-k value.  It is
            # therefore ``None`` for the dense/off arm and an integer for the
            # sparse/on arm, even though the requested top-k is the same
            # top-level configuration field in both jobs.  Keep that raw
            # top-level field in the common signature and remove both
            # treatment-derived runtime entries here; otherwise the dormant
            # ``None`` versus ``2`` value creates a false second factor.
            if str(key) not in {
                "mamba_acf_channel_lag_bank",
                "mamba_acf_channel_lag_top_k",
            }
        }
    if isinstance(value, list):
        return [_strip_runtime_treatment(item) for item in value]
    return value


def _config_signature(config: Mapping[str, Any]) -> Any:
    filtered: Dict[str, Any] = {}
    for key, value in config.items():
        key = str(key)
        if key in CONFIG_EXCLUDED_KEYS or key in SUPPORT_METADATA_KEYS:
            continue
        if key == "mamba_acf_channel_lag_bank":
            continue
        if key == "mamba_runtime_contract":
            filtered[key] = _strip_runtime_treatment(value)
        else:
            filtered[key] = value
    return _canonical(filtered)


def _positive_lags(value: Any, label: str, *, allow_empty: bool = False) -> List[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{label} must be non-empty")
    result: List[int] = []
    for item in value:
        parsed = _int(item, f"{label} item")
        if parsed <= 0:
            raise ValueError(f"{label} contains a non-positive lag")
        result.append(parsed)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate lags")
    return result


def _phase(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    matches = [item for item in phases if isinstance(item, Mapping) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"plan must contain exactly one phase named {name!r}")
    return matches[0]


def _phase_root(plan: Mapping[str, Any], name: str) -> Path:
    phase = _phase(plan, name)
    output_root = phase.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError(f"phase {name} has no output_root")
    return _resolve(output_root)


def _plan_value(plan: Mapping[str, Any], phase: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in phase:
        return phase[key]
    global_cfg = plan.get("global", {})
    return global_cfg.get(key, default) if isinstance(global_cfg, Mapping) else default


def _plan_errors(plan: Mapping[str, Any]) -> List[str]:
    """Return deterministic errors for the frozen two-arm plan contract."""

    errors: List[str] = []
    try:
        errors.extend(str(item) for item in _validate_plan(dict(plan)))
    except Exception as exc:  # pragma: no cover - defensive boundary.
        errors.append(f"generic plan validation raised: {exc}")

    if plan.get("status") != EXPECTED_PLAN_STATUS:
        errors.append(f"unexpected plan status: {plan.get('status')!r}")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution metadata is missing")
    else:
        if execution.get("authorized") is not False:
            errors.append("draft plan must have execution.authorized=false")
        if execution.get("training_started") is not False:
            errors.append("draft plan must have execution.training_started=false")
        if execution.get("status") != plan.get("status"):
            errors.append("execution status disagrees with plan status")

    phases = plan.get("phases")
    expected_names = [name for name, _, _ in EXPECTED_PHASES]
    if not isinstance(phases, list) or [item.get("name") for item in phases if isinstance(item, Mapping)] != expected_names:
        errors.append(f"phase names/order mismatch: expected {expected_names!r}")
    else:
        roots = set()
        for phase, (name, _, enabled) in zip(phases, EXPECTED_PHASES):
            if not isinstance(phase, Mapping):
                errors.append(f"phase {name} is not an object")
                continue
            roots.add(str(phase.get("output_root", "")))
            if phase.get("models") != [EXPECTED_ALIAS]:
                errors.append(f"{name} must use {EXPECTED_ALIAS}")
            if phase.get("datasets") != list(EXPECTED_DATASETS):
                errors.append(f"{name} dataset contract mismatch")
            if phase.get("horizons") != [EXPECTED_HORIZON]:
                errors.append(f"{name} horizon contract mismatch")
            if phase.get("seeds") != list(EXPECTED_SEEDS):
                errors.append(f"{name} seed contract mismatch")
            try:
                if _explicit_bool(phase.get("mamba_acf_channel_lag_bank"), f"{name} lag-bank flag") != enabled:
                    errors.append(f"{name} lag-bank flag mismatch")
            except ValueError as exc:
                errors.append(str(exc))
            for key, expected in {
                "features": EXPECTED_FEATURES,
                "autocon_lambda": EXPECTED_LAMBDA,
                "mamba_acf_channel_lag_top_k": EXPECTED_TOP_K,
                "stream_metrics": False,
                "stream_sample_windows": 1,
            }.items():
                actual = _plan_value(plan, phase, key)
                if isinstance(expected, bool):
                    try:
                        actual = _explicit_bool(actual, f"{name} {key}")
                    except ValueError:
                        pass
                elif isinstance(expected, float):
                    try:
                        actual = float(actual)
                    except (TypeError, ValueError):
                        pass
                elif isinstance(expected, int):
                    try:
                        actual = _int(actual, f"{name} {key}")
                    except ValueError:
                        pass
                if actual != expected:
                    errors.append(f"{name} {key} mismatch")
        if len(roots) != len(EXPECTED_PHASES):
            errors.append("phase output roots must be distinct")

    counts = plan.get("registered_job_counts")
    expected_counts = {
        "channel_lag_bank_off_h96": 6,
        "channel_lag_bank_on_h96": 6,
        "exploratory_total": 12,
    }
    if not isinstance(counts, Mapping):
        errors.append("registered_job_counts is missing")
    else:
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                errors.append(f"registered_job_counts[{key}] mismatch")
        # Some frozen plans include these aliases; when present they must not
        # silently describe a different matrix.
        for key in ("training_total", "total"):
            if key in counts and counts.get(key) != EXPECTED_JOBS:
                errors.append(f"registered_job_counts[{key}] mismatch")

    global_cfg = plan.get("global")
    if not isinstance(global_cfg, Mapping):
        errors.append("global configuration is missing")
    else:
        for key, expected in {
            "features": "M",
            "seq_len": 96,
            "label_len": 48,
            "mamba_acf_lag_mode": "train_peaks_available",
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_target_mode": "hard_selected",
            "mamba_acf_channel_order": "input",
            "mamba_acf_channel_conditioning": False,
            "mamba_variable_mixer": False,
            "autocon_wnorm": "LastVal",
            "autocon_acf_source": "raw",
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

    comparison = plan.get("comparison_contract")
    if not isinstance(comparison, Mapping):
        errors.append("comparison_contract is missing")
    else:
        if comparison.get("primary_metric") != "validation_scaled_mse":
            errors.append("comparison primary metric must be validation_scaled_mse")
        if comparison.get("selection_metric") != "validation_scaled_mse":
            errors.append("comparison selection metric must be validation_scaled_mse")
        if comparison.get("pairing_unit") != "same dataset, horizon, seed":
            errors.append("comparison pairing unit mismatch")
        if comparison.get("test_selection_allowed") is not False:
            errors.append("test metrics must not select the lag-bank arm")
        if comparison.get("test_metrics_produced") is not False:
            errors.append("test metrics must be disabled in this screen")
        if comparison.get("metric_source") != EXPECTED_METRIC_SOURCE:
            errors.append(
                "comparison metric source must be checkpoint/loss_logs.json:val_loss"
            )
        if comparison.get("metric_source_description") != EXPECTED_METRIC_SOURCE_DESCRIPTION:
            errors.append("comparison metric source semantics mismatch")
    return errors


def _validate_plan_contract(plan: Mapping[str, Any]) -> List[str]:
    """Public list-of-errors helper used by contract tests."""

    return _plan_errors(plan)


def validate_plan_contract(plan: Mapping[str, Any]) -> None:
    errors = _plan_errors(plan)
    if errors:
        raise ValueError("channel lag-bank plan validation failed: " + "; ".join(errors))


def _command_value(command: Any, name: str) -> Optional[str]:
    if not isinstance(command, list):
        return None
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
        return None
    return tokens[index + 1]


def _raw_manifest(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"manifest row is not an object: {path}")
            records.append(value)
    return records


def _missing_training_artifacts(plan: Mapping[str, Any]) -> List[str]:
    """Return only presence/completion gaps; integrity drift is an error."""

    missing: List[str] = []
    for phase_name, _, _ in EXPECTED_PHASES:
        try:
            root = _phase_root(plan, phase_name)
        except (KeyError, TypeError, ValueError) as exc:
            missing.append(f"{phase_name}: {exc}")
            continue
        validation_path = root / "phase_validation.json"
        manifest_path = root / "logs" / "job_manifest.jsonl"
        provenance_path = root / "provenance.json"
        for path in (validation_path, manifest_path, provenance_path):
            if not path.is_file():
                missing.append(str(path))
        if not validation_path.is_file() or not manifest_path.is_file():
            continue
        try:
            validation = _read_json_safe(validation_path)
        except (OSError, ValueError, TypeError):
            missing.append(str(validation_path) + " (unreadable)")
            continue
        if validation.get("valid") is not True:
            missing.append(str(validation_path) + " (valid=false)")
            continue
        try:
            records = _raw_manifest(manifest_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            missing.append(str(manifest_path) + " (unreadable/incomplete)")
            continue
        if len(records) != EXPECTED_JOBS_PER_PHASE:
            missing.append(f"{manifest_path} (incomplete: {len(records)}/{EXPECTED_JOBS_PER_PHASE})")
            continue
        for record in records:
            job_id = str(record.get("job_id", ""))
            if record.get("status") != "ok":
                missing.append(f"{job_id or manifest_path} (status is not ok)")
                continue
            marker_raw = record.get("expected_artifact")
            if not marker_raw:
                missing.append(f"{job_id} (expected_artifact missing)")
                continue
            marker = _resolve(str(marker_raw))
            result_dir = marker.parent
            checkpoint_root_raw = _command_value(record.get("command"), "--checkpoints")
            checkpoint_root = _resolve(checkpoint_root_raw) if checkpoint_root_raw else root / "checkpoints"
            checkpoint_dir = checkpoint_root / result_dir.name
            required = (
                marker,
                result_dir / "config.json",
                result_dir / "acf_prior.npz",
                checkpoint_dir / "checkpoint.pth",
                checkpoint_dir / "loss_logs.json",
            )
            for path in required:
                if not path.is_file():
                    missing.append(f"{job_id}: {path}")
            logs_path = checkpoint_dir / "loss_logs.json"
            if logs_path.is_file():
                try:
                    logs = _read_json_safe(logs_path)
                    _validation_metrics(logs, logs_path, job_id)
                except (NotRunError, OSError, ValueError, TypeError, json.JSONDecodeError):
                    missing.append(f"{job_id}: validation trajectory missing or invalid")
            if marker.is_file():
                try:
                    marker_payload = _read_json_safe(marker)
                    if marker_payload.get("status") != "trained_without_test":
                        missing.append(
                            f"{job_id}: training_complete status is incomplete"
                        )
                except (OSError, ValueError, TypeError):
                    missing.append(f"{job_id}: training_complete unreadable")
    return missing


def _validate_provenance(
    plan: Mapping[str, Any], plan_path: Path, phase_name: str, phase_root: Path,
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    path = phase_root / "provenance.json"
    if not path.is_file():
        raise NotRunError(f"missing immutable provenance: {path}")
    payload = _read_json_safe(path)
    if payload.get("phase") != phase_name:
        raise ValueError(f"provenance phase mismatch: {phase_name}")
    if payload.get("plan_sha256") != _sha256(plan_path):
        raise ValueError(f"provenance plan hash mismatch: {phase_name}")
    if validation.get("provenance") and _resolve(str(validation["provenance"])) != path.resolve():
        raise ValueError(f"phase validation points to another provenance file: {phase_name}")
    provenance_id = payload.get("provenance_id")
    source_tree = payload.get("source_tree_sha256")
    dataset_hashes = payload.get("dataset_hashes")
    if not isinstance(provenance_id, str) or not provenance_id:
        raise ValueError(f"provenance_id is missing: {phase_name}")
    if not isinstance(source_tree, str) or len(source_tree) != 64:
        raise ValueError(f"source_tree_sha256 is missing: {phase_name}")
    if not isinstance(dataset_hashes, Mapping):
        raise ValueError(f"dataset hashes are missing: {phase_name}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "provenance_id": provenance_id,
        "source_tree_sha256": source_tree,
        "dataset_hashes": dict(dataset_hashes),
    }


def _validate_phase_binding(
    plan_path: Path, phase_name: str, phase_root: Path,
) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]], Path]:
    validation_path = phase_root / "phase_validation.json"
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    if not validation_path.is_file() or not manifest_path.is_file():
        raise NotRunError(f"missing phase validation or manifest: {phase_name}")
    validation = _read_json_safe(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise NotRunError(f"phase validation is incomplete: {validation_path}")
    if validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError(f"phase validation plan hash mismatch: {phase_name}")
    declared_manifest = validation.get("manifest")
    if not declared_manifest:
        raise ValueError(f"phase validation manifest path is missing: {phase_name}")
    if _resolve(str(declared_manifest)) != manifest_path.resolve():
        raise ValueError(f"phase validation points to another manifest file: {phase_name}")
    try:
        records = _manifest_records(phase_root, validation)
    except ValueError as exc:
        text = str(exc).lower()
        if any(token in text for token in ("missing", "empty", "coverage", "status")):
            raise NotRunError(str(exc)) from exc
        raise
    if len(records) != EXPECTED_JOBS_PER_PHASE:
        raise NotRunError(f"incomplete manifest: {phase_name}")
    ids = [str(item.get("job_id", "")) for item in records]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"manifest job IDs are missing/duplicated: {phase_name}")
    hashes = validation.get("artifact_hashes_by_job")
    if not isinstance(hashes, Mapping) or {str(key) for key in hashes} != set(ids):
        raise ValueError(f"phase artifact hash map does not cover manifest: {phase_name}")
    for record, job_id in zip(records, ids):
        record_hashes = record.get("artifact_hashes")
        phase_hashes = hashes.get(job_id)
        if not isinstance(record_hashes, Mapping) or not isinstance(phase_hashes, Mapping):
            raise ValueError(f"artifact hash map is invalid: {job_id}")
        if dict(record_hashes) != dict(phase_hashes):
            raise ValueError(f"manifest/validation artifact hashes differ for {job_id}")
    return validation, records, manifest_path


def _check_config(
    config: Mapping[str, Any], plan: Mapping[str, Any], phase: Mapping[str, Any],
    enabled: bool, dataset: str, seed: int, job_id: str,
) -> Dict[str, Any]:
    channels = {"ETTh1": 7, "traffic": 862}.get(dataset)
    if channels is None:
        raise ValueError(f"unknown dataset for {job_id}: {dataset}")
    expected: Dict[str, Any] = {
        "model": EXPECTED_ALIAS,
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
        "mamba_use_acf_memory": True,
        "mamba_dynamic_acf": False,
        "mamba_acf_lag_mode": "train_peaks_available",
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_lag_mask": True,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_channel_order": "input",
        "mamba_acf_channel_conditioning": False,
        "mamba_variable_mixer": False,
        "skip_test": True,
        "stream_metrics": False,
    }
    for key, wanted in expected.items():
        actual = config.get(key, "encoder" if key == "mamba_acf_relation_representation" else None)
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

    if str(config.get("mamba_acf_relation_representation", "encoder")) != "encoder":
        raise ValueError(f"relation representation is not encoder for {job_id}")
    if _explicit_bool(config.get("mamba_acf_channel_lag_bank", False), f"config lag-bank flag for {job_id}") != enabled:
        raise ValueError(f"config lag-bank flag mismatch for {job_id}")
    top_k = _int(config.get("mamba_acf_channel_lag_top_k", EXPECTED_TOP_K), f"config lag-bank top-k for {job_id}")
    if top_k != EXPECTED_TOP_K or top_k != _int(_plan_value(plan, phase, "mamba_acf_channel_lag_top_k", EXPECTED_TOP_K), "plan lag-bank top-k"):
        raise ValueError(f"config lag-bank top-k mismatch for {job_id}")
    effective_lags = _positive_lags(config.get("mamba_acf_effective_lags"), f"config effective lags for {job_id}")
    prior_lags = _positive_lags(config.get("mamba_acf_prior_lags"), f"config prior lags for {job_id}")
    if effective_lags != prior_lags:
        raise ValueError(f"effective/prior lag mismatch for {job_id}")
    if str(config.get("mamba_acf_effective_lag_source")) != "train_peaks_available":
        raise ValueError(f"effective lag source is not train_peaks_available for {job_id}")
    if _explicit_bool(config.get("mamba_acf_effective_lag_fallback", False), f"effective lag fallback for {job_id}"):
        raise ValueError(f"effective lag fallback is enabled for {job_id}")
    if config.get("mamba_acf_prior_artifact") != "acf_prior.npz":
        raise ValueError(f"prior artifact name mismatch for {job_id}")
    prior_hash = str(config.get("mamba_acf_prior_sha256", ""))
    if len(prior_hash) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in prior_hash):
        raise ValueError(f"prior sha256 is invalid for {job_id}")
    if _explicit_bool(config.get("save_predictions", False), f"save_predictions for {job_id}"):
        raise ValueError(f"save_predictions must be false for {job_id}")
    return {
        "signature": _config_signature(config),
        "effective_lags": effective_lags,
        "prior_lags": prior_lags,
        "prior_sha256": prior_hash.lower(),
        "top_k": top_k,
        "channels": channels,
    }


def _load_sidecar(path: Path, config: Mapping[str, Any], job_id: str) -> Dict[str, Any]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as payload:
            lags = np.asarray(payload["lags"]).reshape(-1)
            source = str(np.asarray(payload["acf_source"]).item()).lower()
            values = np.asarray(payload["acf_values"], dtype=np.float32)
            full = np.asarray(payload["acf_full_values"], dtype=np.float32)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"invalid ACF sidecar for {job_id}: {path}") from exc
    if source != "raw":
        raise ValueError(f"sidecar source is not raw for {job_id}")
    if lags.dtype.kind not in "iu":
        raise ValueError(f"sidecar lags are not integer for {job_id}")
    sidecar_lags = [int(item) for item in lags.tolist()]
    expected_lags = _positive_lags(config.get("mamba_acf_prior_lags"), f"config prior lags for {job_id}")
    if sidecar_lags != expected_lags:
        raise ValueError(f"sidecar/config lag order differs for {job_id}")
    if values.ndim != 2 or values.shape[1] != len(sidecar_lags):
        raise ValueError(f"sidecar compact shape is invalid for {job_id}")
    if full.ndim != 2 or full.shape[0] != values.shape[0] or full.shape[1] <= max(sidecar_lags):
        raise ValueError(f"sidecar complete-table shape is invalid for {job_id}")
    if not np.isfinite(values).all() or not np.isfinite(full).all():
        raise ValueError(f"sidecar contains non-finite values for {job_id}")
    if values.shape[0] != _int(config.get("enc_in"), f"config enc_in for {job_id}"):
        raise ValueError(f"sidecar channel count mismatch for {job_id}")
    prior_shape = config.get("mamba_acf_prior_shape")
    full_shape = config.get("mamba_acf_prior_full_shape")
    if prior_shape is not None and [int(item) for item in prior_shape] != list(values.shape):
        raise ValueError(f"config compact prior shape mismatch for {job_id}")
    if full_shape is not None and [int(item) for item in full_shape] != list(full.shape):
        raise ValueError(f"config full prior shape mismatch for {job_id}")
    return {
        "lags": sidecar_lags,
        "source": source,
        "compact_shape": list(values.shape),
        "full_shape": list(full.shape),
        "_values": values,
    }


def _support_contract(
    config: Mapping[str, Any], sidecar: Mapping[str, Any], enabled: bool, job_id: str,
) -> Dict[str, Any]:
    import numpy as np

    values = np.asarray(sidecar["_values"], dtype=np.float32)
    support, legal_fraction, canonical_lags = compile_channel_lag_support(
        values,
        sidecar["lags"],
        seq_len=_int(config.get("seq_len"), f"config seq_len for {job_id}"),
        horizon=_int(config.get("pred_len"), f"config pred_len for {job_id}"),
        top_k=_int(config.get("mamba_acf_channel_lag_top_k", EXPECTED_TOP_K), f"top-k for {job_id}"),
    )
    resolved = _positive_lags(config.get("mamba_acf_effective_lags"), f"effective lags for {job_id}")
    if list(canonical_lags) != resolved:
        raise ValueError(f"support/effective lag mismatch for {job_id}")
    support_hash = hashlib.sha256(np.asarray(support, dtype=np.uint8).tobytes(order="C")).hexdigest()
    legal_lags = [int(lag) for lag, fraction in zip(canonical_lags, legal_fraction.tolist()) if fraction > 0.0]
    expected_shape = [int(values.shape[0]), int(values.shape[1])]
    expected_selected = int(values.shape[0]) * EXPECTED_TOP_K
    expected_total = int(values.size)
    expected_density = expected_selected / max(1, expected_total)
    if enabled:
        checks = {
            "mamba_acf_channel_lag_bank_selection": EXPECTED_SELECTION,
            "mamba_acf_channel_lag_bank_resolved_lags": resolved,
            "mamba_acf_channel_lag_bank_legal_lags": legal_lags,
            "mamba_acf_channel_lag_bank_support_shape": expected_shape,
            "mamba_acf_channel_lag_bank_selected_slots": expected_selected,
            "mamba_acf_channel_lag_bank_total_slots": expected_total,
            "mamba_acf_channel_lag_bank_support_sha256": support_hash,
        }
        for key, expected in checks.items():
            actual = config.get(key)
            if key.endswith("support_sha256"):
                if str(actual).lower() != str(expected).lower():
                    raise ValueError(f"support hash mismatch for {job_id}")
            elif actual != expected:
                raise ValueError(f"support metadata {key} mismatch for {job_id}")
        try:
            fractions_match = np.array_equal(
                np.asarray(config.get("mamba_acf_channel_lag_bank_legal_step_fractions"), dtype=np.float32),
                np.asarray(legal_fraction, dtype=np.float32),
            )
        except (TypeError, ValueError):
            fractions_match = False
        if not fractions_match:
            raise ValueError(f"support legal-step fractions mismatch for {job_id}")
        try:
            if not bool(np.isclose(float(config.get("mamba_acf_channel_lag_bank_density")), expected_density, rtol=0.0, atol=1e-12)):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"support density mismatch for {job_id}") from exc
        metadata_valid = True
    else:
        # The off arm may persist these fields as JSON nulls.  Any non-null
        # active annotation would make the claimed one-factor contrast false.
        for key in SUPPORT_METADATA_KEYS:
            value = config.get(key)
            if value not in (None, [], ""):
                raise ValueError(f"disabled arm has support metadata {key}: {job_id}")
        metadata_valid = True
    return {
        "enabled": enabled,
        "top_k": EXPECTED_TOP_K,
        "selection": EXPECTED_SELECTION if enabled else None,
        "resolved_lags": resolved,
        "legal_lags": legal_lags,
        "legal_step_fractions": [float(item) for item in legal_fraction.tolist()],
        "shape": expected_shape,
        "selected_slots": expected_selected if enabled else 0,
        "total_slots": expected_total if enabled else 0,
        "density": expected_density if enabled else 0.0,
        "computed_support_sha256": support_hash,
        "metadata_valid": metadata_valid,
    }


def _validation_metrics(logs: Mapping[str, Any], path: Path, job_id: str) -> Dict[str, Any]:
    """Extract best validation metrics without touching the test split.

    ``exp_long_term_forecasting_with_AutoCon`` records the pure forecasting
    validation trajectory as ``val_loss``.  Newer runners may additionally
    persist explicit ``val_mse``/``validation_mse`` aliases; those aliases are
    accepted only as an element-wise consistency check against ``val_loss``.
    This keeps the persisted metric source auditable instead of allowing an
    alias to silently override the field declared by the plan.  MAE is optional
    because the training loop historically logs validation MSE only.
    """

    def parse_series(key: str, *, required: bool) -> Optional[List[float]]:
        if key not in logs:
            if required:
                raise NotRunError(f"missing validation trajectory {key}: {path}")
            return None
        raw = logs[key]
        values = raw if isinstance(raw, list) else [raw]
        if not values:
            raise NotRunError(f"empty validation trajectory {key}: {path}")
        return [_finite(item, f"{job_id} {key}") for item in values]

    # The plan names val_loss explicitly.  Require it even when a newer log
    # happens to contain one of the optional aliases; alias-only logs cannot be
    # attributed to the declared source and therefore are not auditable.
    mse_values = parse_series("val_loss", required=True)
    assert mse_values is not None
    verified_mse_aliases: List[str] = []
    for alias in VALIDATION_MSE_ALIAS_KEYS:
        alias_values = parse_series(alias, required=False)
        if alias_values is None:
            continue
        if len(alias_values) != len(mse_values) or any(
            not math.isclose(primary, aliased, rel_tol=1e-9, abs_tol=1e-12)
            for primary, aliased in zip(mse_values, alias_values)
        ):
            raise ValueError(
                f"validation MSE alias {alias} disagrees with val_loss: {path}"
            )
        verified_mse_aliases.append(alias)
    result: Dict[str, Any] = {
        "mse": min(mse_values),
        "mse_source": "val_loss",
        "mse_aliases_verified": verified_mse_aliases,
        "mse_trajectory": {
            "count": len(mse_values),
            "first": mse_values[0],
            "last": mse_values[-1],
            "best": min(mse_values),
        },
        "mae": None,
        "mae_source": None,
        "mae_trajectory": None,
    }
    mae_values: Optional[List[float]] = None
    mae_key: Optional[str] = None
    for candidate in VALIDATION_MAE_KEYS:
        candidate_values = parse_series(candidate, required=False)
        if candidate_values is None:
            continue
        if mae_values is None:
            mae_values, mae_key = candidate_values, candidate
            continue
        if len(candidate_values) != len(mae_values) or any(
            not math.isclose(primary, aliased, rel_tol=1e-9, abs_tol=1e-12)
            for primary, aliased in zip(mae_values, candidate_values)
        ):
            raise ValueError(
                f"validation MAE alias {candidate} disagrees with {mae_key}: {path}"
            )
    if mae_values is not None and mae_key is not None:
        result.update(
            {
                "mae": min(mae_values),
                "mae_source": mae_key,
                "mae_trajectory": {
                    "count": len(mae_values),
                    "first": mae_values[0],
                    "last": mae_values[-1],
                    "best": min(mae_values),
                },
            }
        )
    return result


def _load_phase(
    plan: Mapping[str, Any], plan_path: Path, phase_name: str, arm: str, enabled: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase = _phase(plan, phase_name)
    phase_root = _phase_root(plan, phase_name)
    validation, records, manifest_path = _validate_phase_binding(plan_path, phase_name, phase_root)
    provenance = _validate_provenance(plan, plan_path, phase_name, phase_root, validation)
    registered = _job_specs(dict(plan), dict(phase), {}, allow_unresolved=True)
    registered_by_coord = {
        (str(job["dataset"]), int(job["pred_len"]), int(job["seed"])): job
        for job in registered
    }
    result_root = (phase_root / "results").resolve()
    default_checkpoint_root = (phase_root / "checkpoints").resolve()
    observed: set[Tuple[str, int]] = set()
    rows: List[Dict[str, Any]] = []
    validation_hashes = validation.get("artifact_hashes_by_job")
    assert isinstance(validation_hashes, Mapping)
    for record in records:
        job_id = str(record.get("job_id", ""))
        if record.get("phase") != phase_name:
            raise ValueError(f"manifest phase mismatch: {job_id}")
        if record.get("status") != "ok":
            raise NotRunError(f"job is incomplete: {job_id}")
        dataset = str(record.get("dataset"))
        seed = _int(record.get("seed"), f"{job_id} seed")
        horizon = _int(record.get("pred_len"), f"{job_id} horizon")
        coordinate = (dataset, seed)
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS or horizon != EXPECTED_HORIZON or coordinate in observed:
            raise ValueError(f"unexpected or duplicate coordinate: {job_id}")
        observed.add(coordinate)
        registered_job = registered_by_coord.get((dataset, horizon, seed))
        if registered_job is None:
            raise ValueError(f"unregistered coordinate: {job_id}")
        expected_config_hash = _config_hash(dict(registered_job), record.get("provenance_id"))
        if record.get("config_hash") != expected_config_hash:
            raise ValueError(f"registered config hash mismatch: {job_id}")
        command = record.get("command")
        if record.get("command_sha256") and isinstance(command, list) and record.get("command_sha256") != _command_hash(command):
            raise ValueError(f"command hash mismatch: {job_id}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise NotRunError(f"missing expected artifact: {job_id}")
        marker_path = _resolve(str(expected_artifact))
        if marker_path.name != "training_complete.json" or not marker_path.is_file() or not _inside(marker_path, result_root):
            raise NotRunError(f"missing training completion marker: {job_id}")
        marker = _read_json_safe(marker_path)
        if marker.get("status") != "trained_without_test":
            raise NotRunError(f"training completion marker is incomplete: {job_id}")
        result_dir = marker_path.parent
        checkpoint_raw = _command_value(command, "--checkpoints")
        checkpoint_root = _resolve(checkpoint_raw) if checkpoint_raw else default_checkpoint_root
        checkpoint_dir = checkpoint_root / result_dir.name
        if not _inside(checkpoint_dir, phase_root):
            raise ValueError(f"checkpoint root outside phase root: {job_id}")
        required = {
            "config.json": result_dir / "config.json",
            "training_complete.json": marker_path,
            "acf_prior.npz": result_dir / "acf_prior.npz",
            "checkpoint.pth": checkpoint_dir / "checkpoint.pth",
            "loss_logs.json": checkpoint_dir / "loss_logs.json",
        }
        absent = [str(path) for path in required.values() if not path.is_file()]
        if absent:
            raise NotRunError(f"incomplete artifacts for {job_id}: {', '.join(absent)}")
        record_hashes = record.get("artifact_hashes")
        phase_hashes = validation_hashes.get(job_id)
        if not isinstance(record_hashes, Mapping) or not isinstance(phase_hashes, Mapping) or dict(record_hashes) != dict(phase_hashes):
            raise ValueError(f"manifest/phase artifact hashes differ: {job_id}")
        try:
            hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        except ValueError as exc:
            if "missing artifact" in str(exc).lower():
                raise NotRunError(str(exc)) from exc
            raise
        for name in required:
            if name not in hashes:
                raise ValueError(f"artifact hash map is missing {name}: {job_id}")
        # Cross-check the runner's independent hash implementation as well.
        if hashes != _artifact_hashes(result_dir, checkpoint_dir):
            raise ValueError(f"artifact hash replay mismatch: {job_id}")

        config = _read_json_safe(required["config.json"])
        config_info = _check_config(config, plan, phase, enabled, dataset, seed, job_id)
        sidecar_hash = hashes["acf_prior.npz"].lower()
        if config_info["prior_sha256"] != sidecar_hash:
            raise ValueError(f"config/sidecar hash mismatch: {job_id}")
        sidecar = _load_sidecar(required["acf_prior.npz"], config, job_id)
        support = _support_contract(config, sidecar, enabled, job_id)
        logs_path = required["loss_logs.json"]
        logs = _read_json_safe(logs_path)
        validation_metrics = _validation_metrics(logs, logs_path, job_id)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {job_id}")
        relation = {
            key: _summary_series(logs, key, logs_path)
            for key in RELATION_LOG_KEYS
            if key in logs
        }
        rows.append({
            "phase": phase_name,
            "arm": arm,
            "enabled": enabled,
            "dataset": dataset,
            "seed": seed,
            "horizon": horizon,
            "mse": validation_metrics["mse"],
            "mae": validation_metrics["mae"],
            "mse_source": validation_metrics["mse_source"],
            "mse_trajectory": validation_metrics["mse_trajectory"],
            "mae_source": validation_metrics["mae_source"],
            "mae_trajectory": validation_metrics["mae_trajectory"],
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "epochs_ran": _int(record.get("epochs_ran", logs.get("epochs_ran", 0)), f"{job_id} epochs_ran"),
            "best_epoch": _int(record.get("best_epoch", logs.get("best_epoch", 0)), f"{job_id} best_epoch"),
            "job_id": job_id,
            "sidecar_sha256": sidecar_hash,
            "artifact_hashes": dict(hashes),
            "config_signature": config_info["signature"],
            "effective_lags": config_info["effective_lags"],
            "support": support,
            "validation_metrics": validation_metrics,
            "relation_diagnostics": relation,
            "autocon_loss": (
                _summary_series(logs, "autocon_loss", logs_path)
                if "autocon_loss" in logs else None
            ),
        })
    expected = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    if observed != expected:
        raise NotRunError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows, {
        "phase": phase_name,
        "arm": arm,
        "enabled": enabled,
        "root": str(phase_root),
        "validation": str(phase_root / "phase_validation.json"),
        "validation_sha256": _sha256(phase_root / "phase_validation.json"),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "provenance": provenance,
        "expected_jobs": EXPECTED_JOBS_PER_PHASE,
        "observed_jobs": len(rows),
        "valid": True,
    }


def _paired_test(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [_finite(value, "paired delta") for value in values]
    if not numbers:
        raise ValueError("empty paired delta vector")
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    t_stat: Optional[float] = None
    p_value: Optional[float] = None
    if scipy_stats is not None and len(numbers) > 1:
        result = scipy_stats.ttest_1samp(numbers, 0.0)
        if math.isfinite(float(result.statistic)):
            t_stat = float(result.statistic)
        if math.isfinite(float(result.pvalue)):
            p_value = float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat = mean / (sd / math.sqrt(len(numbers)))
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd if math.isfinite(sd) else None,
        "wins_lower": sum(value < 0 for value in numbers),
        "losses_higher": sum(value > 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
        "delta_values": numbers,
    }


def _relative(delta: float, reference: float) -> Optional[float]:
    return delta / reference if reference else None


def _contrast(
    pairs: Sequence[Mapping[str, Any]], metric: str,
) -> Optional[Dict[str, Any]]:
    """Summarize a paired metric, omitting optional metrics when unavailable.

    Validation MSE and elapsed time are required for every pair.  Validation
    MAE is an optional diagnostic because the current training loop persists
    only ``val_loss``.  A partially populated MAE vector is therefore omitted
    as a whole instead of being reduced over a different set of coordinates.
    """

    values = [item.get(f"delta_{metric}") for item in pairs]
    if any(value is None for value in values):
        if metric == "mae":
            return None
        raise ValueError(f"required paired metric is missing: {metric}")
    deltas = [_finite(value, f"paired {metric} delta") for value in values]
    result = _paired_test(deltas)
    reference = statistics.fmean(
        _finite(item[f"off_{metric}"], f"off {metric}") for item in pairs
    )
    treatment = statistics.fmean(
        _finite(item[f"on_{metric}"], f"on {metric}") for item in pairs
    )
    result.update({
        "reference_mean": reference,
        "treatment_mean": treatment,
        "relative_delta_mean": _relative(result["mean"], reference),
    })
    return result


def _make_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    expected_arms = {"off", "on"}
    index: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset")), _int(row.get("seed"), "row seed"), str(row.get("arm")))
        if key in index:
            raise ValueError(f"duplicate row coordinate: {key}")
        index[key] = row
    if {str(row.get("arm")) for row in rows} != expected_arms or len(rows) != EXPECTED_JOBS:
        raise ValueError(f"row matrix must contain exactly {EXPECTED_JOBS} off/on rows")
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            off = index.get((dataset, seed, "off"))
            on = index.get((dataset, seed, "on"))
            if off is None or on is None:
                raise ValueError(f"incomplete paired coordinate: {dataset} seed {seed}")
            if off.get("sidecar_sha256") != on.get("sidecar_sha256"):
                raise ValueError(f"sidecars differ at {dataset} seed {seed}")
            if _canonical(off.get("config_signature")) != _canonical(on.get("config_signature")):
                raise ValueError(f"common config differs at {dataset} seed {seed}")
            off_support = off.get("support", {})
            on_support = on.get("support", {})
            if off_support.get("computed_support_sha256") != on_support.get("computed_support_sha256"):
                raise ValueError(f"computed support differs at {dataset} seed {seed}")
            if off_support.get("resolved_lags") != on_support.get("resolved_lags"):
                raise ValueError(f"resolved lag bank differs at {dataset} seed {seed}")
            if off_support.get("legal_step_fractions") != on_support.get("legal_step_fractions"):
                raise ValueError(f"legal support differs at {dataset} seed {seed}")
            off_mae = off.get("mae")
            on_mae = on.get("mae")
            if off_mae is None or on_mae is None:
                # Keep the pair shape stable while allowing the optional MAE
                # diagnostic to be omitted for the entire matrix.
                off_mae_value = on_mae_value = delta_mae = None
            else:
                off_mae_value = _finite(off_mae, f"off mae {dataset} seed {seed}")
                on_mae_value = _finite(on_mae, f"on mae {dataset} seed {seed}")
                delta_mae = on_mae_value - off_mae_value
            pairs.append({
                "dataset": dataset,
                "seed": seed,
                "off_mse": _finite(off.get("mse"), f"off mse {dataset} seed {seed}"),
                "on_mse": _finite(on.get("mse"), f"on mse {dataset} seed {seed}"),
                "delta_mse": _finite(on.get("mse"), f"on mse {dataset} seed {seed}")
                - _finite(off.get("mse"), f"off mse {dataset} seed {seed}"),
                "off_mae": off_mae_value,
                "on_mae": on_mae_value,
                "delta_mae": delta_mae,
                "off_elapsed_sec": _finite(
                    off.get("elapsed_sec"), f"off elapsed_sec {dataset} seed {seed}"
                ),
                "on_elapsed_sec": _finite(
                    on.get("elapsed_sec"), f"on elapsed_sec {dataset} seed {seed}"
                ),
                "delta_elapsed_sec": _finite(
                    on.get("elapsed_sec"), f"on elapsed_sec {dataset} seed {seed}"
                ) - _finite(
                    off.get("elapsed_sec"), f"off elapsed_sec {dataset} seed {seed}"
                ),
                "mae_available": delta_mae is not None,
                "sidecar_sha256": off.get("sidecar_sha256"),
                "off_job_id": off.get("job_id"),
                "on_job_id": on.get("job_id"),
                "support": {"off": off_support, "on": on_support},
            })
    return pairs


def _stopping_decision(pairs: Sequence[Mapping[str, Any]], mse: Mapping[str, Any]) -> Dict[str, Any]:
    wins_by_dataset = {
        dataset: sum(float(item["delta_mse"]) < 0.0 for item in pairs if item["dataset"] == dataset)
        for dataset in EXPECTED_DATASETS
    }
    all_worse_by_dataset = {
        dataset: all(float(item["delta_mse"]) > 0.0 for item in pairs if item["dataset"] == dataset)
        for dataset in EXPECTED_DATASETS
    }
    pooled_relative = mse.get("relative_delta_mean")
    pooled_worse = pooled_relative is not None and float(pooled_relative) > 0.01
    stop = bool(pooled_worse or any(all_worse_by_dataset.values()))
    return {
        "criterion": "stop if pooled paired validation scaled MSE worsens by >1% or either dataset worsens on all three seeds",
        "pooled_relative_delta": pooled_relative,
        "pooled_worsens_more_than_one_percent": pooled_worse,
        "favorable_coordinates": int(mse.get("wins_lower", 0)),
        "favorable_coordinates_by_dataset": wins_by_dataset,
        "all_three_worse_by_dataset": all_worse_by_dataset,
        "meets_descriptive_screen": not stop,
        "action": (
            "stop exploratory program; do not add arms, seeds, horizons, or lambdas"
            if stop else
            "eligible only for a separately reviewed frozen replication"
        ),
    }


def build_readout(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    validation: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = [dict(row) for row in rows]
    # Rows loaded from loss_logs carry the field selected by
    # ``_validation_metrics``.  Reject a caller-supplied alternate source
    # rather than emitting a readout whose declared primary source disagrees
    # with the values used in its contrasts.  Synthetic contract fixtures may
    # omit the field entirely, so the check is conditional for compatibility.
    observed_mse_sources = {
        str(row["mse_source"])
        for row in normalized
        if row.get("mse_source") is not None
    }
    if observed_mse_sources and observed_mse_sources != {"val_loss"}:
        raise ValueError(
            "readout MSE source must be val_loss; observed "
            + ", ".join(sorted(observed_mse_sources))
        )
    pairs = _make_pairs(normalized)
    # Do not report MAE for only a subset of jobs.  The training loop currently
    # writes val_loss (MSE) but may not persist a validation MAE trajectory.
    mae_available = bool(normalized) and all(
        row.get("mae") is not None for row in normalized
    )
    pooled = {
        "mse": _contrast(pairs, "mse"),
        "elapsed_sec": _contrast(pairs, "elapsed_sec"),
    }
    if mae_available:
        mae_contrast = _contrast(pairs, "mae")
        if mae_contrast is not None:
            pooled["mae"] = mae_contrast
    by_dataset: Dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        block: Dict[str, Any] = {
            "n": len(subset),
            "mse": _contrast(subset, "mse"),
            "elapsed_sec": _contrast(subset, "elapsed_sec"),
        }
        if mae_available:
            mae_contrast = _contrast(subset, "mae")
            if mae_contrast is not None:
                block["mae"] = mae_contrast
        by_dataset[dataset] = block
    support_summary: Dict[str, Any] = {}
    for arm in ("off", "on"):
        arm_rows = [row for row in normalized if row.get("arm") == arm]
        values = [row.get("support", {}) for row in arm_rows]
        densities = [float(item.get("density", 0.0)) for item in values]
        support_summary[arm] = {
            "n": len(values),
            "density_mean": statistics.fmean(densities) if densities else None,
            "computed_support_hashes": sorted({item.get("computed_support_sha256") for item in values}),
            "resolved_lags": sorted({tuple(item.get("resolved_lags", [])) for item in values}),
            "metadata_valid_all": all(item.get("metadata_valid") is True for item in values),
        }
    invariants = {
        "paired_coordinates": len(pairs),
        "sidecar_hashes_equal_per_coordinate": True,
        "common_config_equal_after_excluding_treatment_and_derived_metadata": True,
        "only_registered_treatment": "mamba_acf_channel_lag_bank",
        "raw_static_prior": True,
        "support_metadata_replayed": True,
        "support_hashes_equal_per_coordinate": True,
    }
    plan_path = Path(str(plan.get("_path", ""))).resolve()
    result = {
        "schema_version": 1,
        "status": "exploratory_channel_lag_bank_forecast_screen_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "phases": [name for name, _, _ in EXPECTED_PHASES],
        "rows": normalized,
        "pairs": pairs,
        "validation": list(validation or []),
        "invariants": invariants,
        "support_summary": support_summary,
        "metric_source": {
            "primary": EXPECTED_METRIC_SOURCE,
            "primary_semantics": EXPECTED_METRIC_SOURCE_DESCRIPTION,
            "mse_canonical_field": "val_loss",
            "mse_aliases": list(VALIDATION_MSE_ALIAS_KEYS),
            "mse_alias_policy": (
                "optional aliases are accepted only when their complete "
                "trajectory matches val_loss element-wise; val_loss is required"
            ),
            "mae_aliases": list(VALIDATION_MAE_KEYS),
            "mae_available_all_rows": mae_available,
            "mse_sources_observed": sorted(observed_mse_sources),
            "test_metrics_read": False,
        },
        "contrasts": {
            "on_minus_off": pooled,
            "by_dataset": by_dataset,
        },
        "stopping_decision": _stopping_decision(pairs, pooled["mse"]),
        "evidence_boundary": "Validation-only paired descriptive evidence; this is explicitly non-confirmatory. P-values are unadjusted descriptive diagnostics, and no universal accuracy, speed, or representation-learning claim is permitted.",
    }
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def markdown(result: Mapping[str, Any]) -> str:
    pooled = result["contrasts"]["on_minus_off"]
    mse = pooled["mse"]
    mae = pooled.get("mae")
    metric_source = result.get("metric_source", {})
    mae_status = (
        "reported because every row has a validation MAE trajectory"
        if metric_source.get("mae_available_all_rows")
        else "omitted because val_mae/validation_mae is not present for every row"
    )
    lines = [
        "# MambaACF Channel Lag-Bank Forecast Screen",
        "",
        "> Exploratory, validation-only, and non-confirmatory. The registered treatment is the detached `mamba_acf_channel_lag_bank` support mask; no test metric is used for selection.",
        "",
        "- Primary metric: best validation scaled MSE from `checkpoint/loss_logs.json:val_loss`; optional `val_mse`/`validation_mse` fields are consistency-checked aliases and cannot override it. Test metrics are not read.",
        f"- Validation MAE is {mae_status}.",
        "",
        "## Paired summary",
        "",
        "| Contrast (on - off) | n | MSE delta | MSE relative | MSE wins | MSE p (descriptive) | MAE delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    p = mse.get("two_sided_p")
    p_text = "n/a" if p is None else f"{float(p):.4g}"
    lines.append(
        f"| on - off | {mse['n']} | {_fmt(mse['mean'])} | {_fmt(100 * (mse.get('relative_delta_mean') or 0.0), 3)}% | {mse['wins_lower']}/{mse['n']} | {p_text} | {_fmt(mae.get('mean') if mae else None)} |"
    )
    lines += [
        "",
        "Negative deltas favor the sparse on arm. P-values and pooled means are descriptive only.",
        "",
        "## Dataset splits",
        "",
        "| Dataset | n | MSE delta | MSE relative | MSE wins | MSE p | MAE delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in EXPECTED_DATASETS:
        block = result["contrasts"]["by_dataset"][dataset]
        dataset_mse = block["mse"]
        dataset_mae = block.get("mae")
        lines.append(
            f"| {dataset} | {block['n']} | {_fmt(dataset_mse['mean'])} | {_fmt(100 * (dataset_mse.get('relative_delta_mean') or 0.0), 3)}% | {dataset_mse['wins_lower']}/{dataset_mse['n']} | {_fmt(dataset_mse.get('two_sided_p'), 4)} | {_fmt(dataset_mae.get('mean') if dataset_mae else None)} |"
        )
    lines += [
        "",
        "## Support and invariants",
        "",
        "- The reader recomputes `abs(ACF) * legal_step_fraction` from each raw train-only sidecar and checks the recorded support shape, density, selected count, legal fractions, and SHA-256.",
        "- Each paired coordinate must share the same sidecar, resolved lag bank, derived support hash, and common config after excluding only the treatment and derived support metadata.",
        "- The forecast-level scope does not make the `LastVal` AutoCon representation prefix-causal; this screen concerns forecast-memory indexing only.",
        "",
        "## Stopping rule",
        "",
        f"- Criterion: `{result['stopping_decision']['criterion']}`.",
        f"- Favorable coordinates: `{result['stopping_decision']['favorable_coordinates']}/6`; by dataset `{result['stopping_decision']['favorable_coordinates_by_dataset']}`.",
        f"- Action: **{result['stopping_decision']['action']}**.",
        "- This screen cannot support a publication claim without an independently reviewed frozen replication and a broader baseline/resource protocol.",
        "",
        "## Provenance boundary",
        "",
        "- Missing or incomplete artifacts return `NOT_RUN` (exit code 3) and do not create readout files.",
        "- Artifact hashes, phase validation, and immutable provenance are checked before any output is written.",
        "",
    ]
    return "\n".join(lines)


def _not_run_message(plan: Mapping[str, Any]) -> str:
    missing = _missing_training_artifacts(plan)
    if not missing:
        return "NOT_RUN: no complete training artifacts found for the registered channel lag-bank screen"
    preview = "; ".join(missing[:6])
    suffix = " ..." if len(missing) > 6 else ""
    return f"NOT_RUN: registered channel lag-bank screen is incomplete ({preview}{suffix})"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/research/mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json"),
    )
    parser.add_argument("--output-json", "--output_json", dest="output_json", type=Path, default=None)
    parser.add_argument("--output-md", "--output-markdown", "--output_md", dest="output_md", type=Path, default=None)
    args = parser.parse_args(argv)
    plan_path = _resolve(args.plan)
    try:
        plan = dict(_read_json_safe(plan_path))
        plan["_path"] = str(plan_path)
        validate_plan_contract(plan)
        missing = _missing_training_artifacts(plan)
        if missing:
            print(_not_run_message(plan), file=sys.stderr)
            return 3
        rows: List[Dict[str, Any]] = []
        validation: List[Dict[str, Any]] = []
        for phase_name, arm, enabled in EXPECTED_PHASES:
            phase_rows, phase_meta = _load_phase(plan, plan_path, phase_name, arm, enabled)
            rows.extend(phase_rows)
            validation.append(phase_meta)
        result = build_readout(plan, rows, validation)
        output_json = _resolve(
            args.output_json
            or Path("artifacts/exploratory/mamba_acf_channel_lag_bank_forecast_screen_20260903/readout.json")
        )
        output_md = _resolve(
            args.output_md
            or Path("artifacts/exploratory/mamba_acf_channel_lag_bank_forecast_screen_20260903/readout.md")
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
        output_md.write_text(markdown(result), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "paired_coordinates": len(result["pairs"]), "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
        return 0
    except NotRunError as exc:
        print(f"NOT_RUN: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"analyze_mamba_acf_channel_lag_bank_screen: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
