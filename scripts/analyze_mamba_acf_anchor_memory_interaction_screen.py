#!/usr/bin/env python3
"""Read-only analysis for the current-tree ACF anchor x soft-memory screen.

The registered experiment is a four-arm 2x2 factorial matrix.  Every arm uses
the same variable-token ``MambaACFVariable`` alias (resolved to ``MambaACF``)
and differs only in ``mamba_acf_anchor_copy`` and
``mamba_use_acf_memory``.  The reader consumes completed validation artifacts
only; it never trains, evaluates a test split, or mutates an experiment root.

The primary interaction is a difference-in-differences on validation scaled
MSE::

    (anchor+memory - memory-only) - (anchor-only - plain)

Negative values indicate a super-additive reduction in error.  The result is
descriptive and does not authorize a paper claim or a test-set selection.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - descriptive statistics remain usable.
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
    )
    from scripts.run_training_plan import _hash_file_map, _job_specs, _validate_plan
except ModuleNotFoundError:  # pragma: no cover - direct script execution.
    from analyze_mamba_relation_null_control import (  # type: ignore
        _hash_map_for_record,
        _manifest_records,
        _read_json,
        _resolve,
        _sha256,
    )
    from run_training_plan import _hash_file_map, _job_specs, _validate_plan  # type: ignore


EXPECTED_PHASES: Tuple[Tuple[str, str, bool, bool], ...] = (
    ("plain_h96", "plain", False, False),
    ("anchor_only_h96", "anchor_only", True, False),
    ("memory_only_h96", "memory_only", False, True),
    ("anchor_memory_h96", "anchor_memory", True, True),
)
PHASES = EXPECTED_PHASES
EXPECTED_DATASETS: Tuple[str, ...] = ("ETTh1", "traffic")
EXPECTED_SEEDS: Tuple[int, ...] = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
EXPECTED_FEATURES = "M"
EXPECTED_ALIAS = "MambaACFVariable"
EXPECTED_EXECUTABLE = "MambaACF"
EXPECTED_JOBS_PER_PHASE = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)
EXPECTED_JOBS = len(EXPECTED_PHASES) * EXPECTED_JOBS_PER_PHASE
EXPECTED_PLAN_STATUS = "exploratory_mamba_acf_anchor_memory_interaction_screen_draft"
EXPECTED_METRIC_SOURCE = "checkpoint/loss_logs.json:val_loss"

# Construction metadata and the two registered factors are removed from a
# common signature and checked separately.  Runtime contracts can repeat the
# same factors in a nested object, so the recursive treatment stripper handles
# both locations.
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
    "mamba_acf_prior_sha256",
    "mamba_acf_prior_shape",
    "mamba_acf_prior_lags",
    "mamba_acf_prior_full_shape",
    "mamba_acf_prior_max_lag",
    "mamba_acf_prior_artifact",
    "mamba_runtime_contract_version",
}
TREATMENT_KEYS = {"mamba_acf_anchor_copy", "mamba_use_acf_memory"}
# ``mamba_runtime_contract`` records effective values after branch validation.
# Some branch-inactive options are serialized as ``None`` (for example the
# memory temperature when the memory path is off), while the same requested
# value is retained when the path is on.  These are derived implementation
# fields, not extra experimental factors, and must not break the factorial
# common-config comparison.
RUNTIME_DERIVED_KEYS = {"mamba_acf_temperature"}
PHASE_METADATA_KEYS = {
    "name",
    "purpose",
    "output_root",
    "datasets",
    "models",
    "features",
    "horizons",
    "seeds",
    "baseline_check",
}


class NotRunError(RuntimeError):
    """Raised when a registered coordinate has no complete artifacts."""


class ContractError(ValueError):
    """Raised when a completed artifact violates the registered contract."""


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
    raise ContractError(f"{label} is not an explicit boolean: {value!r}")


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ContractError(f"{label} is not finite: {value!r}")
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _strip_treatment(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_treatment(item)
            for key, item in value.items()
            if str(key) not in TREATMENT_KEYS
        }
    if isinstance(value, list):
        return [_strip_treatment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_treatment(item) for item in value)
    return value


def _strip_runtime_derived(value: Any) -> Any:
    """Remove treatment and branch-derived fields from runtime metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _strip_runtime_derived(item)
            for key, item in value.items()
            if str(key) not in TREATMENT_KEYS | RUNTIME_DERIVED_KEYS
        }
    if isinstance(value, list):
        return [_strip_runtime_derived(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_runtime_derived(item) for item in value)
    return value


def _config_signature(config: Mapping[str, Any]) -> Any:
    filtered: Dict[str, Any] = {}
    for key, value in config.items():
        key = str(key)
        if key in CONFIG_EXCLUDED_KEYS or key in TREATMENT_KEYS:
            continue
        if key == "mamba_runtime_contract":
            filtered[key] = _strip_runtime_derived(value)
        else:
            filtered[key] = _strip_treatment(value)
    return _canonical(filtered)


def _phase(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ContractError("plan phases must be a list")
    matches = [item for item in phases if isinstance(item, Mapping) and item.get("name") == name]
    if len(matches) != 1:
        raise ContractError(f"plan must contain exactly one phase named {name!r}")
    return matches[0]


def _phase_root(plan: Mapping[str, Any], name: str) -> Path:
    output_root = _phase(plan, name).get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ContractError(f"phase {name} has no output_root")
    return _resolve(output_root)


def _plan_value(plan: Mapping[str, Any], phase: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in phase:
        return phase[key]
    global_cfg = plan.get("global", {})
    return global_cfg.get(key, default) if isinstance(global_cfg, Mapping) else default


def _plan_errors(plan: Mapping[str, Any]) -> List[str]:
    """Return deterministic errors for the four-arm interaction contract."""

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
        if execution.get("status") != EXPECTED_PLAN_STATUS:
            errors.append("execution status disagrees with interaction plan status")
    if plan.get("require_acf_sidecar") is not True:
        errors.append("require_acf_sidecar must be true")

    phases = plan.get("phases")
    expected_names = [item[0] for item in EXPECTED_PHASES]
    if not isinstance(phases, list) or [item.get("name") for item in phases if isinstance(item, Mapping)] != expected_names:
        errors.append(f"phase names/order mismatch: expected {expected_names!r}")
    else:
        roots = set()
        for phase_item, (name, arm, anchor, memory) in zip(phases, EXPECTED_PHASES):
            if not isinstance(phase_item, Mapping):
                errors.append(f"phase {name} is not an object")
                continue
            roots.add(str(phase_item.get("output_root", "")))
            global_cfg = plan.get("global", {})
            if isinstance(global_cfg, Mapping):
                # Keep the factorial estimand honest: a phase may override
                # only registered treatments (plus structural metadata).  A
                # silent phase-level temperature, optimizer, or architecture
                # override would turn this into a confounded comparison.
                for key, value in phase_item.items():
                    key = str(key)
                    if key in PHASE_METADATA_KEYS or key in TREATMENT_KEYS:
                        continue
                    if key not in global_cfg:
                        errors.append(f"{name} contains unknown phase field {key}")
                    elif _canonical(value) != _canonical(global_cfg[key]):
                        errors.append(f"{name} overrides unregistered global factor {key}")
            if phase_item.get("models") != [EXPECTED_ALIAS]:
                errors.append(f"{name} must use {EXPECTED_ALIAS}")
            if phase_item.get("datasets") != list(EXPECTED_DATASETS):
                errors.append(f"{name} dataset contract mismatch")
            if phase_item.get("horizons") != [EXPECTED_HORIZON]:
                errors.append(f"{name} horizon contract mismatch")
            if phase_item.get("seeds") != list(EXPECTED_SEEDS):
                errors.append(f"{name} seed contract mismatch")
            for key, expected in {
                "features": EXPECTED_FEATURES,
                "mamba_acf_anchor_copy": anchor,
                "mamba_use_acf_memory": memory,
            }.items():
                try:
                    actual = _explicit_bool(_plan_value(plan, phase_item, key), f"{name} {key}") if isinstance(expected, bool) else _plan_value(plan, phase_item, key)
                except ContractError as exc:
                    errors.append(str(exc))
                    continue
                if actual != expected:
                    errors.append(f"{name} {key} mismatch: {actual!r} != {expected!r}")
            for key, expected in {
                "mamba_variable_mixer": True,
                "mamba_variable_group_size": 1,
                "mamba_acf_lag_mode": "train_peaks_available",
                "mamba_dynamic_acf": False,
                "mamba_acf_memory_gate_mode": "hidden",
                "mamba_acf_relation_loss": False,
                "autocon_wnorm": "MeanStd",
                "autocon_acf_source": "raw",
                "epochs": 8,
                "patience": 3,
                "skip_test": True,
                "stream_metrics": False,
            }.items():
                actual = _plan_value(plan, phase_item, key)
                if isinstance(expected, bool):
                    try:
                        actual = _explicit_bool(actual, f"{name} {key}")
                    except ContractError as exc:
                        errors.append(str(exc))
                        continue
                if actual != expected:
                    errors.append(f"{name} {key} mismatch: {actual!r} != {expected!r}")
        if len(roots) != len(EXPECTED_PHASES):
            errors.append("phase output roots must be distinct")

    counts = plan.get("registered_job_counts")
    expected_counts = {name: EXPECTED_JOBS_PER_PHASE for name, *_ in EXPECTED_PHASES}
    expected_counts.update({
        "exploratory_total": EXPECTED_JOBS,
        "training_total": EXPECTED_JOBS,
        "total": EXPECTED_JOBS,
        "training_started": False,
        "resource_runs_not_in_training_matrix": True,
    })
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
            "mamba_use_acf_memory": False,
            "mamba_acf_anchor_copy": False,
            "mamba_acf_lag_mode": "train_peaks_available",
            "mamba_dynamic_acf": False,
            "mamba_acf_relation_loss": False,
            "mamba_variable_mixer": True,
            "mamba_variable_group_size": 1,
            "mamba_acf_memory_gate_mode": "hidden",
            "autocon_wnorm": "MeanStd",
            "autocon_acf_source": "raw",
            "skip_test": True,
            "stream_metrics": False,
        }.items():
            actual = global_cfg.get(key)
            if isinstance(expected, bool):
                try:
                    actual = _explicit_bool(actual, f"global {key}")
                except ContractError as exc:
                    errors.append(str(exc))
                    continue
            if actual != expected:
                errors.append(f"global {key} mismatch: {actual!r} != {expected!r}")

    aliases = plan.get("alias_contract")
    alias = aliases.get(EXPECTED_ALIAS) if isinstance(aliases, Mapping) else None
    if not isinstance(alias, Mapping) or alias.get("executable_model") != EXPECTED_EXECUTABLE:
        errors.append("alias contract must resolve MambaACFVariable to MambaACF")
    comparison = plan.get("comparison_contract")
    if not isinstance(comparison, Mapping):
        errors.append("comparison_contract is missing")
    else:
        expected_comparison = {
            "primary_metric": "validation_scaled_mse",
            "metric_source": EXPECTED_METRIC_SOURCE,
            "pairing_unit": "same dataset, horizon, and seed across all four arms",
            "test_selection_allowed": False,
            "test_metrics_produced": False,
        }
        for key, expected in expected_comparison.items():
            if comparison.get(key) != expected:
                errors.append(f"comparison_contract[{key}] mismatch")
        planned = comparison.get("planned_contrasts")
        if not isinstance(planned, list) or not any("difference-in-differences" in str(item) for item in planned):
            errors.append("comparison_contract must register a difference-in-differences contrast")
    return errors


def validate_plan_contract(plan: Mapping[str, Any]) -> None:
    errors = _plan_errors(plan)
    if errors:
        raise ContractError("anchor-memory interaction plan validation failed: " + "; ".join(errors))


def _command_value(command: Sequence[Any], name: str) -> Optional[str]:
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
        return None
    return tokens[index + 1]


def _command_flag(command: Sequence[Any], name: str) -> bool:
    tokens = [str(item) for item in command]
    if name not in tokens:
        return False
    index = tokens.index(name)
    if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
        try:
            return _explicit_bool(tokens[index + 1], name)
        except ContractError:
            return True
    return True


def _validation_metrics(logs: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    values = logs.get("val_loss")
    if not isinstance(values, list) or not values:
        raise NotRunError(f"missing validation trajectory val_loss: {path}")
    numbers = [_finite(value, f"{path}:val_loss") for value in values]
    best_index = min(range(len(numbers)), key=numbers.__getitem__)
    result: Dict[str, Any] = {
        "mse": numbers[best_index],
        "mse_source": "val_loss",
        "best_epoch": int(logs.get("best_epoch", best_index + 1)),
        "epochs_ran": int(logs.get("epochs_ran", len(numbers))),
    }
    for key in ("val_mae", "validation_mae"):
        raw = logs.get(key)
        if raw is None:
            continue
        if not isinstance(raw, list) or len(raw) != len(numbers):
            raise ContractError(f"{key} trajectory length disagrees with val_loss: {path}")
        mae = [_finite(item, f"{path}:{key}") for item in raw]
        result["mae"] = mae[best_index]
        result["mae_source"] = key
        break
    return result


def _load_sidecar(path: Path, config: Mapping[str, Any], job_id: str) -> str:
    try:
        import numpy as np

        with np.load(str(path), allow_pickle=False) as payload:
            values = np.asarray(payload["acf_values"])
            full = np.asarray(payload["acf_full_values"])
            lags = np.asarray(payload["lags"]).reshape(-1)
            source = str(np.asarray(payload["acf_source"]).item()).lower()
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ContractError(f"invalid ACF sidecar for {job_id}: {exc}") from exc
    expected_lags = [int(item) for item in config.get("mamba_acf_prior_lags", [])]
    if source != "raw" or lags.dtype.kind not in "iu" or values.ndim != 2 or full.ndim != 2:
        raise ContractError(f"sidecar source/shape contract failed for {job_id}")
    if [int(item) for item in lags.tolist()] != expected_lags:
        raise ContractError(f"sidecar lag contract failed for {job_id}")
    if not np.isfinite(values).all() or not np.isfinite(full).all():
        raise ContractError(f"sidecar contains non-finite values for {job_id}")
    if values.shape[1] != len(expected_lags) or full.shape[0] != values.shape[0]:
        raise ContractError(f"sidecar dimensions disagree for {job_id}")
    return _sha256(path)


def _checkpoint_contract(path: Path, anchor_enabled: bool) -> Dict[str, Any]:
    try:
        import torch

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch.
            state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ContractError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(state, Mapping):
        raise ContractError(f"checkpoint is not a mapping: {path}")
    model_state = state.get("model", state)
    if not isinstance(model_state, Mapping):
        raise ContractError(f"checkpoint model state is not a mapping: {path}")
    keys = {str(key)[7:] if str(key).startswith("module.") else str(key) for key in model_state}
    has_anchor = "acf_anchor_scale" in keys
    if has_anchor != anchor_enabled:
        raise ContractError(f"checkpoint anchor-key mismatch: {path}")
    numel = sum(int(value.numel()) for value in model_state.values() if hasattr(value, "numel"))
    return {"numel": numel, "has_anchor_key": has_anchor, "keys": sorted(keys)}


def _phase_provenance(plan_path: Path, phase_root: Path, phase_name: str) -> Dict[str, Any]:
    path = phase_root / "provenance.json"
    if not path.is_file():
        raise NotRunError(f"missing provenance: {path}")
    provenance = _read_json(path)
    if provenance.get("plan_sha256") != _sha256(plan_path) or provenance.get("phase") != phase_name:
        raise ContractError(f"provenance plan/phase mismatch: {path}")
    snapshot = Path(str(provenance.get("source_snapshot", "")))
    if not snapshot.is_dir():
        raise ContractError(f"missing source snapshot: {snapshot}")
    source_files, tree_hash = _hash_file_map(snapshot)
    if tree_hash != provenance.get("source_tree_sha256"):
        raise ContractError(f"source snapshot hash drifted: {snapshot}")
    declared_files = provenance.get("source_files")
    if isinstance(declared_files, Mapping) and dict(declared_files) != source_files:
        raise ContractError(f"source snapshot file map drifted: {snapshot}")
    # ``run_training_plan.py`` predates this explicit field and leaves it
    # absent on exploratory provenance.  Absence is interpreted as the legacy
    # safe default; an explicit true value is always rejected.
    if provenance.get("test_selection_allowed", False) is not False:
        raise ContractError(f"provenance permits test selection: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "provenance_id": provenance.get("provenance_id"),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
    }


def _missing_training_artifacts(plan: Mapping[str, Any]) -> List[str]:
    """Return presence gaps without writing or parsing partial outputs."""

    missing: List[str] = []
    for phase_name, _, _, _ in EXPECTED_PHASES:
        try:
            root = _phase_root(plan, phase_name)
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            missing.append(f"{phase_name}: {exc}")
            continue
        for relative in ("phase_validation.json", "provenance.json", "logs/job_manifest.jsonl"):
            if not (root / relative).is_file():
                missing.append(str(root / relative))
        validation = root / "phase_validation.json"
        if validation.is_file():
            try:
                if _read_json(validation).get("valid") is not True:
                    missing.append(str(validation) + " (valid=false)")
            except (OSError, ValueError, TypeError):
                missing.append(str(validation) + " (unreadable)")
    return missing


def _load_phase(plan: Mapping[str, Any], plan_path: Path, phase_name: str, arm: str, anchor: bool, memory: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase_root = _phase_root(plan, phase_name)
    validation_path = phase_root / "phase_validation.json"
    if not validation_path.is_file():
        raise NotRunError(f"missing phase validation: {validation_path}")
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ContractError(f"invalid phase validation: {validation_path}")
    try:
        records = _manifest_records(phase_root, validation)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ContractError(str(exc)) from exc
    provenance = _phase_provenance(plan_path, phase_root, phase_name)
    # The launcher records data hashes in provenance.  Bind them to the
    # registered plan fingerprints so a copied/rewritten CSV cannot pass the
    # artifact checks under an otherwise identical source snapshot.
    declared_data_hashes = _read_json(phase_root / "provenance.json").get("dataset_hashes")
    if not isinstance(declared_data_hashes, Mapping):
        raise ContractError(f"provenance lacks dataset hashes: {phase_name}")
    fingerprints = plan.get("data_fingerprints", {})
    for dataset in EXPECTED_DATASETS:
        contract = fingerprints.get(dataset) if isinstance(fingerprints, Mapping) else None
        relative = contract.get("relative_path") if isinstance(contract, Mapping) else None
        expected_hash = contract.get("sha256") if isinstance(contract, Mapping) else None
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ContractError(f"plan data fingerprint is incomplete: {dataset}")
        data_path = ROOT / relative
        if not data_path.is_file() or _sha256(data_path) != expected_hash:
            raise ContractError(f"live dataset fingerprint mismatch: {dataset}")
        if declared_data_hashes.get(dataset) != expected_hash:
            raise ContractError(f"provenance dataset hash mismatch: {dataset}")
    result_root = phase_root / "results"
    checkpoint_root = phase_root / "checkpoints"
    expected_coordinates = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    observed: set[Tuple[str, int]] = set()
    rows: List[Dict[str, Any]] = []
    for record in records:
        job_id = str(record.get("job_id"))
        if record.get("status") != "ok" or record.get("phase") != phase_name:
            raise ContractError(f"unsuccessful manifest record: {job_id}")
        dataset = str(record.get("dataset"))
        seed = int(record.get("seed"))
        coordinate = (dataset, seed)
        if coordinate not in expected_coordinates or coordinate in observed:
            raise ContractError(f"unexpected or duplicate coordinate: {phase_name} {coordinate}")
        observed.add(coordinate)
        if int(record.get("pred_len")) != EXPECTED_HORIZON or str(record.get("model")) != EXPECTED_ALIAS:
            raise ContractError(f"manifest model/horizon mismatch: {job_id}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ContractError(f"missing expected artifact: {job_id}")
        expected_path = _resolve(str(expected_artifact))
        if expected_path.name != "training_complete.json" or not expected_path.is_file() or not _inside(expected_path, result_root):
            raise NotRunError(f"missing training-complete artifact: {job_id}")
        result_dir = expected_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        try:
            hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        except (OSError, ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        required_hashes = {"config.json", "training_complete.json", "acf_prior.npz", "checkpoint.pth", "loss_logs.json"}
        if not required_hashes.issubset(hashes):
            raise NotRunError(f"required artifacts are not hashed: {job_id}")
        config = _read_json(result_dir / "config.json")
        if config.get("model") != EXPECTED_EXECUTABLE or config.get("data") != dataset:
            raise ContractError(f"config model/dataset mismatch: {job_id}")
        for key, expected in {
            "features": EXPECTED_FEATURES,
            "pred_len": EXPECTED_HORIZON,
            "mamba_variable_mixer": True,
            "mamba_variable_group_size": 1,
            "mamba_acf_lag_mode": "train_peaks_available",
            "mamba_dynamic_acf": False,
            "mamba_acf_memory_gate_mode": "hidden",
            "mamba_acf_relation_loss": False,
            "AutoCon_acf_source": "raw",
        }.items():
            actual = config.get(key)
            if isinstance(expected, bool):
                actual = _explicit_bool(actual, f"{job_id} {key}")
            if actual != expected:
                raise ContractError(f"config {key} mismatch for {job_id}: {actual!r}")
        if _explicit_bool(config.get("mamba_acf_anchor_copy", False), f"{job_id} anchor") != anchor:
            raise ContractError(f"config anchor factor mismatch: {job_id}")
        if _explicit_bool(config.get("mamba_use_acf_memory", False), f"{job_id} memory") != memory:
            raise ContractError(f"config memory factor mismatch: {job_id}")
        sidecar_hash = _load_sidecar(result_dir / "acf_prior.npz", config, job_id)
        if str(config.get("mamba_acf_prior_sha256", "")).lower() not in {"", sidecar_hash}:
            raise ContractError(f"declared sidecar hash mismatch: {job_id}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        metrics = _validation_metrics(logs, logs_path)
        checkpoint = _checkpoint_contract(checkpoint_dir / "checkpoint.pth", anchor)
        rows.append({
            "phase": phase_name,
            "arm": arm,
            "anchor": anchor,
            "memory": memory,
            "dataset": dataset,
            "seed": seed,
            "horizon": EXPECTED_HORIZON,
            "mse": metrics["mse"],
            "mae": metrics.get("mae"),
            "mse_source": metrics["mse_source"],
            "best_epoch": metrics["best_epoch"],
            "epochs_ran": metrics["epochs_ran"],
            "sidecar_sha256": sidecar_hash,
            "config_signature": _config_signature(config),
            "checkpoint": checkpoint,
            "artifact_hashes": hashes,
            "job_id": job_id,
        })
    if observed != expected_coordinates:
        raise NotRunError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows, {
        "phase": phase_name,
        "arm": arm,
        "anchor": anchor,
        "memory": memory,
        "root": str(phase_root),
        "validation_sha256": _sha256(validation_path),
        "provenance": provenance,
        "expected_jobs": EXPECTED_JOBS_PER_PHASE,
        "observed_jobs": len(rows),
    }


def _paired_summary(deltas: Sequence[float], reference: Optional[Sequence[float]] = None, treatment: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    numbers = [_finite(value, "paired delta") for value in deltas]
    if not numbers:
        raise ContractError("empty paired delta vector")
    result: Dict[str, Any] = {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "sd": statistics.stdev(numbers) if len(numbers) > 1 else None,
        "wins_lower": sum(value < 0 for value in numbers),
        "losses_higher": sum(value > 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "deltas": numbers,
    }
    if reference is not None and treatment is not None:
        ref = [_finite(value, "reference metric") for value in reference]
        trt = [_finite(value, "treatment metric") for value in treatment]
        result.update({
            "reference_mean": statistics.fmean(ref),
            "treatment_mean": statistics.fmean(trt),
            "ratio_percent": (statistics.fmean(trt) / statistics.fmean(ref) - 1.0) * 100.0,
            "pairwise_relative_mean_percent": statistics.fmean((b / a - 1.0) * 100.0 for a, b in zip(ref, trt)),
        })
    if scipy_stats is not None and len(numbers) >= 2:
        test = scipy_stats.ttest_1samp(numbers, 0.0)
        result["paired_t_two_sided_p"] = float(test.pvalue) if math.isfinite(float(test.pvalue)) else None
        try:
            wilcoxon = scipy_stats.wilcoxon(numbers, alternative="less", method="exact")
        except TypeError:  # pragma: no cover - older scipy.
            wilcoxon = scipy_stats.wilcoxon(numbers, alternative="less", mode="exact")
        result["wilcoxon_less_p"] = float(wilcoxon.pvalue) if math.isfinite(float(wilcoxon.pvalue)) else None
        mean = statistics.fmean(numbers)
        sd = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        if sd == 0.0:
            result["mean_95ci"] = [mean, mean]
        else:
            sem = scipy_stats.sem(numbers)
            ci = scipy_stats.t.interval(0.95, len(numbers) - 1, loc=mean, scale=sem)
            result["mean_95ci"] = [float(ci[0]), float(ci[1])]
    return result


def _contrast(rows: Sequence[Mapping[str, Any]], treatment_arm: str, reference_arm: str, metric: str = "mse") -> Dict[str, Any]:
    index = {(str(row.get("dataset")), int(row.get("seed")), str(row.get("arm"))): row for row in rows}
    deltas: List[float] = []
    reference: List[float] = []
    treatment: List[float] = []
    coordinates: List[Tuple[str, int]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            ref = index.get((dataset, seed, reference_arm))
            trt = index.get((dataset, seed, treatment_arm))
            if ref is None or trt is None:
                raise ContractError(f"missing pair for {treatment_arm}-{reference_arm}: {dataset} seed {seed}")
            ref_value = _finite(ref.get(metric), f"{reference_arm} {metric} {dataset} {seed}")
            trt_value = _finite(trt.get(metric), f"{treatment_arm} {metric} {dataset} {seed}")
            reference.append(ref_value)
            treatment.append(trt_value)
            deltas.append(trt_value - ref_value)
            coordinates.append((dataset, seed))
    result = _paired_summary(deltas, reference, treatment)
    result["treatment_arm"] = treatment_arm
    result["reference_arm"] = reference_arm
    result["coordinates"] = [[dataset, seed] for dataset, seed in coordinates]
    return result


def _interaction(rows: Sequence[Mapping[str, Any]], metric: str = "mse") -> Dict[str, Any]:
    index = {(str(row.get("dataset")), int(row.get("seed")), str(row.get("arm"))): row for row in rows}
    values: List[float] = []
    coordinates: List[Tuple[str, int]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            def value(arm: str) -> float:
                row = index.get((dataset, seed, arm))
                if row is None:
                    raise ContractError(f"missing interaction arm {arm}: {dataset} seed {seed}")
                return _finite(row.get(metric), f"{arm} {metric} {dataset} {seed}")
            values.append(value("anchor_memory") - value("memory_only") - value("anchor_only") + value("plain"))
            coordinates.append((dataset, seed))
    result = _paired_summary(values)
    result.update({
        "contrast": "(anchor_memory - memory_only) - (anchor_only - plain)",
        "coordinates": [[dataset, seed] for dataset, seed in coordinates],
        "sign_interpretation": "negative=super-additive error reduction; positive=sub-additive/interfering effects",
    })
    return result


def _make_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    expected_arms = {item[1] for item in EXPECTED_PHASES}
    if len(rows) != EXPECTED_JOBS or {str(row.get("arm")) for row in rows} != expected_arms:
        raise ContractError(f"row matrix must contain exactly {EXPECTED_JOBS} rows and four arms")
    index: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset")), int(row.get("seed")), str(row.get("arm")))
        if key in index:
            raise ContractError(f"duplicate row coordinate: {key}")
        index[key] = row
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            coordinate_rows = [index.get((dataset, seed, arm)) for _, arm, _, _ in EXPECTED_PHASES]
            if any(item is None for item in coordinate_rows):
                raise ContractError(f"incomplete four-arm coordinate: {dataset} seed {seed}")
            typed = [item for item in coordinate_rows if item is not None]
            signatures = {_canonical(item.get("config_signature")) for item in typed}
            if len(signatures) != 1:
                raise ContractError(f"common config differs at {dataset} seed {seed}")
            sidecars = {item.get("sidecar_sha256") for item in typed}
            if len(sidecars) != 1:
                raise ContractError(f"sidecars differ at {dataset} seed {seed}")
            factors = {
                str(item.get("arm")): (
                    _explicit_bool(item.get("anchor"), f"{dataset} {seed} anchor"),
                    _explicit_bool(item.get("memory"), f"{dataset} {seed} memory"),
                )
                for item in typed
            }
            expected_factors = {arm: (anchor, memory) for _, arm, anchor, memory in EXPECTED_PHASES}
            if factors != expected_factors:
                raise ContractError(f"factor encoding differs at {dataset} seed {seed}")
            pairs.append({
                "dataset": dataset,
                "seed": seed,
                "mse": {str(item.get("arm")): _finite(item.get("mse"), f"{item.get('arm')} mse {dataset} {seed}") for item in typed},
                "mae": {str(item.get("arm")): item.get("mae") for item in typed},
                "sidecar_sha256": next(iter(sidecars)),
                "config_signature": next(iter(signatures)),
            })
    return pairs


def _stopping_decision(pairs: Sequence[Mapping[str, Any]], contrasts: Mapping[str, Any]) -> Dict[str, Any]:
    plain = [float(item["mse"]["plain"]) for item in pairs]
    combined = [float(item["mse"]["anchor_memory"]) for item in pairs]
    combined_delta = [b - a for a, b in zip(plain, combined)]
    by_dataset: Dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        deltas = [float(item["mse"]["anchor_memory"] - item["mse"]["plain"]) for item in subset]
        baseline = statistics.fmean(float(item["mse"]["plain"]) for item in subset)
        combined_mean = statistics.fmean(float(item["mse"]["anchor_memory"]) for item in subset)
        by_dataset[dataset] = {
            "combined_minus_plain_mean": statistics.fmean(deltas),
            "combined_vs_plain_ratio_percent": (combined_mean / baseline - 1.0) * 100.0,
            "wins_lower": sum(delta < 0 for delta in deltas),
            "all_worse": all(delta > 0 for delta in deltas),
        }
    pooled_ratio = statistics.fmean(combined) / statistics.fmean(plain) - 1.0
    stop = pooled_ratio > 0.01 or any(item["all_worse"] for item in by_dataset.values())
    interaction = contrasts["interaction"]["pooled"]
    interaction_negative_both = all(
        float(contrasts["interaction"]["by_dataset"][dataset]["mean"]) < 0.0
        for dataset in EXPECTED_DATASETS
    )
    interaction_ci_excludes_zero = all(
        (
            float(contrasts["interaction"]["by_dataset"][dataset]["mean_95ci"][1]) < 0.0
            or float(contrasts["interaction"]["by_dataset"][dataset]["mean_95ci"][0]) > 0.0
        )
        for dataset in EXPECTED_DATASETS
    )
    interaction_coherent = interaction_negative_both and interaction_ci_excludes_zero
    descriptive_screen_passes = not stop
    return {
        "criterion": "stop if combined arm worsens versus plain by >1% pooled or on all three seeds in either dataset",
        "combined_pooled_ratio_percent": pooled_ratio * 100.0,
        "combined_pooled_wins_lower": sum(delta < 0 for delta in combined_delta),
        "by_dataset": by_dataset,
        "interaction_negative_in_both_datasets": interaction_negative_both,
        "interaction_ci_excludes_zero_in_both_datasets": interaction_ci_excludes_zero,
        "interaction_coherent_for_follow_up": interaction_coherent,
        "meets_descriptive_screen": descriptive_screen_passes,
        "action": (
            "stop; do not stack anchor and memory or tune interaction from this screen"
            if stop else
            (
                "eligible only for a separately reviewed frozen replication; this does not authorize test selection"
                if interaction_coherent else
                "do not promote the combination: the no-harm screen passed, but the interaction is inconclusive or directionally inconsistent"
            )
        ),
    }


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], validation: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    normalized = [dict(row) for row in rows]
    if any(row.get("mse_source") not in (None, "val_loss") for row in normalized):
        raise ContractError("primary MSE source must be val_loss")
    pairs = _make_pairs(normalized)
    contrast_specs = {
        "anchor_at_memory_off": ("anchor_only", "plain"),
        "anchor_at_memory_on": ("anchor_memory", "memory_only"),
        "memory_at_anchor_off": ("memory_only", "plain"),
        "memory_at_anchor_on": ("anchor_memory", "anchor_only"),
    }
    pooled: Dict[str, Any] = {}
    by_dataset: Dict[str, Dict[str, Any]] = {dataset: {} for dataset in EXPECTED_DATASETS}
    for name, (treatment, reference) in contrast_specs.items():
        pooled[name] = _contrast(normalized, treatment, reference)
        for dataset in EXPECTED_DATASETS:
            subset = [row for row in normalized if row.get("dataset") == dataset]
            by_dataset[dataset][name] = _contrast_dataset(subset, treatment, reference)
    interaction_pooled = _interaction(normalized)
    interaction_by_dataset = {
        dataset: _interaction_dataset([row for row in normalized if row.get("dataset") == dataset])
        for dataset in EXPECTED_DATASETS
    }
    mae_available = all(
        row.get("mae") is not None and math.isfinite(float(row.get("mae")))
        for row in normalized
    )
    mae_contrasts: Optional[Dict[str, Any]] = None
    if mae_available:
        mae_pooled = {
            name: _contrast(normalized, treatment, reference, metric="mae")
            for name, (treatment, reference) in contrast_specs.items()
        }
        mae_by_dataset = {
            dataset: {
                name: _contrast_dataset(
                    [row for row in normalized if row.get("dataset") == dataset],
                    treatment,
                    reference,
                    metric="mae",
                )
                for name, (treatment, reference) in contrast_specs.items()
            }
            for dataset in EXPECTED_DATASETS
        }
        mae_interaction = _interaction(normalized, metric="mae")
        mae_interaction_by_dataset = {
            dataset: _interaction_dataset(
                [row for row in normalized if row.get("dataset") == dataset],
                metric="mae",
            )
            for dataset in EXPECTED_DATASETS
        }
        mae_contrasts = {
            "pooled": {**mae_pooled, "interaction": mae_interaction},
            "by_dataset": mae_by_dataset,
            "interaction": {"pooled": mae_interaction, "by_dataset": mae_interaction_by_dataset},
        }
    contrasts = {
        "pooled": {**pooled, "interaction": interaction_pooled},
        "by_dataset": by_dataset,
        "interaction": {"pooled": interaction_pooled, "by_dataset": interaction_by_dataset},
    }
    result = {
        "schema_version": 1,
        "status": "exploratory_mamba_acf_anchor_memory_interaction_screen_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan": str(plan.get("_path", "")),
        "plan_sha256": _sha256(Path(str(plan["_path"]))) if plan.get("_path") and Path(str(plan["_path"])).is_file() else None,
        "execution_context": {
            "plan_metadata_remains_draft": True,
            "artifact_execution": "completed post-hoc under the immutable draft plan and frozen source snapshot",
            "confirmatory": False,
            "test_metrics_read": False,
        },
        "phases": [name for name, _, _, _ in EXPECTED_PHASES],
        "rows": normalized,
        "pairs": pairs,
        "validation": list(validation or []),
        "metric_source": {
            "primary": EXPECTED_METRIC_SOURCE,
            "mse_canonical_field": "val_loss",
            "validation_mae_available_all_rows": mae_available,
            "test_metrics_read": False,
            "forecast_windows_are_not_inferential_replicates": True,
        },
        "invariants": {
            "paired_coordinates": len(pairs),
            "sidecar_hashes_equal_per_coordinate": True,
            "common_config_equal_after_excluding_two_treatments": True,
            "only_registered_treatments": ["mamba_acf_anchor_copy", "mamba_use_acf_memory"],
            "raw_static_train_only_prior": True,
            "factorial_cells_complete": True,
        },
        "contrasts": contrasts,
        "validation_mae_contrasts": mae_contrasts,
        "stopping_decision": _stopping_decision(pairs, contrasts),
        "evidence_boundary": "Validation-only paired descriptive evidence; interaction and p-values are exploratory and cannot establish a test-set or publication claim.",
    }
    return result


def _contrast_dataset(rows: Sequence[Mapping[str, Any]], treatment: str, reference: str, metric: str = "mse") -> Dict[str, Any]:
    index = {(int(row.get("seed")), str(row.get("arm"))): row for row in rows}
    ref_values: List[float] = []
    trt_values: List[float] = []
    for seed in EXPECTED_SEEDS:
        ref = index.get((seed, reference))
        trt = index.get((seed, treatment))
        if ref is None or trt is None:
            raise ContractError(f"missing dataset pair {treatment}-{reference} seed {seed}")
        ref_values.append(_finite(ref.get(metric), f"{reference} {metric} seed {seed}"))
        trt_values.append(_finite(trt.get(metric), f"{treatment} {metric} seed {seed}"))
    result = _paired_summary([b - a for a, b in zip(ref_values, trt_values)], ref_values, trt_values)
    result.update({"treatment_arm": treatment, "reference_arm": reference})
    return result


def _interaction_dataset(rows: Sequence[Mapping[str, Any]], metric: str = "mse") -> Dict[str, Any]:
    index = {(int(row.get("seed")), str(row.get("arm"))): row for row in rows}
    values: List[float] = []
    for seed in EXPECTED_SEEDS:
        def value(arm: str) -> float:
            row = index.get((seed, arm))
            if row is None:
                raise ContractError(f"missing interaction arm {arm} seed {seed}")
            return _finite(row.get(metric), f"{arm} {metric} seed {seed}")
        values.append(value("anchor_memory") - value("memory_only") - value("anchor_only") + value("plain"))
    result = _paired_summary(values)
    result.update({
        "contrast": "(anchor_memory - memory_only) - (anchor_only - plain)",
        "sign_interpretation": "negative=super-additive error reduction; positive=sub-additive/interfering effects",
    })
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MambaACF Anchor x Soft-ACF-Memory Interaction Screen",
        "",
        "> Exploratory validation-only 2x2 factorial screen. Test metrics are disabled; all contrasts are paired by dataset and seed.",
        "> The immutable plan metadata remains a draft; the 24 artifacts were executed post-hoc under its frozen source snapshot, so this is not preregistered or confirmatory evidence.",
        "",
        "Primary metric: best finite `val_loss` from `checkpoint/loss_logs.json`; lower is better.",
        "",
        "## Simple effects",
        "",
        "| Contrast | Dataset | n | mean MSE delta | relative ratio | wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, block in result["contrasts"]["pooled"].items():
        if name == "interaction":
            continue
        for dataset in ("pooled", *EXPECTED_DATASETS):
            value = block if dataset == "pooled" else result["contrasts"]["by_dataset"][dataset][name]
            lines.append(
                f"| {name} | {dataset} | {value['n']} | {_fmt(value['mean'])} | {_fmt(value.get('ratio_percent'), 3)}% | {value['wins_lower']}/{value['n']} |"
            )
    lines += [
        "",
        "## Interaction",
        "",
        "| Dataset | n | difference-in-differences | 95% CI | wins (<0) | p (descriptive) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("pooled", *EXPECTED_DATASETS):
        value = result["contrasts"]["interaction"]["pooled"] if dataset == "pooled" else result["contrasts"]["interaction"]["by_dataset"][dataset]
        ci = value.get("mean_95ci", [None, None])
        lines.append(
            f"| {dataset} | {value['n']} | {_fmt(value['mean'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | {value['wins_lower']}/{value['n']} | {_fmt(value.get('paired_t_two_sided_p'), 4)} |"
        )
    decision = result["stopping_decision"]
    lines += [
        "",
        "## Decision boundary",
        "",
        f"- {decision['criterion']}",
        f"- Combined-vs-plain pooled ratio: `{_fmt(decision['combined_pooled_ratio_percent'], 3)}%`; pooled wins: `{decision['combined_pooled_wins_lower']}/{len(result['pairs'])}`.",
        f"- Interaction coherent for follow-up: `{decision['interaction_coherent_for_follow_up']}` (negative in both datasets: `{decision['interaction_negative_in_both_datasets']}`; each dataset CI excludes zero: `{decision['interaction_ci_excludes_zero_in_both_datasets']}`).",
        f"- Action: **{decision['action']}**",
        "",
        "Negative interaction values mean the combined arm reduces validation error more than the sum of its two isolated effects. A positive value indicates interference or sub-additivity; it is not evidence that either mechanism is intrinsically invalid.",
        "",
        "The sidecar, common configuration, and factor-key invariants are checked before this readout is written. Missing or incomplete artifacts return `NOT_RUN` (exit code 3).",
        "",
        result["evidence_boundary"],
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/mamba_acf_anchor_memory_interaction_screen_plan_20260904.json"))
    parser.add_argument("--output-json", "--output_json", dest="output_json", type=Path, default=None)
    parser.add_argument("--output-md", "--output_md", dest="output_md", type=Path, default=None)
    args = parser.parse_args(argv)
    plan_path = _resolve(args.plan)
    try:
        plan = dict(_read_json(plan_path))
        plan["_path"] = str(plan_path)
        validate_plan_contract(plan)
        missing = _missing_training_artifacts(plan)
        if missing:
            preview = "; ".join(missing[:6])
            suffix = " ..." if len(missing) > 6 else ""
            print(f"NOT_RUN: registered anchor-memory interaction screen is incomplete ({preview}{suffix})", file=sys.stderr)
            return 3
        rows: List[Dict[str, Any]] = []
        validation: List[Dict[str, Any]] = []
        for phase_name, arm, anchor, memory in EXPECTED_PHASES:
            phase_rows, phase_meta = _load_phase(plan, plan_path, phase_name, arm, anchor, memory)
            rows.extend(phase_rows)
            validation.append(phase_meta)
        result = build_readout(plan, rows, validation)
    except (ContractError, NotRunError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"NOT_RUN: {exc}", file=sys.stderr)
        return 3
    output_json = _resolve(args.output_json or Path("artifacts/exploratory/mamba_acf_anchor_memory_interaction_screen_20260904/readout.json"))
    output_md = _resolve(args.output_md or Path("docs/research/mamba_acf_anchor_memory_interaction_screen_readout_20260904.md"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output_json": str(output_json),
        "output_md": str(output_md),
        "interaction_mean": result["contrasts"]["interaction"]["pooled"]["mean"],
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
