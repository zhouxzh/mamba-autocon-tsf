#!/usr/bin/env python3
"""Run an isolated, read-only checkpoint/ACF-sidecar replay.

The source phase roots are never modified.  Each job receives a temporary
copy of its checkpoint, config, and sidecar, then the live ``run.py`` is
invoked with ``is_training=0``.  The output is an exploratory compatibility
report, not a confirmation aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Support direct CLI execution from any working directory.  The replay tool is
# routinely invoked as ``python scripts/replay_mamba_acf_sidecars.py`` rather
# than imported as a package, so the repository root must be available before
# loading the shared, data-independent contract module.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.acf_relation_contract import mamba_acf_prior_required


DEFAULT_PHASE_NAMES = (
    "variable_plain_h96_screen",
    "variable_memory_h96_screen",
    "variable_memory_relation_h96_screen",
)
# Kept as a module-level value because the provenance/hash helpers use the
# selected phase set.  The CLI can replace it for an exploratory plan with a
# different, explicitly registered pair of phase names.
PHASE_NAMES = DEFAULT_PHASE_NAMES
METRIC_NAMES = ("mae", "mse", "rmse", "mape", "mspe")
METRIC_TOLERANCES = {
    "mae": {"rtol": 1e-6, "atol": 1e-7},
    "mse": {"rtol": 1e-6, "atol": 1e-7},
    "rmse": {"rtol": 1e-6, "atol": 1e-7},
    "mape": {"rtol": 1e-5, "atol": 1e-6},
    "mspe": {"rtol": 1e-5, "atol": 1e-6},
}
MODEL_EXECUTABLE_ALIASES = {
    # The training-plan runner records these user-facing arm aliases while
    # executing the shared MambaACF implementation.
    "MambaACFVariable": "MambaACF",
    "MambaACFVariableAutoCon": "MambaACFAutoCon",
    "MambaACFLiteAutoCon": "MambaACFAutoCon",
}
CHECKPOINT_ARTIFACT_NAMES = {"checkpoint.pth", "loss_logs.json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replay_tool_identity() -> Dict[str, str]:
    """Identify the exact replay implementation that produced an audit."""

    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} is not an integer".format(label)) from exc


def _resolve_dataset_path(config: Mapping[str, Any]) -> Path:
    root_value = config.get("root_path")
    data_value = config.get("data_path")
    if not str(root_value or "").strip() or not str(data_value or "").strip():
        raise ValueError("source config must declare root_path and data_path")
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    data = Path(str(data_value)).expanduser()
    path = data if data.is_absolute() else root / data
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("dataset file is missing: {}".format(path))
    return path


def _executable_model(record: Mapping[str, Any]) -> str:
    registered = str(record.get("model") or "").strip()
    if not registered:
        raise ValueError("manifest record has no model")
    return MODEL_EXECUTABLE_ALIASES.get(registered, registered)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object: {}".format(path))
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_environment() -> Dict[str, Any]:
    """Capture lightweight runtime facts without initializing CUDA."""
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _package_version("numpy"),
        "torch": _package_version("torch"),
        "mamba_ssm": _package_version("mamba-ssm"),
    }


def _canonical_lags(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    try:
        values = [int(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(item <= 0 for item in values):
        return None
    return sorted(set(values))


def _ordered_lags(config: Mapping[str, Any]) -> Optional[List[int]]:
    for key in ("mamba_acf_effective_lags", "mamba_acf_prior_lags", "mamba_acf_lags"):
        value = config.get(key)
        if isinstance(value, (list, tuple)):
            try:
                values = [int(item) for item in value]
            except (TypeError, ValueError, OverflowError):
                continue
            if values and all(item > 0 for item in values):
                return values
    return None


def _metric_matches(actual: Mapping[str, float], expected: Mapping[str, float]) -> bool:
    for name in METRIC_NAMES:
        tolerance = METRIC_TOLERANCES[name]
        difference = abs(float(actual[name]) - float(expected[name]))
        limit = tolerance["atol"] + tolerance["rtol"] * abs(float(expected[name]))
        if difference > limit:
            return False
    return True


def _phase_metadata(source_root: Path, phase_name: str) -> Dict[str, Any]:
    phase_root = source_root / phase_name
    metadata: Dict[str, Any] = {"phase": phase_name}
    for name in ("logs/job_manifest.jsonl", "provenance.json", "environment.json", "phase_validation.json"):
        path = phase_root / name
        metadata[name] = {"path": str(path), "sha256": _sha256(path)} if path.is_file() else None
    return metadata


def _live_source_hashes(source_root: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Hash source files named by the historical provenance snapshot when present."""
    snapshot_sha = next(
        (str(item.get("source_tree_sha256_before")) for item in records if item.get("source_tree_sha256_before")),
        None,
    )
    hashes: Dict[str, str] = {"run.py": _sha256(REPO_ROOT / "run.py")}
    if not snapshot_sha:
        return hashes
    for phase_name in PHASE_NAMES:
        provenance_path = source_root / phase_name / "provenance.json"
        if not provenance_path.is_file():
            continue
        payload = _read_json(provenance_path)
        source_files = payload.get("source_files", {})
        if not isinstance(source_files, Mapping):
            continue
        for relative in source_files:
            path = REPO_ROOT / str(relative)
            if path.is_file():
                hashes[str(relative)] = _sha256(path)
        break
    return hashes


def _manifest_records(phase_root: Path) -> List[Mapping[str, Any]]:
    path = phase_root / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError("manifest is missing: {}".format(path))
    records: List[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if isinstance(record, Mapping):
                    records.append(record)
    return records


def _source_artifact_paths(
    result_dir: Path, checkpoint_dir: Path, hashes: Mapping[str, Any]
) -> Dict[str, Path]:
    """Resolve the artifact names emitted by the training-plan manifest."""

    paths: Dict[str, Path] = {}
    for raw_name in hashes:
        name = str(raw_name)
        relative = Path(name)
        if not name or relative.name != name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid artifact name: {}".format(name))
        base = checkpoint_dir if name in CHECKPOINT_ARTIFACT_NAMES else result_dir
        paths[name] = base / name
    return paths


def _validate_source_artifacts(
    record: Mapping[str, Any],
    result_dir: Path,
    checkpoint_dir: Path,
    phase_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Recompute every source artifact hash and compare it to the manifest."""

    expected = record.get("artifact_hashes")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("manifest record has no artifact_hashes")
    expected_hashes = {str(name): str(value) for name, value in expected.items()}
    if any(not _is_sha256(value) for value in expected_hashes.values()):
        raise ValueError("manifest contains an invalid artifact hash")
    if phase_contract is not None:
        phase_hashes = phase_contract.get("artifact_hashes_by_job", {})
        job_id = str(record.get("job_id", ""))
        registered = phase_hashes.get(job_id) if isinstance(phase_hashes, Mapping) else None
        if not isinstance(registered, Mapping) or dict(registered) != expected_hashes:
            raise ValueError("manifest/phase_validation artifact hashes differ: {}".format(job_id))
    paths = _source_artifact_paths(result_dir, checkpoint_dir, expected_hashes)
    actual: Dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError("source artifact is missing: {}".format(path))
        actual[name] = _sha256(path)
        if actual[name].lower() != expected_hashes[name].lower():
            raise ValueError("source artifact hash mismatch: {}".format(name))
    return actual


def _source_config_identity(
    record: Mapping[str, Any],
    source_config: Mapping[str, Any],
    setting: str,
    sidecar_sha256: str,
) -> Dict[str, Any]:
    """Ensure the copied config cannot be detached from its manifest job."""

    job_id = str(record.get("job_id", ""))
    dataset = str(record.get("dataset", ""))
    expected_model = _executable_model(record)
    errors: List[str] = []
    if source_config.get("setting") != setting:
        errors.append("setting")
    if source_config.get("plan_job_id") != job_id:
        errors.append("plan_job_id")
    if source_config.get("data") != dataset:
        errors.append("data")
    if source_config.get("model") != expected_model:
        errors.append("model")
    for key in ("seed", "run_seed"):
        try:
            if _as_int(source_config.get(key), "config {}".format(key)) != _as_int(
                record.get("seed"), "manifest seed"
            ):
                errors.append(key)
        except ValueError:
            errors.append(key)
    declared_prior = str(source_config.get("mamba_acf_prior_sha256", "")).lower()
    if declared_prior != str(sidecar_sha256).lower():
        errors.append("mamba_acf_prior_sha256")
    if errors:
        raise ValueError(
            "source config identity mismatch for {}: {}".format(job_id, ", ".join(errors))
        )
    return {
        "setting": setting,
        "plan_job_id": job_id,
        "dataset": dataset,
        "model": expected_model,
        "seed": _as_int(record.get("seed"), "manifest seed"),
        "sidecar_sha256": str(sidecar_sha256),
    }


def _validate_source_snapshot(provenance: Mapping[str, Any]) -> None:
    """Check the immutable source snapshot referenced by phase provenance."""

    snapshot_value = provenance.get("source_snapshot")
    source_files = provenance.get("source_files")
    if not snapshot_value or not isinstance(source_files, Mapping):
        raise ValueError("phase provenance has no complete source snapshot")
    snapshot = Path(str(snapshot_value)).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError("phase source snapshot is missing: {}".format(snapshot))
    actual_files: Dict[str, str] = {}
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshot).as_posix()
        actual_files[relative] = _sha256(path)
    expected_files = {str(name): str(value) for name, value in source_files.items()}
    if actual_files != expected_files:
        raise ValueError("phase source snapshot changed")
    encoded = json.dumps(actual_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tree_hash = hashlib.sha256(encoded).hexdigest()
    if provenance.get("source_tree_sha256") != tree_hash:
        raise ValueError("phase source snapshot tree hash mismatch")


def _phase_replay_contract(phase_root: Path) -> Dict[str, Any]:
    """Load and validate the strict phase files required by a data-bound replay."""

    phase_name = phase_root.name
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    provenance_path = phase_root / "provenance.json"
    validation_path = phase_root / "phase_validation.json"
    for path in (manifest_path, provenance_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError("phase contract file is missing: {}".format(path))
    manifest_sha256 = _sha256(manifest_path)
    provenance_sha256 = _sha256(provenance_path)
    validation_sha256 = _sha256(validation_path)
    provenance = _read_json(provenance_path)
    validation = _read_json(validation_path)
    if provenance.get("schema") != 1 or provenance.get("phase") != phase_name:
        raise ValueError("phase provenance identity mismatch: {}".format(phase_name))
    provenance_id = str(provenance.get("provenance_id", ""))
    if not provenance_id:
        raise ValueError("phase provenance has no provenance_id: {}".format(phase_name))
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ValueError("phase validation is not valid: {}".format(phase_name))
    if validation.get("manifest_sha256") != manifest_sha256:
        raise ValueError("phase validation manifest hash mismatch: {}".format(phase_name))
    if validation.get("provenance_id") != provenance_id:
        raise ValueError("phase validation provenance mismatch: {}".format(phase_name))
    for key, path in (("manifest", manifest_path), ("provenance", provenance_path)):
        declared = validation.get(key)
        if declared and Path(str(declared)).expanduser().resolve() != path.resolve():
            raise ValueError("phase validation points to another {}: {}".format(key, phase_name))
    plan_value = provenance.get("plan")
    plan_sha256 = str(provenance.get("plan_sha256", ""))
    plan_path = Path(str(plan_value)).expanduser().resolve() if plan_value else None
    if plan_path is None or not plan_path.is_file() or not _is_sha256(plan_sha256):
        raise ValueError("phase provenance has no usable plan: {}".format(phase_name))
    if _sha256(plan_path).lower() != plan_sha256.lower():
        raise ValueError("phase provenance plan hash mismatch: {}".format(phase_name))
    _validate_source_snapshot(provenance)
    dataset_hashes = provenance.get("dataset_hashes")
    if not isinstance(dataset_hashes, Mapping) or not dataset_hashes:
        raise ValueError("phase provenance has no dataset hashes: {}".format(phase_name))
    if any(not _is_sha256(value) for value in dataset_hashes.values()):
        raise ValueError("phase provenance contains an invalid dataset hash: {}".format(phase_name))
    records = _manifest_records(phase_root)
    record_ids = [str(item.get("job_id", "")) for item in records]
    if any(not item for item in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("phase manifest has missing or duplicate job IDs: {}".format(phase_name))
    if any(item.get("phase", phase_name) != phase_name or item.get("status") != "ok" for item in records):
        raise ValueError("phase manifest contains a non-successful record: {}".format(phase_name))
    hash_map = validation.get("artifact_hashes_by_job")
    if not isinstance(hash_map, Mapping) or set(hash_map) != set(record_ids):
        raise ValueError("phase validation artifact map is incomplete: {}".format(phase_name))
    return {
        "phase": phase_name,
        "manifest_path": str(manifest_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "validation_path": str(validation_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "provenance_sha256": provenance_sha256,
        "validation_sha256": validation_sha256,
        "provenance_id": provenance_id,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "dataset_hashes": {str(name): str(value) for name, value in dataset_hashes.items()},
        "artifact_hashes_by_job": hash_map,
    }


def _flag_value(command: Sequence[Any], flag: str) -> Optional[str]:
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _replace_flag(command: List[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise ValueError("manifest command has no {}".format(flag)) from exc
    if index + 1 >= len(command):
        raise ValueError("manifest command has no value for {}".format(flag))
    command[index + 1] = value


def _set_or_append_flag(command: List[str], flag: str, value: str) -> None:
    """Set a replay-only flag without allowing duplicate CLI values."""

    if flag in command:
        _replace_flag(command, flag, value)
    else:
        command.extend([flag, value])


def _source_result_dir(record: Mapping[str, Any]) -> Path:
    expected = record.get("expected_artifact")
    if not expected:
        raise ValueError("manifest record has no expected_artifact")
    result_dir = Path(str(expected)).resolve().parent
    if not result_dir.is_dir():
        raise FileNotFoundError("source result directory is missing: {}".format(result_dir))
    return result_dir


def _source_checkpoint_dir(record: Mapping[str, Any], setting: str) -> Path:
    configured = _flag_value(record.get("command", []), "--checkpoints")
    if not configured:
        raise ValueError("manifest command has no --checkpoints")
    checkpoint_dir = Path(configured).expanduser().resolve() / setting
    if not (checkpoint_dir / "checkpoint.pth").is_file():
        raise FileNotFoundError("source checkpoint is missing: {}".format(checkpoint_dir))
    return checkpoint_dir


def _copy_inputs(record: Mapping[str, Any], job_root: Path) -> tuple[Path, Path, str]:
    source_result = _source_result_dir(record)
    config = _read_json(source_result / "config.json")
    setting = str(config.get("setting", ""))
    if not setting:
        raise ValueError("source config has no setting: {}".format(source_result))
    source_checkpoint = _source_checkpoint_dir(record, setting)
    result_dir = job_root / "results" / setting
    checkpoint_dir = job_root / "checkpoints" / setting
    result_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "acf_prior.npz"):
        source = source_result / name
        if not source.is_file():
            raise FileNotFoundError("source sidecar input is missing: {}".format(source))
        shutil.copy2(source, result_dir / name)
    shutil.copy2(source_checkpoint / "checkpoint.pth", checkpoint_dir / "checkpoint.pth")
    loss_logs = source_checkpoint / "loss_logs.json"
    if loss_logs.is_file():
        shutil.copy2(loss_logs, checkpoint_dir / loss_logs.name)
    return result_dir, checkpoint_dir, setting


def _command_sha256(command: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            [str(item) for item in command],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _metrics(path: Path) -> Dict[str, float]:
    payload = _read_json(path)
    section = payload.get("scaled")
    if not isinstance(section, Mapping):
        raise ValueError("stream metrics has no scaled section: {}".format(path))
    output: Dict[str, float] = {}
    for name in METRIC_NAMES:
        value = float(section[name])
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite {} in {}".format(name, path))
        output[name] = value
    return output


def _sidecar_replay_contract_satisfied(result: Mapping[str, Any]) -> bool:
    """Require sidecar loading only when the replayed branch consumes it."""

    # The harness copies the sidecar as an immutable input for every historical
    # record, so its identity must remain unchanged even for ACF-off arms.
    if not bool(result.get("sidecar_hash_equal")):
        return False
    # ACF-off evaluation is intentionally independent of the sidecar, so a
    # load message is itself a contract violation.  A live ACF branch must
    # prove that the loader ran.
    # A replay must never refit the train prior: required branches consume the
    # copied sidecar, while ACF-off branches remain independent of both the
    # sidecar and the train split.
    if bool(result.get("prior_fit_message")):
        return False
    if bool(result.get("source_acf_prior_required")):
        return bool(result.get("sidecar_load_message"))
    return not bool(result.get("sidecar_load_message"))


def _replay_config_identity(
    source_config: Mapping[str, Any],
    replay_config: Mapping[str, Any],
    *,
    source_sidecar_sha256: str,
    source_prior_required: bool,
) -> tuple[bool, List[str]]:
    """Compare the effective forecast contract after an isolated replay.

    ``run.py`` persists many output paths and evaluation-only fields that are
    expected to differ in the temporary replay directory.  Comparing raw
    JSON would therefore reject valid replays.  The normalized runtime
    contract is the same construction-time identity used by live evaluation;
    the small set of job/sidecar fields below binds that contract to the
    historical manifest record.
    """

    mismatches: List[str] = []
    for key in ("setting", "plan_job_id", "data", "model"):
        if source_config.get(key) != replay_config.get(key):
            mismatches.append(key)
    for key in ("seed", "run_seed"):
        try:
            if _as_int(source_config.get(key), "source config {}".format(key)) != _as_int(
                replay_config.get(key), "replay config {}".format(key)
            ):
                mismatches.append(key)
        except ValueError:
            mismatches.append(key)

    source_sha = str(source_config.get("mamba_acf_prior_sha256", "")).lower()
    replay_sha = str(replay_config.get("mamba_acf_prior_sha256", "")).lower()
    expected_sha = str(source_sidecar_sha256).lower()
    if source_sha != expected_sha or replay_sha != expected_sha:
        mismatches.append("mamba_acf_prior_sha256")

    try:
        # Import lazily so read-only manifest validation and unit tests do not
        # initialize model modules before a subprocess replay is requested.
        from run import _mamba_runtime_contract_payload

        source_contract = _mamba_runtime_contract_payload(source_config)
        replay_contract = _mamba_runtime_contract_payload(replay_config)
        if source_contract is None or replay_contract is None:
            mismatches.append("mamba_runtime_contract")
        else:
            for key in sorted(set(source_contract) | set(replay_contract)):
                if source_contract.get(key) != replay_contract.get(key):
                    mismatches.append("mamba_runtime_contract." + key)
    except (ImportError, TypeError, ValueError, AttributeError):
        mismatches.append("mamba_runtime_contract")

    try:
        replay_prior_required = bool(mamba_acf_prior_required(replay_config))
    except (TypeError, ValueError, AttributeError):
        replay_prior_required = not source_prior_required
    if replay_prior_required != bool(source_prior_required):
        mismatches.append("mamba_acf_prior_required")
    return not mismatches, sorted(set(mismatches))


def _run_job(
    record: Mapping[str, Any],
    phase_root: Path,
    output_root: Path,
    timeout: int,
    phase_contract: Optional[Mapping[str, Any]] = None,
    frozen_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    job_id = str(record.get("job_id", ""))
    if not job_id:
        raise ValueError("manifest record has no job_id")
    if record.get("status") != "ok":
        raise ValueError("manifest record is not successful: {}".format(job_id))
    if record.get("phase", phase_root.name) != phase_root.name:
        raise ValueError("manifest record phase mismatch: {}".format(job_id))
    source_command = [str(item) for item in record.get("command", [])]
    if record.get("command_sha256") != _command_sha256(source_command):
        raise ValueError("manifest command hash mismatch: {}".format(job_id))
    source_result = _source_result_dir(record)
    source_config = _read_json(source_result / "config.json")
    setting = str(source_config.get("setting", ""))
    if not setting:
        raise ValueError("source config has no setting: {}".format(source_result))
    source_checkpoint = _source_checkpoint_dir(record, setting)
    source_artifact_hashes = _validate_source_artifacts(
        record, source_result, source_checkpoint, phase_contract
    )
    source_sidecar_sha256 = source_artifact_hashes.get("acf_prior.npz")
    if source_sidecar_sha256 is None:
        raise ValueError("manifest artifact hashes omit acf_prior.npz: {}".format(job_id))
    config_identity = _source_config_identity(
        record, source_config, setting, source_sidecar_sha256
    )
    source_prior_required = mamba_acf_prior_required(source_config)
    dataset_path = _resolve_dataset_path(source_config)
    dataset_sha256 = _sha256(dataset_path)
    if phase_contract is None:
        raise ValueError("replay requires a validated phase contract: {}".format(phase_root.name))
    expected_dataset_sha = phase_contract["dataset_hashes"].get(str(record.get("dataset", "")))
    if not expected_dataset_sha:
        raise ValueError("phase provenance has no hash for dataset: {}".format(record.get("dataset")))
    if dataset_sha256.lower() != str(expected_dataset_sha).lower():
        raise ValueError("dataset hash mismatch: {}".format(dataset_path))
    if frozen_record is not None:
        expected_input_hashes = frozen_record.get("input_hashes")
        if not isinstance(expected_input_hashes, Mapping):
            raise ValueError("frozen replay record has no input hashes: {}".format(job_id))
        for name, actual in source_artifact_hashes.items():
            frozen = expected_input_hashes.get(name)
            if frozen is None or str(frozen).lower() != actual.lower():
                raise ValueError("source input changed after replay freeze: {}".format(name))
        if frozen_record.get("source_dataset_sha256") != dataset_sha256:
            raise ValueError("dataset changed after replay freeze: {}".format(dataset_path))
        if frozen_record.get("source_provenance_sha256") != phase_contract.get("provenance_sha256"):
            raise ValueError("phase provenance changed after replay freeze: {}".format(phase_root.name))
    job_root = output_root / str(record.get("phase", phase_root.name)) / job_id
    job_root.mkdir(parents=True, exist_ok=False)
    result_dir, checkpoint_dir, setting = _copy_inputs(record, job_root)
    copied_hashes = {
        "config.json": _sha256(result_dir / "config.json"),
        "acf_prior.npz": _sha256(result_dir / "acf_prior.npz"),
        "checkpoint.pth": _sha256(checkpoint_dir / "checkpoint.pth"),
    }
    for name, copied in copied_hashes.items():
        if copied != source_artifact_hashes.get(name):
            raise ValueError("copied replay input hash mismatch: {}".format(name))
    command = source_command
    if len(command) < 3:
        raise ValueError("manifest command is too short: {}".format(job_id))
    command[2] = str(REPO_ROOT / "run.py")
    _replace_flag(command, "--checkpoints", str(job_root / "checkpoints"))
    _replace_flag(command, "--results_dir", str(job_root / "results"))
    _replace_flag(command, "--test_results_dir", str(job_root / "test_results"))
    _replace_flag(command, "--result_log", str(job_root / "results.txt"))
    _set_or_append_flag(command, "--is_training", "0")
    _set_or_append_flag(command, "--setting_override", setting)
    command_sha256 = _command_sha256(command)
    log_path = job_root / "replay.log"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    started_monotonic = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        log_path.write_text(completed.stdout or "", encoding="utf-8")
        returncode = int(completed.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(output, encoding="utf-8")
        returncode = None
        timed_out = True
    elapsed_sec = round(time.monotonic() - started_monotonic, 3)
    finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    replay_result = job_root / "results" / setting
    source_stream = source_result / "stream_metrics.json"
    replay_stream = replay_result / "stream_metrics.json"
    source_sidecar = source_result / "acf_prior.npz"
    replay_sidecar = replay_result / "acf_prior.npz"
    result: Dict[str, Any] = {
        "phase": str(record.get("phase", phase_root.name)),
        "job_id": job_id,
        "dataset": record.get("dataset"),
        "seed": record.get("seed"),
        "setting": setting,
        "returncode": returncode,
        "timed_out": timed_out,
        "log": str(log_path),
        "source_snapshot_sha256": record.get("source_tree_sha256_before"),
        "source_command_sha256": record.get("command_sha256"),
        "replay_command_sha256": command_sha256,
        "replay_command": command,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_sec": elapsed_sec,
        "source_sidecar_sha256": source_sidecar_sha256,
        "replay_sidecar_sha256": _sha256(replay_sidecar) if replay_sidecar.is_file() else None,
        "sidecar_hash_equal": replay_sidecar.is_file() and source_sidecar_sha256 == _sha256(replay_sidecar),
        "source_acf_prior_required": source_prior_required,
        "sidecar_expected_to_load": source_prior_required,
        "source_config_is_training": source_config.get("is_training"),
        "source_artifact_hashes": source_artifact_hashes,
        "source_effective_lags": _ordered_lags(source_config),
        "source_config_identity": config_identity,
        "source_config_identity_equal": False,
        "source_config_identity_mismatches": ["replay_config_not_read"],
        "source_dataset_path": str(dataset_path),
        "source_dataset_sha256": dataset_sha256,
        "source_dataset_expected_sha256": str(expected_dataset_sha),
        "dataset_hash_equal": True,
        "source_provenance_id": phase_contract.get("provenance_id"),
        "source_provenance_sha256": phase_contract.get("provenance_sha256"),
        "source_phase_validation_sha256": phase_contract.get("validation_sha256"),
        "source_input_hashes": copied_hashes,
        "copied_input_hashes_equal": True,
    }
    if returncode == 0 and replay_stream.is_file():
        source_metrics = _metrics(source_stream)
        replay_metrics = _metrics(replay_stream)
        result["source_scaled"] = source_metrics
        result["replay_scaled"] = replay_metrics
        result["absolute_differences"] = {
            name: abs(replay_metrics[name] - source_metrics[name])
            for name in METRIC_NAMES
        }
        replay_config = _read_json(replay_result / "config.json")
        log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        result["replay_is_training"] = replay_config.get("is_training")
        identity_equal, identity_mismatches = _replay_config_identity(
            source_config,
            replay_config,
            source_sidecar_sha256=source_sidecar_sha256,
            source_prior_required=source_prior_required,
        )
        result["source_config_identity_equal"] = identity_equal
        result["source_config_identity_mismatches"] = identity_mismatches
        result["sidecar_load_message"] = "sidecar" in log_text and "loaded" in log_text
        result["prior_fit_message"] = "train-only prior:" in log_text
        result["replay_effective_lags"] = _ordered_lags(replay_config)
        source_lags = _canonical_lags(result.get("source_effective_lags"))
        replay_lags = _canonical_lags(result.get("replay_effective_lags"))
        result["lag_set_equal"] = (
            source_lags is not None
            and replay_lags is not None
            and source_lags == replay_lags
        )
        result["lag_order_normalized"] = (
            result.get("source_effective_lags") is not None
            and result.get("replay_effective_lags") is not None
            and result.get("source_effective_lags") != result.get("replay_effective_lags")
            and source_lags == replay_lags
        )
        result["contract_status"] = (
            "pass"
            if result["replay_is_training"] == 0
            and _sidecar_replay_contract_satisfied(result)
            and result["lag_set_equal"]
            and result["source_config_identity_equal"]
            and result["dataset_hash_equal"]
            and result["copied_input_hashes_equal"]
            else "fail"
        )
        result["metric_status"] = "match" if _metric_matches(replay_metrics, source_metrics) else "drift"
        result["status"] = (
            "ok"
            if result["contract_status"] == "pass" and result["metric_status"] == "match"
            else "metric_drift"
            if result["contract_status"] == "pass"
            else "contract_error"
        )
    else:
        result["contract_status"] = "fail"
        result["metric_status"] = "unavailable"
        result["status"] = "timeout" if timed_out else "process_error"
    result["output_artifact_hashes"] = {
        name: _sha256(path)
        for name, path in {
            "config.json": replay_result / "config.json",
            "stream_metrics.json": replay_stream,
            "acf_prior.npz": replay_sidecar,
            "checkpoint.pth": checkpoint_dir / "checkpoint.pth",
        }.items()
        if path.is_file()
    }
    return result


def _freeze_record(
    phase_root: Path,
    record: Mapping[str, Any],
    phase_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve and hash every input before the first replay subprocess starts."""

    job_id = str(record.get("job_id", ""))
    if not job_id:
        raise ValueError("manifest record has no job_id")
    if record.get("status") != "ok" or record.get("phase", phase_root.name) != phase_root.name:
        raise ValueError("manifest record is not a successful {} job: {}".format(phase_root.name, job_id))
    source_result = _source_result_dir(record)
    if not _inside(source_result, phase_root / "results"):
        raise ValueError("source result directory is outside phase results: {}".format(source_result))
    source_config = _read_json(source_result / "config.json")
    setting = str(source_config.get("setting", ""))
    if not setting:
        raise ValueError("source config has no setting: {}".format(source_result))
    if source_result.name != setting:
        raise ValueError("source config setting does not name its result directory: {}".format(job_id))
    source_checkpoint = _source_checkpoint_dir(record, setting)
    if not _inside(source_checkpoint, phase_root / "checkpoints"):
        raise ValueError(
            "source checkpoint directory is outside phase checkpoints: {}".format(source_checkpoint)
        )
    source_command = [str(item) for item in record.get("command", [])]
    if len(source_command) < 3:
        raise ValueError("manifest command is too short: {}".format(job_id))
    if record.get("command_sha256") != _command_sha256(source_command):
        raise ValueError("manifest command hash mismatch: {}".format(job_id))
    artifact_hashes = _validate_source_artifacts(
        record, source_result, source_checkpoint, phase_contract
    )
    sidecar_sha256 = artifact_hashes.get("acf_prior.npz")
    if sidecar_sha256 is None:
        raise ValueError("manifest artifact hashes omit acf_prior.npz: {}".format(job_id))
    config_identity = _source_config_identity(record, source_config, setting, sidecar_sha256)
    dataset_path = _resolve_dataset_path(source_config)
    dataset_sha256 = _sha256(dataset_path)
    expected_dataset_sha = phase_contract["dataset_hashes"].get(str(record.get("dataset", "")))
    if not expected_dataset_sha:
        raise ValueError("phase provenance has no hash for dataset: {}".format(record.get("dataset")))
    if dataset_sha256.lower() != str(expected_dataset_sha).lower():
        raise ValueError("dataset hash mismatch: {}".format(dataset_path))
    replay_template = list(source_command)
    replay_template[2] = "{repo_root}/run.py"
    _replace_flag(replay_template, "--checkpoints", "{job_root}/checkpoints")
    _replace_flag(replay_template, "--results_dir", "{job_root}/results")
    _replace_flag(replay_template, "--test_results_dir", "{job_root}/test_results")
    _replace_flag(replay_template, "--result_log", "{job_root}/results.txt")
    _set_or_append_flag(replay_template, "--is_training", "0")
    _set_or_append_flag(replay_template, "--setting_override", setting)
    return {
        "phase": str(record.get("phase", phase_root.name)),
        "job_id": job_id,
        "dataset": record.get("dataset"),
        "seed": record.get("seed"),
        "setting": setting,
        "source_command": source_command,
        "source_command_sha256": record.get("command_sha256"),
        "replay_command_template": replay_template,
        "source_result_dir": str(source_result),
        "source_checkpoint_dir": str(source_checkpoint),
        "source_tree_sha256": record.get("source_tree_sha256_before"),
        "source_config_sha256": artifact_hashes["config.json"],
        "source_effective_lags": _ordered_lags(source_config),
        "source_artifact_hashes": artifact_hashes,
        "input_hashes": artifact_hashes,
        "source_config_identity": config_identity,
        "source_dataset_path": str(dataset_path),
        "source_dataset_sha256": dataset_sha256,
        "source_dataset_expected_sha256": str(expected_dataset_sha),
        "source_provenance_id": phase_contract.get("provenance_id"),
        "source_provenance_sha256": phase_contract.get("provenance_sha256"),
        "source_phase_validation_sha256": phase_contract.get("validation_sha256"),
        "source_manifest_sha256": phase_contract.get("manifest_sha256"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    records = report.get("records", [])
    lines = [
        "# MambaACF Standalone Sidecar Replay",
        "",
        "> Exploratory compatibility replay only. It is not confirmation evidence or a publication claim.",
        "",
        "- Source phase root: `{}`".format(report.get("source_root")),
        "- Replay tool SHA-256: `{}`".format(
            report.get("replay_tool", {}).get("sha256")
        ),
        "- Live `run.py` SHA-256: `{}`".format(report.get("live_run_sha256")),
        "- Jobs: `{}/{}` contract and metric checks passed.".format(
            report.get("successful_jobs"), report.get("job_count")
        ),
        "- Contract passes: `{}/{}`; metric matches: `{}/{}`.".format(
            report.get("contract_successful_jobs"),
            report.get("job_count"),
            report.get("metric_matching_jobs"),
            report.get("job_count"),
        ),
        "- Frozen metric tolerances: `{}`.".format(report.get("metric_tolerances")),
        "- Each job was evaluated with `is_training=0` in an isolated output directory; source phase roots were not modified.",
        "- Before execution, the replay validates the phase provenance, immutable source snapshot, registered plan, dataset SHA-256, source config identity, and every declared input artifact hash.",
        "",
        "| Phase | Dataset | Seed | Contract | Metric | MSE abs diff | MAE abs diff | Sidecar hash |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    for item in records:
        differences = item.get("absolute_differences", {})
        lines.append(
            "| `{}` | {} | {} | `{}` | `{}` | {:.3e} | {:.3e} | {} |".format(
                item.get("phase"),
                item.get("dataset"),
                item.get("seed"),
                item.get("contract_status"),
                item.get("metric_status"),
                float(differences.get("mse", float("nan"))),
                float(differences.get("mae", float("nan"))),
                "ok" if item.get("sidecar_hash_equal") else "FAIL",
            )
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This replay tests the current loader against historical checkpoints and sidecars. It does not rerun training under the live tree.",
        "- Small floating-point differences are expected for GPU accumulation; the JSON report retains per-metric absolute differences.",
        "- `contract_status` always checks the copied sidecar hash; a live ACF-dependent branch must emit a sidecar-load log, while ACF-off must emit neither a load nor a train-only-prior-fit log. It also covers evaluation mode, canonical lag-set equality, normalized runtime/config identity, dataset hash, and copied-input hashes. `metric_status` applies the frozen per-metric tolerances.",
        "- `replay_manifest.json` freezes inputs and commands before execution; `replay_results.jsonl` is appended after every job for interruption-safe audit.",
        "- The replay remains exploratory and is not merged into the registered 884-job confirmation matrix.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase", action="append", dest="phases",
        help="phase name to replay; repeat for a multi-phase plan (defaults to the live screen phases)",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    global PHASE_NAMES
    if args.phases:
        if len(set(args.phases)) != len(args.phases):
            raise SystemExit("--phase names must be unique")
        PHASE_NAMES = tuple(str(item) for item in args.phases)
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit("source root is missing: {}".format(source_root))
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("output root is non-empty; choose a new root: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    phase_contracts: Dict[str, Dict[str, Any]] = {}
    all_records: List[tuple[Path, Mapping[str, Any]]] = []
    for phase_name in PHASE_NAMES:
        phase_root = source_root / phase_name
        phase_contract = _phase_replay_contract(phase_root)
        phase_contracts[phase_name] = phase_contract
        for record in _manifest_records(phase_root):
            all_records.append((phase_root, record))
    all_records.sort(key=lambda item: (str(item[1].get("phase")), str(item[1].get("job_id"))))
    if args.limit is not None:
        all_records = all_records[: max(0, int(args.limit))]
    frozen_records = [
        _freeze_record(phase_root, record, phase_contracts[phase_root.name])
        for phase_root, record in all_records
    ]
    frozen_by_key = {
        (str(item["phase"]), str(item["job_id"])): item for item in frozen_records
    }
    plan_path = source_root.parent.parent.parent / "docs" / "research" / "mamba_acf_sidecar_replay_plan.json"
    if not plan_path.is_file():
        plan_path = REPO_ROOT / "docs" / "research" / "mamba_acf_sidecar_replay_plan.json"
    source_snapshot_shas = sorted({
        str(item.get("source_tree_sha256"))
        for item in frozen_records
        if item.get("source_tree_sha256")
    })
    manifest: Dict[str, Any] = {
        "schema_version": 4,
        "kind": "mamba_acf_standalone_sidecar_replay_manifest",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_root": str(source_root),
        "source_plan": str(plan_path),
        "source_plan_sha256": _sha256(plan_path) if plan_path.is_file() else None,
        "source_snapshot_sha256": source_snapshot_shas,
        "phase_metadata": [_phase_metadata(source_root, phase_name) for phase_name in PHASE_NAMES],
        "phase_contracts": phase_contracts,
        "replay_tool": _replay_tool_identity(),
        "live_run": str(REPO_ROOT / "run.py"),
        "live_run_sha256": _sha256(REPO_ROOT / "run.py"),
        "live_source_hashes": _live_source_hashes(source_root, [record for _, record in all_records]),
        "environment": _runtime_environment(),
        "metric_tolerances": METRIC_TOLERANCES,
        "invocation": {
            "timeout_sec": max(1, int(args.timeout)),
            "limit": args.limit,
            "python": sys.executable,
            "argv": [str(item) for item in (argv if argv is not None else sys.argv[1:])],
        },
        "job_count": len(frozen_records),
        "records": frozen_records,
    }
    manifest_path = output_root / "replay_manifest.json"
    results_path = output_root / "replay_results.jsonl"
    _write_json(manifest_path, manifest)
    results_path.touch()
    records: List[Dict[str, Any]] = []
    for phase_root, record in all_records:
        try:
            phase_name = phase_root.name
            item = _run_job(
                record,
                phase_root,
                output_root,
                max(1, int(args.timeout)),
                phase_contract=phase_contracts[phase_name],
                frozen_record=frozen_by_key[(phase_name, str(record.get("job_id")))],
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            item = {
                "phase": record.get("phase", phase_root.name),
                "job_id": record.get("job_id"),
                "dataset": record.get("dataset"),
                "seed": record.get("seed"),
                "status": "setup_error",
                "contract_status": "fail",
                "metric_status": "unavailable",
                "error": str(exc),
            }
        records.append(item)
        _append_jsonl(results_path, item)
        print("{} | {} | {}".format(item.get("phase"), item.get("job_id"), item.get("status")), flush=True)
    successful = sum(item.get("status") == "ok" for item in records)
    contract_successful = sum(item.get("contract_status") == "pass" for item in records)
    metric_matching = sum(item.get("metric_status") == "match" for item in records)
    metric_drift = sum(item.get("metric_status") == "drift" for item in records)
    report: Dict[str, Any] = {
        "schema_version": 4,
        "status": "exploratory_standalone_sidecar_replay_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "results_jsonl": str(results_path),
        "replay_tool": manifest["replay_tool"],
        "live_run": str(REPO_ROOT / "run.py"),
        "live_run_sha256": _sha256(REPO_ROOT / "run.py"),
        "environment": manifest["environment"],
        "phase_contracts": phase_contracts,
        "metric_tolerances": METRIC_TOLERANCES,
        "job_count": len(records),
        "successful_jobs": successful,
        "contract_successful_jobs": contract_successful,
        "metric_matching_jobs": metric_matching,
        "metric_drift_jobs": metric_drift,
        "records": records,
        "limitations": [
            "historical checkpoints were not retrained under the live source tree",
            "GPU accumulation can produce small floating-point differences",
            "dataset, phase provenance, source config identity, and manifest artifact hashes are checked before replay",
            "this replay is exploratory and not part of the 884-job confirmation matrix",
        ],
    }
    _write_json(output_root / "replay_report.json", report)
    (output_root / "replay_report.md").write_text(_markdown(report), encoding="utf-8")
    print("wrote {}".format(output_root / "replay_report.json"))
    print("successful {}/{}".format(successful, len(records)))
    return 0 if successful == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
