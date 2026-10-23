#!/usr/bin/env python3
"""Strict, read-only readout for the variable-token ACF/AutoCon screen.

The screen has three registered phases and two pre-declared paired contrasts:

* variable-token MambaACF with train-only memory minus the plain arm;
* variable-token memory + signed relation objective minus variable-token memory.

This reader treats a seed/run as the inferential unit.  It binds every row to
the registered plan, immutable phase provenance, validated manifest, and all
per-job artifact hashes before reading the streaming metrics.  It writes only
the requested analysis JSON and Markdown outputs; training artifacts are never
modified.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # SciPy is available in the project environment, but keep the reader portable.
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - minimal environments only.
    scipy_stats = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES: Tuple[Tuple[str, str], ...] = (
    ("variable_plain_h96_screen", "plain"),
    ("variable_memory_h96_screen", "memory"),
    ("variable_memory_relation_h96_screen", "memory_relation"),
)
EXPECTED_DATASETS: Tuple[str, ...] = ("ETTh1", "traffic")
EXPECTED_SEEDS: Tuple[int, ...] = (2021, 2022, 2023)
EXPECTED_FEATURES = "M"
EXPECTED_HORIZON = 96
EXPECTED_JOBS_PER_PHASE = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)
RESULT_ARTIFACT_NAMES = {
    "config.json",
    "metrics.npy",
    "original_metrics.npy",
    "channel_metrics.npy",
    "stream_metrics.json",
    "stream_samples.npz",
    "pred.npy",
    "true.npy",
    "training_complete.json",
    "acf_prior.npz",
}
CHECKPOINT_ARTIFACT_NAMES = {"checkpoint.pth", "loss_logs.json"}
RELATION_LOG_KEYS = (
    "acf_relation_global_valid_fraction",
    "acf_relation_global_candidate_fraction",
    "acf_relation_global_candidate_active_anchor_fraction",
    "acf_relation_global_positive_count_mean",
)
SUPPORTED_PLAN_STATUSES = frozenset({
    "exploratory_variable_memory_relation_screen_not_confirmatory",
    "exploratory_current_code_sidecar_replay_not_confirmatory",
    "exploratory_live_sidecar_screen_not_confirmatory",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_file_map(root: Path) -> Tuple[Dict[str, str], str]:
    files: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[path.relative_to(root).as_posix()] = _sha256(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


def _live_source_mismatches(recorded_files: Mapping[str, Any]) -> List[str]:
    """Compare the live tree with the exact files captured in provenance.

    The phase snapshot remains the authority for historical metrics.  This
    comparison is informational: it makes later source drift visible in a
    regenerated readout without rewriting or invalidating the immutable
    phase record.
    """

    if not isinstance(recorded_files, Mapping):
        return ["<source_files metadata missing>"]
    mismatches: List[str] = []
    for relative, expected_hash in sorted(recorded_files.items()):
        relative = str(relative)
        path = (REPO_ROOT / relative).resolve()
        if not _inside(path, REPO_ROOT) or not path.is_file():
            mismatches.append(relative + " (missing)")
            continue
        if _sha256(path) != str(expected_hash):
            mismatches.append(relative)
    return mismatches


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _bool(value: Any, default: bool = False) -> bool:
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


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer: {value!r}") from exc


def _command_arg(command: Sequence[Any], name: str) -> Optional[str]:
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _command_hash(command: Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _phase_from_plan(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan has no phases list")
    matches = [item for item in phases if isinstance(item, Mapping) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"plan must contain exactly one phase named {name!r}")
    return matches[0]


def _resolve_plan_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _phase_root_from_plan(plan: Mapping[str, Any], phase: Mapping[str, Any]) -> Path:
    output_root = phase.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError(f"phase {phase.get('name')} has no output_root")
    path = Path(output_root)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _expected_flags(plan: Mapping[str, Any], phase: Mapping[str, Any]) -> Dict[str, Any]:
    global_cfg = plan.get("global")
    if not isinstance(global_cfg, Mapping):
        raise ValueError("plan has no global configuration")
    models = phase.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError(f"phase {phase.get('name')} must contain one registered model")
    alias = str(models[0])
    aliases = plan.get("alias_contract")
    if not isinstance(aliases, Mapping) or not isinstance(aliases.get(alias), Mapping):
        raise ValueError(f"missing alias contract for {alias}")
    contract = aliases[alias]
    executable = str(contract.get("executable_model", ""))
    overrides = contract.get("config_overrides", {})
    if not isinstance(overrides, Mapping) or not executable:
        raise ValueError(f"invalid alias contract for {alias}")
    # These values are the effective values after the runner applies the
    # alias's config_overrides.  Phase keys intentionally override global keys.
    def cfg(name: str, default: Any = None) -> Any:
        if name in phase:
            return phase[name]
        return global_cfg.get(name, default)

    relation_loss = _bool(cfg("mamba_acf_relation_loss", False))
    relation_lag_mask = _bool(cfg("mamba_acf_relation_lag_mask", False))
    # The runner emits the relation-only mask flag only when the auxiliary
    # objective is active.  Normalize the global plan default for plain and
    # memory arms so the readout checks the effective command configuration.
    if not relation_loss:
        relation_lag_mask = False
    expected: Dict[str, Any] = {
        "alias": alias,
        "executable_model": executable,
        "features": str(cfg("features", EXPECTED_FEATURES)),
        "seq_len": _int(cfg("seq_len", 96), "plan seq_len"),
        "label_len": _int(cfg("label_len", 48), "plan label_len"),
        "pred_len": EXPECTED_HORIZON,
        "d_model": _int(cfg("d_model", 32), "plan d_model"),
        "d_ff": _int(cfg("d_ff", 64), "plan d_ff"),
        "e_layers": _int(cfg("e_layers", 1), "plan e_layers"),
        "batch_size": _int(cfg("batch_size", 32), "plan batch_size"),
        "train_epochs": _int(cfg("epochs", 8), "plan epochs"),
        "patience": _int(cfg("patience", 3), "plan patience"),
        "train_ratio": float(cfg("train_ratio", 0.6)),
        "mamba_variable_mixer": _bool(overrides.get("mamba_variable_mixer", cfg("mamba_variable_mixer", False))),
        "mamba_variable_group_size": _int(overrides.get("mamba_variable_group_size", cfg("mamba_variable_group_size", 1)), "variable group size"),
        "mamba_variable_permutation": _bool(cfg("mamba_variable_permutation", False)),
        "mamba_causal_embedding": _bool(cfg("mamba_causal_embedding", False)),
        "mamba_use_acf_memory": _bool(cfg("mamba_use_acf_memory", False)),
        "mamba_acf_lag_mode": str(cfg("mamba_acf_lag_mode", "fixed")),
        "mamba_acf_relation_loss": relation_loss,
        "mamba_acf_relation_scope": str(cfg("mamba_acf_relation_scope", "both")),
        "mamba_acf_relation_lag_mask": relation_lag_mask,
        "mamba_acf_relation_min_lag": _int(cfg("mamba_acf_relation_min_lag", 0), "relation min lag"),
        "mamba_acf_relation_target_mode": str(cfg("mamba_acf_relation_target_mode", "hard_selected")),
        "skip_test": _bool(cfg("skip_test", False)),
        "stream_metrics": _bool(cfg("stream_metrics", False)),
        "stream_sample_windows": _int(cfg("stream_sample_windows", 1), "sample windows"),
        "AutoCon": bool("AutoCon" in executable),
    }
    if phase.get("autocon_lambda") is not None:
        expected["AutoCon_lambda"] = float(phase["autocon_lambda"])
    return expected


def _dataset_path(plan: Mapping[str, Any], dataset: str) -> Path:
    fingerprints = plan.get("data_fingerprints")
    if not isinstance(fingerprints, Mapping) or not isinstance(fingerprints.get(dataset), Mapping):
        raise ValueError(f"plan has no fingerprint for dataset {dataset}")
    relative = fingerprints[dataset].get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid relative_path for dataset {dataset}")
    path = (REPO_ROOT / relative).resolve()
    if not _inside(path, REPO_ROOT) or not path.is_file():
        raise ValueError(f"dataset path is missing or outside repository: {path}")
    return path


def _validate_provenance(
    phase_root: Path,
    phase: Mapping[str, Any],
    plan_path: Path,
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    provenance_path = phase_root / "provenance.json"
    if not provenance_path.is_file():
        raise ValueError(f"missing provenance: {provenance_path}")
    provenance = _read_json(provenance_path)
    if validation.get("provenance") and Path(str(validation["provenance"])).resolve() != provenance_path.resolve():
        raise ValueError(f"phase validation points to a different provenance: {phase_root}")
    if provenance.get("phase") != phase.get("name"):
        raise ValueError(f"provenance phase mismatch: {phase_root}")
    plan_hash = _sha256(plan_path)
    if provenance.get("plan_sha256") != plan_hash or validation.get("plan_sha256") != plan_hash:
        raise ValueError(f"plan hash mismatch in provenance/validation: {phase_root}")
    recorded_plan = Path(str(provenance.get("plan", ""))).resolve()
    if recorded_plan != plan_path.resolve():
        raise ValueError(f"provenance plan path mismatch: {phase_root}")

    snapshot_value = provenance.get("source_snapshot")
    if not snapshot_value:
        raise ValueError(f"provenance has no source_snapshot: {phase_root}")
    snapshot = Path(str(snapshot_value)).resolve()
    if not _inside(snapshot, phase_root / "provenance") or not snapshot.is_dir():
        raise ValueError(f"source snapshot is outside phase provenance directory: {snapshot}")
    files, tree_hash = _hash_file_map(snapshot)
    if files != provenance.get("source_files") or tree_hash != provenance.get("source_tree_sha256"):
        raise ValueError(f"source snapshot hash map changed: {phase_root}")
    live_source_mismatches = _live_source_mismatches(provenance.get("source_files", {}))

    expected_dataset_hashes: Dict[str, str] = {}
    for dataset in EXPECTED_DATASETS:
        path = _dataset_path(plan, dataset)
        expected = str(plan["data_fingerprints"][dataset].get("sha256", ""))
        actual = _sha256(path)
        if not expected or actual != expected:
            raise ValueError(f"dataset fingerprint mismatch for {dataset}: {path}")
        expected_dataset_hashes[dataset] = actual
    if provenance.get("dataset_hashes") != expected_dataset_hashes:
        raise ValueError(f"provenance dataset hashes mismatch: {phase_root}")

    ids = {str(record.get("provenance_id", "")) for record in records}
    if len(ids) != 1 or not next(iter(ids)) or next(iter(ids)) != str(provenance.get("provenance_id")):
        raise ValueError(f"manifest/provenance IDs are inconsistent: {phase_root}")
    source_before = {str(record.get("source_tree_sha256_before", "")) for record in records}
    source_after = {str(record.get("source_tree_sha256_after", "")) for record in records}
    expected_source = str(provenance.get("source_tree_sha256", ""))
    if source_before != {expected_source} or source_after != {expected_source}:
        raise ValueError(f"job source hashes are inconsistent: {phase_root}")
    return {
        "path": str(provenance_path.resolve()),
        "sha256": _sha256(provenance_path),
        "provenance_id": str(provenance.get("provenance_id")),
        "source_snapshot": str(snapshot),
        "source_tree_sha256": expected_source,
        "live_source_mismatches": live_source_mismatches,
        "live_source_matches_snapshot": not live_source_mismatches,
        "dataset_hashes": expected_dataset_hashes,
    }


def _artifact_paths(result_dir: Path, checkpoint_dir: Path, hashes: Mapping[str, Any]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for name in hashes:
        name = str(name)
        if name in RESULT_ARTIFACT_NAMES:
            path = result_dir / name
        elif name in CHECKPOINT_ARTIFACT_NAMES:
            path = checkpoint_dir / name
        else:
            raise ValueError(f"unknown artifact hash entry {name!r} in {result_dir}")
        if not path.is_file():
            raise ValueError(f"artifact listed in hash map is missing: {path}")
        if name in RESULT_ARTIFACT_NAMES and not _inside(path, result_dir):
            raise ValueError(f"result artifact escaped result directory: {path}")
        if name in CHECKPOINT_ARTIFACT_NAMES and not _inside(path, checkpoint_dir):
            raise ValueError(f"checkpoint artifact escaped checkpoint directory: {path}")
        output[name] = path
    return output


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(item)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _coverage_from_logs(
    phase: str,
    config: Mapping[str, Any],
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    relation_enabled = _bool(config.get("mamba_acf_relation_loss", False))
    log_path = checkpoint_dir / "loss_logs.json"
    if not log_path.is_file():
        raise ValueError(f"missing loss_logs.json: {checkpoint_dir}")
    logs = _read_json(log_path)
    if not relation_enabled:
        if logs.get("acf_relation_objective_version") not in (None, "", 0):
            raise ValueError(f"non-relation arm unexpectedly has relation objective metadata: {phase}")
        return {"enabled": False, "path": str(log_path), "sha256": _sha256(log_path)}
    if logs.get("acf_relation_objective_version") != 3:
        raise ValueError(f"relation objective is not v3: {checkpoint_dir}")
    if str(logs.get("acf_relation_target_mode")) != str(config.get("mamba_acf_relation_target_mode")):
        raise ValueError(f"relation target mode mismatch between config and loss log: {checkpoint_dir}")
    summary: Dict[str, Any] = {"enabled": True, "path": str(log_path), "sha256": _sha256(log_path), "objective_version": 3}
    for key in RELATION_LOG_KEYS:
        values = logs.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"relation coverage series missing/empty ({key}): {checkpoint_dir}")
        numbers = [_finite(value, f"{checkpoint_dir.name} {key}") for value in values]
        if any(value <= 0.0 for value in numbers):
            raise ValueError(f"relation coverage is zero/non-positive ({key}): {checkpoint_dir}")
        summary[key] = {"n": len(numbers), "min": min(numbers), "max": max(numbers), "mean": statistics.fmean(numbers)}
    effective = config.get("mamba_acf_relation_effective_lags", [])
    selected = config.get("mamba_acf_relation_selected_lags", [])
    if not isinstance(effective, list) or not effective or not isinstance(selected, list) or not selected:
        raise ValueError(f"relation effective/selected lags are empty: {checkpoint_dir}")
    effective_int = {_int(value, "effective relation lag") for value in effective}
    selected_int = {_int(value, "selected relation lag") for value in selected}
    minimum = _int(config.get("mamba_acf_relation_min_lag", 0), "relation min lag")
    if not effective_int.issubset(selected_int) or any(value < minimum for value in effective_int):
        raise ValueError(f"relation lag feasibility contract failed: {checkpoint_dir}")
    summary["effective_lags"] = sorted(effective_int)
    summary["selected_lags"] = sorted(selected_int)
    summary["min_lag"] = minimum
    return summary


def _load_phase(
    root: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    phase_name: str,
    phase_kind: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    phase = _phase_from_plan(plan, phase_name)
    phase_root = (root / phase_name).resolve()
    registered_root = _phase_root_from_plan(plan, phase)
    if phase_root != registered_root:
        raise ValueError(f"phase output_root mismatch for {phase_name}: {phase_root} != {registered_root}")
    validation_path = phase_root / "phase_validation.json"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ValueError(f"invalid phase validation: {validation_path}")
    if validation.get("plan") and Path(str(validation["plan"])).resolve() != plan_path.resolve():
        raise ValueError(f"phase validation points to another plan: {phase_name}")
    if validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError(f"phase validation plan hash mismatch: {phase_name}")
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file() or validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"manifest missing or changed after validation: {phase_name}")
    if validation.get("manifest") and Path(str(validation["manifest"])).resolve() != manifest_path.resolve():
        raise ValueError(f"phase validation points to another manifest: {phase_name}")
    records = _read_manifest(manifest_path)
    expected_jobs = _int(validation.get("expected_jobs", -1), "expected jobs")
    if expected_jobs != EXPECTED_JOBS_PER_PHASE or len(records) != expected_jobs:
        raise ValueError(f"unexpected coverage for {phase_name}: {len(records)} / {expected_jobs}")
    if _int(validation.get("manifest_unique_jobs", -1), "unique jobs") != expected_jobs:
        raise ValueError(f"manifest unique-job count mismatch: {phase_name}")
    if validation.get("malformed_manifest_lines", 0) != 0 or validation.get("errors") not in ([], None):
        raise ValueError(f"phase validation contains errors: {phase_name}")
    status_counts = validation.get("status_counts", {})
    if status_counts != {"ok": expected_jobs}:
        raise ValueError(f"phase status counts are not all ok: {phase_name}")
    ids = [str(item.get("job_id", "")) for item in records]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"manifest job IDs are missing/duplicated: {phase_name}")

    provenance_meta = _validate_provenance(phase_root, phase, plan_path, plan, validation, records)
    hash_map = validation.get("artifact_hashes_by_job")
    if not isinstance(hash_map, Mapping) or set(hash_map) != set(ids):
        raise ValueError(f"phase artifact hash map does not cover manifest exactly: {phase_name}")
    expected = _expected_flags(plan, phase)
    rows: List[Dict[str, Any]] = []
    observed_coordinates: set[Tuple[str, int]] = set()
    result_root = (phase_root / "results").resolve()
    checkpoint_root = (phase_root / "checkpoints").resolve()
    for record in records:
        job_id = str(record.get("job_id"))
        if record.get("phase") != phase_name or record.get("status") != "ok":
            raise ValueError(f"manifest record is not a successful {phase_name} job: {job_id}")
        if record.get("command_sha256") != _command_hash(record.get("command", [])):
            raise ValueError(f"command hash mismatch: {job_id}")
        if str(record.get("model", "")) != str(expected["alias"]):
            raise ValueError(
                f"manifest registered alias mismatch for {job_id}: "
                f"{record.get('model')!r} != {expected['alias']!r}"
            )
        dataset = str(record.get("dataset", ""))
        seed = _int(record.get("seed"), f"{job_id} seed")
        features = str(record.get("features", ""))
        horizon = _int(record.get("pred_len"), f"{job_id} pred_len")
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected coordinate: {job_id}")
        if features != EXPECTED_FEATURES or horizon != EXPECTED_HORIZON:
            raise ValueError(f"unexpected feature/horizon contract: {job_id}")
        coordinate = (dataset, seed)
        if coordinate in observed_coordinates:
            raise ValueError(f"duplicate dataset/seed coordinate: {job_id}")
        observed_coordinates.add(coordinate)
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected_artifact: {job_id}")
        metrics_path = Path(str(expected_artifact)).expanduser().resolve()
        result_dir = metrics_path.parent
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not _inside(result_dir, result_root):
            raise ValueError(f"expected artifact is missing/outside result root: {job_id}")
        checkpoint_dir = checkpoint_root / result_dir.name
        command_checkpoint_root = _command_arg(record.get("command", []), "--checkpoints")
        if command_checkpoint_root and Path(command_checkpoint_root).expanduser().resolve() != checkpoint_root:
            raise ValueError(f"command checkpoint root mismatch: {job_id}")
        record_hashes = record.get("artifact_hashes")
        phase_hashes = hash_map.get(job_id)
        if not isinstance(record_hashes, Mapping) or not isinstance(phase_hashes, Mapping):
            raise ValueError(f"missing per-job artifact hash map: {job_id}")
        if dict(record_hashes) != dict(phase_hashes):
            raise ValueError(f"manifest/phase_validation artifact hashes differ: {job_id}")
        paths = _artifact_paths(result_dir, checkpoint_dir, record_hashes)
        actual_hashes = {name: _sha256(path) for name, path in paths.items()}
        if actual_hashes != dict(record_hashes):
            raise ValueError(f"artifact hash mismatch after validation: {job_id}")

        config = _read_json(result_dir / "config.json")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {job_id}")
        for metric in ("mse", "mae"):
            _finite(scaled.get(metric), f"{job_id} scaled.{metric}")
        checks: Dict[str, Any] = {
            "data": dataset,
            "features": features,
            "seq_len": expected["seq_len"],
            "label_len": expected["label_len"],
            "pred_len": horizon,
            "model": expected["executable_model"],
            "AutoCon": expected["AutoCon"],
            "run_seed": seed,
            "seed": seed,
            "mamba_variable_mixer": expected["mamba_variable_mixer"],
            "mamba_variable_group_size": expected["mamba_variable_group_size"],
            "mamba_variable_permutation": expected["mamba_variable_permutation"],
            "mamba_causal_embedding": expected["mamba_causal_embedding"],
            "mamba_use_acf_memory": expected["mamba_use_acf_memory"],
            "mamba_acf_lag_mode": expected["mamba_acf_lag_mode"],
            "mamba_acf_relation_loss": expected["mamba_acf_relation_loss"],
            "mamba_acf_relation_scope": expected["mamba_acf_relation_scope"],
            "mamba_acf_relation_lag_mask": expected["mamba_acf_relation_lag_mask"],
            "mamba_acf_relation_min_lag": expected["mamba_acf_relation_min_lag"],
            "mamba_acf_relation_target_mode": expected["mamba_acf_relation_target_mode"],
            "train_epochs": expected["train_epochs"],
            "patience": expected["patience"],
            "batch_size": expected["batch_size"],
            "skip_test": expected["skip_test"],
            "stream_metrics": expected["stream_metrics"],
            "stream_sample_windows": expected["stream_sample_windows"],
        }
        for key, expected_value in checks.items():
            actual = config.get(key)
            if isinstance(expected_value, bool):
                matches = _bool(actual) == expected_value
            elif isinstance(expected_value, float):
                try:
                    matches = math.isclose(float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-12)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected_value
            if not matches:
                raise ValueError(f"config flag mismatch {key} for {job_id}: {actual!r} != {expected_value!r}")
        if expected["AutoCon"] and not math.isclose(float(config.get("AutoCon_lambda")), expected["AutoCon_lambda"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"AutoCon lambda mismatch: {job_id}")
        manifest_lambda = record.get("autocon_lambda")
        if expected["AutoCon"]:
            if manifest_lambda is None or not math.isclose(
                float(manifest_lambda), expected["AutoCon_lambda"], rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"manifest AutoCon lambda mismatch: {job_id}")
        elif manifest_lambda is not None:
            raise ValueError(f"plain arm unexpectedly records an AutoCon lambda: {job_id}")
        if config.get("plan_job_id") != job_id:
            raise ValueError(f"config plan_job_id mismatch: {job_id}")
        coverage = _coverage_from_logs(phase_name, config, checkpoint_dir)
        rows.append({
            "phase": phase_name,
            "kind": phase_kind,
            "dataset": dataset,
            "seed": seed,
            "features": features,
            "horizon": horizon,
            "mse": _finite(scaled.get("mse"), f"{job_id} scaled.mse"),
            "mae": _finite(scaled.get("mae"), f"{job_id} scaled.mae"),
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "job_id": job_id,
            "config_sha256": _sha256(result_dir / "config.json"),
            "metrics_sha256": _sha256(metrics_path),
            "coverage": coverage,
        })
    expected_coordinates = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    if observed_coordinates != expected_coordinates:
        raise ValueError(f"phase coordinates do not match the registered 6-job matrix: {phase_name}")
    meta = {
        "phase": phase_name,
        "kind": phase_kind,
        "root": str(phase_root),
        "validation": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "provenance": provenance_meta,
        "expected_jobs": expected_jobs,
        "observed_jobs": len(rows),
        "valid": True,
    }
    return rows, meta


def _paired_t(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot run a paired t test on an empty vector")
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    if math.isfinite(sd) and sd > 0.0:
        t_stat = mean / (sd / math.sqrt(len(numbers)))
    elif mean == 0.0:
        t_stat = 0.0
    else:
        t_stat = math.copysign(math.inf, mean)
    p_value: Optional[float] = None
    if scipy_stats is not None and len(numbers) > 1:
        try:
            result = scipy_stats.ttest_1samp(numbers, 0.0)
            t_stat = float(result.statistic)
            p_value = float(result.pvalue)
        except Exception:
            p_value = None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0.0 for value in numbers),
        "ties": sum(value == 0.0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
    }


def _make_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    indexed = {(str(row["kind"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            plain = indexed[("plain", dataset, seed)]
            memory = indexed[("memory", dataset, seed)]
            relation = indexed[("memory_relation", dataset, seed)]
            pairs.append({
                "dataset": dataset,
                "seed": seed,
                "features": plain["features"],
                "horizon": int(plain["horizon"]),
                "plain_mse": float(plain["mse"]),
                "memory_mse": float(memory["mse"]),
                "relation_mse": float(relation["mse"]),
                "delta_memory_minus_plain_mse": float(memory["mse"] - plain["mse"]),
                "delta_relation_minus_memory_mse": float(relation["mse"] - memory["mse"]),
                "plain_mae": float(plain["mae"]),
                "memory_mae": float(memory["mae"]),
                "relation_mae": float(relation["mae"]),
                "delta_memory_minus_plain_mae": float(memory["mae"] - plain["mae"]),
                "delta_relation_minus_memory_mae": float(relation["mae"] - memory["mae"]),
                "plain_job_id": plain["job_id"],
                "memory_job_id": memory["job_id"],
                "relation_job_id": relation["job_id"],
            })
    return pairs


def _contrast(
    label: str,
    pairs: Sequence[Mapping[str, Any]],
    treatment_metric: str,
    reference_metric: str,
    delta_metric: str,
) -> Dict[str, Any]:
    deltas = [float(pair[delta_metric]) for pair in pairs]
    references = [float(pair[reference_metric]) for pair in pairs]
    treatments = [float(pair[treatment_metric]) for pair in pairs]
    test = _paired_t(deltas)
    return {
        "label": label,
        "n": len(deltas),
        "reference_mean": statistics.fmean(references),
        "treatment_mean": statistics.fmean(treatments),
        "delta_mean": test["mean"],
        "delta_sd": test["sd"],
        "wins_lower": test["wins_lower"],
        "ties": test["ties"],
        "relative_change_pct": 100.0 * test["mean"] / statistics.fmean(references),
        "paired_t": test["t_statistic"],
        "paired_t_two_sided_p": test["two_sided_p"],
        "delta_values": deltas,
    }


def _summaries(pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    contrasts = [
        ("variable memory minus variable plain", "memory_mse", "plain_mse", "delta_memory_minus_plain_mse"),
        ("variable memory + relation minus variable memory", "relation_mse", "memory_mse", "delta_relation_minus_memory_mse"),
    ]
    mae_contrasts = [
        ("variable memory minus variable plain", "memory_mae", "plain_mae", "delta_memory_minus_plain_mae"),
        ("variable memory + relation minus variable memory", "relation_mae", "memory_mae", "delta_relation_minus_memory_mae"),
    ]
    output: List[Dict[str, Any]] = []
    for dataset in list(EXPECTED_DATASETS) + ["pooled descriptive (not independent)"]:
        subset = list(pairs) if dataset.startswith("pooled") else [item for item in pairs if item["dataset"] == dataset]
        for (label, treatment_mse, reference_mse, delta_mse), (_, treatment_mae, reference_mae, delta_mae) in zip(contrasts, mae_contrasts):
            mse = _contrast(label, subset, treatment_mse, reference_mse, delta_mse)
            mae = _contrast(label, subset, treatment_mae, reference_mae, delta_mae)
            output.append({"dataset": dataset, "label": label, "mse": mse, "mae": mae})
    return output


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _fmt_pm(summary: Mapping[str, Any]) -> str:
    return f"{_fmt(summary.get('delta_mean'))} +/- {_fmt(summary.get('delta_sd'))}"


def _markdown(payload: Mapping[str, Any]) -> str:
    provenance = payload["validation"]
    pairs = payload["per_seed"]
    summaries = payload["contrasts"]
    lines = [
        "# MambaACF Variable Memory + Relation Persistence Readout",
        "",
        "> Exploratory paired screen only. Each dataset has n=3 seeds; no result here is confirmatory evidence or a basis for publication claims.",
        "",
        "## Contract and validation",
        "",
        f"- Plan: `{payload['plan']['path']}`; SHA-256 `{payload['plan']['sha256']}`.",
        f"- Screen root: `{payload['screen_root']}`; successful jobs: `{payload['job_count']}/18`.",
        "- Pairing unit: same dataset, feature setting, horizon, and seed. Delta is treatment minus reference; negative is favorable.",
        "- Metric source: `stream_metrics.json` scaled MSE/MAE. Forecast windows are not treated as independent observations.",
        "",
        "| Phase | Jobs | Manifest SHA-256 | Provenance ID | Source-tree SHA-256 |",
        "|---|---:|---|---|---|",
    ]
    for item in provenance:
        lines.append(
            f"| `{item['phase']}` | {item['observed_jobs']}/{item['expected_jobs']} | `{item['manifest_sha256']}` | `{item['provenance']['provenance_id']}` | `{item['provenance']['source_tree_sha256']}` |"
        )
    live_drift = sorted({
        mismatch
        for item in provenance
        for mismatch in item.get("provenance", {}).get("live_source_mismatches", [])
    })
    if live_drift:
        preview = ", ".join(live_drift[:8])
        suffix = " ..." if len(live_drift) > 8 else ""
        lines.extend([
            "",
            f"- Source status: the metrics are bound to the immutable provenance snapshot; the live working tree differs in {len(live_drift)} captured source file(s): `{preview}{suffix}`. This readout must therefore be treated as a provenance-snapshot training result, not a current-tree training result.",
        ])
    elif any("live_source_mismatches" in item.get("provenance", {}) for item in provenance):
        lines.extend([
            "",
            "- Source status: the live working tree matches the captured provenance source files.",
        ])
    if payload["plan"].get("status") in {
        "exploratory_current_code_sidecar_replay_not_confirmatory",
        "exploratory_live_sidecar_screen_not_confirmatory",
    }:
        lines.insert(
            10,
            "- Artifact contract: the provenance-snapshot exploratory runs persisted a complete `acf_prior.npz` and a per-job SHA-256 binding; this remains non-confirmatory.",
        )
        lines.insert(
            11,
            "- Evidence boundary: each memory/relation job fitted its train-only prior during training and wrote the sidecar after that fit. The reported test metrics came from that same training run; no metric here came from a later standalone evaluation that loaded the sidecar. This is a persistence/contract screen, not an independent confirmatory reproduction.",
        )
        lines.insert(
            12,
            "- A separate 18-job `is_training=0` standalone sidecar replay has now been completed in `artifacts/exploratory/mamba_acf_sidecar_standalone_replay_20260831_audit/`; its manifest freezes inputs/tolerances and its JSONL records each job. It validates the current evaluation loader against these historical checkpoints but does not turn them into current-tree training or confirmatory evidence.",
        )
    lines.extend([
        "",
        "## Per-seed paired metrics",
        "",
        "| Dataset | Seed | Plain MSE | Memory MSE | Memory-Plain | Memory+Relation MSE | Relation-Memory | Plain MAE | Memory MAE | Memory-Plain | Memory+Relation MAE | Relation-Memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pairs:
        lines.append("| " + " | ".join([
            str(row["dataset"]), str(row["seed"]), _fmt(row["plain_mse"]), _fmt(row["memory_mse"]),
            _fmt(row["delta_memory_minus_plain_mse"]), _fmt(row["relation_mse"]),
            _fmt(row["delta_relation_minus_memory_mse"]), _fmt(row["plain_mae"]), _fmt(row["memory_mae"]),
            _fmt(row["delta_memory_minus_plain_mae"]), _fmt(row["relation_mae"]),
            _fmt(row["delta_relation_minus_memory_mae"]),
        ]) + " |")
    lines.extend([
        "",
        "## Paired summaries",
        "",
        "SD is the sample standard deviation across seed-level deltas. Paired t values and p-values are descriptive only (n=3 per dataset).",
        "",
        "| Dataset | Contrast | Metric | n | Reference mean | Treatment mean | Delta mean +/- SD | Wins | Relative change | t | two-sided p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in summaries:
        for metric in ("mse", "mae"):
            summary = item[metric]
            lines.append("| " + " | ".join([
                str(item["dataset"]), str(item["label"]), metric.upper(), str(summary["n"]),
                _fmt(summary["reference_mean"]), _fmt(summary["treatment_mean"]), _fmt_pm(summary),
                f"{summary['wins_lower']}/{summary['n']}", _fmt(summary["relative_change_pct"], 3) + "%",
                _fmt(summary["paired_t"], 4), _fmt(summary["paired_t_two_sided_p"], 4),
            ]) + " |")
    lines.extend(["", "## Relation objective coverage", "", "The relation arm was required to use objective v3 and strictly positive global coverage series.", "", "| Dataset | Seed | Objective | Valid min | Candidate min | Candidate-anchor min | Positive-count min | Effective lags |", "|---|---:|---:|---:|---:|---:|---:|---|"])
    relation_rows = [row for row in payload["rows"] if row["kind"] == "memory_relation"]
    for row in sorted(relation_rows, key=lambda x: (x["dataset"], x["seed"])):
        cov = row["coverage"]
        lines.append("| " + " | ".join([
            str(row["dataset"]), str(row["seed"]), str(cov["objective_version"]),
            _fmt(cov["acf_relation_global_valid_fraction"]["min"]),
            _fmt(cov["acf_relation_global_candidate_fraction"]["min"]),
            _fmt(cov["acf_relation_global_candidate_active_anchor_fraction"]["min"]),
            _fmt(cov["acf_relation_global_positive_count_mean"]["min"]),
            ", ".join(str(v) for v in cov["effective_lags"]),
        ]) + " |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- Three seeds per dataset provide a mechanism-screening readout only. The exact paired Wilcoxon minimum p-value is not used to inflate evidence; only descriptive paired t statistics are shown here.",
        "- The pooled row is a cross-dataset descriptive summary, not six independent replications of one data-generating process.",
        "- The variable branch, train-only ACF memory, and relation objective are jointly confounded in the second contrast; this screen cannot identify a causal contribution of each component.",
        "- A separate, frozen confirmation matrix with validation-only selection and an independent evaluation protocol is required before any generalization or publication claim.",
        "",
    ])
    return "\n".join(lines)


def build_payload(root: Path, plan_path: Path) -> Dict[str, Any]:
    root = root.expanduser().resolve()
    plan_path = _resolve_plan_path(plan_path)
    if not root.is_dir():
        raise ValueError(f"screen root does not exist: {root}")
    plan = _read_json(plan_path)
    if plan.get("status") not in SUPPORTED_PLAN_STATUSES:
        raise ValueError(f"unexpected plan status: {plan.get('status')!r}")
    all_rows: List[Dict[str, Any]] = []
    phase_meta: List[Dict[str, Any]] = []
    for phase_name, phase_kind in PHASES:
        rows, meta = _load_phase(root, plan_path, plan, phase_name, phase_kind)
        all_rows.extend(rows)
        phase_meta.append(meta)
    if len(all_rows) != len(PHASES) * EXPECTED_JOBS_PER_PHASE:
        raise ValueError(f"expected 18 rows, found {len(all_rows)}")
    pairs = _make_pairs(all_rows)
    # Explicitly verify that every contrast has exactly the intended paired keys.
    if len(pairs) != EXPECTED_JOBS_PER_PHASE:
        raise ValueError(f"expected 6 paired rows, found {len(pairs)}")
    return {
        "schema_version": 1,
        "status": "exploratory_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "screen_root": str(root),
        "job_count": len(all_rows),
        "plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "status": str(plan.get("status")),
        },
        "protocol": {
            "phases": [name for name, _ in PHASES],
            "datasets": list(EXPECTED_DATASETS),
            "seeds": list(EXPECTED_SEEDS),
            "features": EXPECTED_FEATURES,
            "horizon": EXPECTED_HORIZON,
            "inferential_unit": "seed/run",
            "contrasts": [
                "variable memory minus variable plain",
                "variable memory + relation minus variable memory",
            ],
        },
        "validation": phase_meta,
        "rows": all_rows,
        "per_seed": pairs,
        "contrasts": _summaries(pairs),
        "limitations": [
            "n=3 paired seeds per dataset; descriptive t tests only",
            "pooled six-pair row is not six independent replications",
            "forecast windows are not inferential replicates",
            "no test-based architecture or lambda selection",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="completed screen root")
    parser.add_argument("--plan", type=Path, required=True, help="registered plan JSON")
    parser.add_argument("--output-json", type=Path, default=None, help="machine JSON (default: <root>/analysis.json)")
    parser.add_argument("--output-md", type=Path, default=None, help="Markdown report (default: docs/research/..._readout.md)")
    args = parser.parse_args(argv)
    try:
        payload = build_payload(args.root, args.plan)
        root = args.root.expanduser().resolve()
        output_json = (args.output_json.expanduser() if args.output_json else root / "analysis.json")
        output_md = (args.output_md.expanduser() if args.output_md else REPO_ROOT / "docs/research/mamba_acf_variable_memory_relation_screen_readout.md")
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_md.write_text(_markdown(payload), encoding="utf-8")
        print(f"wrote {output_json}")
        print(f"wrote {output_md}")
        for item in payload["contrasts"]:
            print(
                f"{item['dataset']} | {item['label']} | "
                f"MSE delta={item['mse']['delta_mean']:.9f} +/- {item['mse']['delta_sd']:.9f} "
                f"wins={item['mse']['wins_lower']}/{item['mse']['n']} "
                f"p={item['mse']['paired_t_two_sided_p']}"
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"analyze_mamba_variable_memory_relation_screen: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
