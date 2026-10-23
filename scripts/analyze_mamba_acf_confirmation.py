#!/usr/bin/env python3
"""Analyze frozen MambaACF confirmation phases with strict provenance gates.

The script intentionally has a narrower contract than the exploratory
statistics utilities.  It consumes one registered plan and one or more
validated phase roots, then computes paired seed-level contrasts.  Forecast
windows and channels are never treated as inferential replicates.

The simultaneous interval is a pre-fixed Bonferroni adjusted-alpha interval:
for a comparison with ``m`` registered dataset-horizon cells, each cell uses
``alpha / m`` in a one-sided paired Student-t upper bound.  Holm is applied to
the one-sided p-values as a separate multiplicity-sensitive summary.  This is
not an inversion of the Holm step-down procedure and is reported as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = (
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
)
METRIC_INDEX = 1  # metrics.npy order is [mae, mse, rmse, mape, mspe]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _hash_file_map(root: Path) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            files[path.relative_to(root).as_posix()] = _sha256(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, str]:
    paths = [result_dir / name for name in ARTIFACT_NAMES]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _command_arg(command: Sequence[Any], name: str) -> str | None:
    tokens = [str(item) for item in command]
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _strict_lag_list(value: Any, label: str, *, required: bool = True) -> list[int] | None:
    """Return a finite, positive, duplicate-free integer lag list."""

    if value is None and not required:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a non-empty list of positive integer lags")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} contains a non-finite/non-positive lag")
        number = _finite(item, f"{label} lag")
        if number <= 0.0 or not number.is_integer():
            raise ValueError(f"{label} contains a non-finite/non-positive lag")
        result.append(int(number))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate lags")
    return result


def _relation_log_contract(
    config: Mapping[str, Any], checkpoint_dir: Path | None, job_id: str
) -> dict[str, Any]:
    """Validate relation lag provenance and epoch-level coverage for one job."""

    relation_loss = _bool_value(config.get("mamba_acf_relation_loss", False))
    if not relation_loss:
        return {
            "relation_loss": False,
            "relation_effective_lags": None,
            "relation_selected_lags": None,
            "train_peak_lags": _strict_lag_list(
                config.get("mamba_acf_train_peak_lags"),
                f"{job_id} train peak lags",
                required=False,
            ),
        }
    effective = _strict_lag_list(
        config.get("mamba_acf_relation_effective_lags"),
        f"{job_id} relation effective lags",
    )
    selected = _strict_lag_list(
        config.get("mamba_acf_relation_selected_lags"),
        f"{job_id} relation selected lags",
    )
    train_peaks = _strict_lag_list(
        config.get("mamba_acf_train_peak_lags"),
        f"{job_id} train peak lags",
    )
    max_lag = _finite(
        config.get("mamba_acf_prior_max_lag", config.get("mamba_acf_max_lag", 720)),
        f"{job_id} prior max lag",
    )
    min_lag = _finite(
        config.get("mamba_acf_relation_min_lag", 0),
        f"{job_id} relation min lag",
    )
    if max_lag <= 0.0 or min_lag < 0.0:
        raise ValueError(f"{job_id} relation lag bounds are invalid")
    if any(lag > max_lag for lag in effective):
        raise ValueError(f"{job_id} relation effective lag exceeds measured ACF range")
    if any(lag < min_lag for lag in effective):
        raise ValueError(f"{job_id} relation effective lag is below relation_min_lag")
    if not set(effective).issubset(set(selected)):
        raise ValueError(f"{job_id} relation effective lags are not selected lags")
    if not set(effective).issubset(set(train_peaks)):
        raise ValueError(f"{job_id} relation effective lags are not train-peak lags")
    lag_mode = str(config.get("mamba_acf_lag_mode", "")).strip().lower()
    if lag_mode not in {"train_peaks", "train_peaks_available"}:
        raise ValueError(f"{job_id} relation lag mode is not train-peak based")
    if str(config.get("mamba_acf_relation_lag_source", "")).strip().lower() != "train_peak_subset":
        raise ValueError(f"{job_id} relation lag source is not a train-peak subset")
    if _bool_value(config.get("mamba_acf_relation_lag_fallback", False)):
        raise ValueError(f"{job_id} relation lag fallback is forbidden")
    if checkpoint_dir is None:
        raise ValueError(f"{job_id} relation loss has no checkpoint directory")
    logs_path = checkpoint_dir / "loss_logs.json"
    if not logs_path.is_file():
        raise ValueError(f"{job_id} relation loss log is missing")
    logs = _read_json(logs_path)
    val_loss = logs.get("val_loss")
    if not isinstance(val_loss, list) or not val_loss:
        raise ValueError(f"{job_id} relation validation trajectory is empty")
    if logs.get("acf_relation_objective_version") != 3:
        raise ValueError(f"{job_id} relation objective version is not 3")
    logged_effective = _strict_lag_list(
        logs.get("acf_relation_effective_lags"),
        f"{job_id} logged relation effective lags",
    )
    logged_selected = _strict_lag_list(
        logs.get("acf_relation_selected_lags"),
        f"{job_id} logged relation selected lags",
    )
    logged_peaks = _strict_lag_list(
        logs.get("acf_relation_train_peak_lags"),
        f"{job_id} logged train peak lags",
    )
    if logged_effective != effective or logged_selected != selected or logged_peaks != train_peaks:
        raise ValueError(f"{job_id} relation lag metadata differs between config and loss log")
    if str(logs.get("acf_relation_lag_source", "")).strip().lower() != "train_peak_subset":
        raise ValueError(f"{job_id} relation loss log lag source is not train_peak_subset")
    if _bool_value(logs.get("acf_relation_lag_fallback", False)):
        raise ValueError(f"{job_id} relation loss log records a forbidden fallback")
    scope = str(config.get("mamba_acf_relation_scope", "both")).strip().lower()
    if scope not in {"global", "local", "both"}:
        raise ValueError(f"{job_id} relation scope is invalid")
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
    coverage: dict[str, Any] = {}
    for key, kind, positive in required:
        values = logs.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{job_id} relation coverage {key} is missing/empty")
        if len(values) != len(val_loss):
            raise ValueError(f"{job_id} relation coverage {key} length mismatches val_loss")
        numbers = [_finite(value, f"{job_id} {key}") for value in values]
        if kind == "fraction" and any(value < 0.0 or value > 1.0 for value in numbers):
            raise ValueError(f"{job_id} relation coverage {key} is outside [0, 1]")
        if positive and any(value <= 0.0 for value in numbers):
            raise ValueError(f"{job_id} relation coverage {key} is not strictly positive")
        coverage[key] = numbers
    return {
        "relation_loss": True,
        "relation_effective_lags": effective,
        "relation_selected_lags": selected,
        "train_peak_lags": train_peaks,
        "relation_lag_source": "train_peak_subset",
        "relation_lag_fallback": False,
        "relation_scope": scope,
        "relation_coverage": coverage,
    }


def _memory_lag_metadata(config: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    """Validate/return resolved forecast-memory lag metadata when present."""

    effective = _strict_lag_list(
        config.get("mamba_acf_effective_lags"),
        f"{job_id} effective memory lags",
        required=False,
    )
    train_peaks = _strict_lag_list(
        config.get("mamba_acf_train_peak_lags"),
        f"{job_id} train peak lags",
        required=False,
    )
    return {
        "use_acf_memory": _bool_value(config.get("mamba_use_acf_memory", False)),
        "effective_lags": effective,
        "effective_lag_source": str(config.get("mamba_acf_effective_lag_source", "")).strip().lower() or None,
        "effective_lag_fallback": _bool_value(config.get("mamba_acf_effective_lag_fallback", False)),
        "lag_mode": str(config.get("mamba_acf_lag_mode", "")).strip().lower() or None,
        "train_peak_lags": train_peaks,
    }


def _validate_memory_contract(
    plan: Mapping[str, Any],
    phase_name: str,
    model: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    job_id: str,
) -> None:
    """Fail closed for enabled MambaACF memory arms lacking lag provenance."""

    if "MambaACF" not in str(model) or not metadata.get("use_acf_memory"):
        return
    lags = metadata.get("effective_lags")
    if not isinstance(lags, list) or not lags:
        raise ValueError(f"{job_id} enabled ACF memory has no effective lags")
    if metadata.get("effective_lag_fallback"):
        raise ValueError(f"{job_id} enabled ACF memory uses a forbidden lag fallback")
    falsification = plan.get("acf_falsification_contract", {})
    placebo_phase = (
        str(falsification.get("fixed_placebo_arm", ""))
        if isinstance(falsification, Mapping) else ""
    )
    if phase_name == placebo_phase:
        if metadata.get("lag_mode") != "fixed" or metadata.get("effective_lag_source") != "fixed_configured":
            raise ValueError(f"{job_id} fixed placebo memory source is not fixed_configured")
        placebo_lags = _strict_lag_list(
            falsification.get("placebo_lags") if isinstance(falsification, Mapping) else None,
            "registered placebo lags",
        )
        if set(lags) != set(placebo_lags or ()):
            raise ValueError(f"{job_id} fixed placebo effective lags differ from registered lags")
        return
    if metadata.get("lag_mode") not in {"train_peaks", "train_peaks_available"}:
        raise ValueError(f"{job_id} enabled ACF memory is not train-peak based")
    if metadata.get("effective_lag_source") not in {"train_peaks", "train_peaks_available"}:
        raise ValueError(f"{job_id} enabled ACF memory source is not train-peak selection")
    train_peaks = metadata.get("train_peak_lags")
    if not isinstance(train_peaks, list) or not train_peaks:
        raise ValueError(f"{job_id} enabled ACF memory has no train-peak provenance")
    if not set(lags).issubset(set(train_peaks)):
        raise ValueError(f"{job_id} enabled ACF memory lags are not train-peak qualified")


def _registered_job_counts_contract(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    """Independently re-expand and verify the plan's registered job counts."""

    declared = plan.get("registered_job_counts")
    if declared is None:
        if isinstance(plan.get("confirmation_contract"), Mapping):
            raise ValueError("confirmation plan requires registered_job_counts")
        return None
    if not isinstance(declared, Mapping):
        raise ValueError("registered_job_counts must be an object")
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("cannot recompute registered_job_counts without phases")
    global_cfg = plan.get("global", {})
    if not isinstance(global_cfg, Mapping):
        global_cfg = {}
    phase_counts: dict[str, int] = {}
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("plan phase is not an object while recomputing job counts")
        name = str(phase.get("name", "")).strip()
        if not name or name in phase_counts:
            raise ValueError(f"invalid/duplicate phase while recomputing job counts: {name!r}")
        datasets = phase.get("datasets", [])
        models = phase.get("models", [])
        seeds = phase.get("seeds", [])
        if not isinstance(datasets, list) or not isinstance(models, list) or not isinstance(seeds, list):
            raise ValueError(f"phase {name} has invalid datasets/models/seeds")
        horizons_map = phase.get("horizons_by_dataset")
        if horizons_map is None:
            horizons = phase.get("horizons", global_cfg.get("horizons", []))
            if not isinstance(horizons, list):
                raise ValueError(f"phase {name} has invalid horizons")
            horizons_map = {str(dataset): horizons for dataset in datasets}
        count = 0
        for dataset in datasets:
            values = horizons_map.get(str(dataset), []) if isinstance(horizons_map, Mapping) else []
            if not isinstance(values, list) or not values:
                raise ValueError(f"phase {name} has no horizons for {dataset}")
            for model in models:
                # Only AutoCon model specs expand a lambda grid.  A global
                # lambda declaration must not multiply ordinary baselines.
                if str(model).endswith("AutoCon"):
                    lambda_setting = phase.get(
                        "autocon_lambdas",
                        phase.get(
                            "autocon_lambda",
                            global_cfg.get("autocon_lambdas", [0.03]),
                        ),
                    )
                    if lambda_setting == "selected":
                        lambda_count = 1
                    elif isinstance(lambda_setting, list):
                        lambda_count = len(lambda_setting)
                    else:
                        lambda_count = 1
                else:
                    lambda_count = 1
                count += len(values) * len(seeds) * lambda_count
        phase_counts[name] = count
    recomputed: dict[str, int] = dict(phase_counts)
    recomputed["selection_total"] = sum(
        count for name, count in phase_counts.items() if "selection" in name
    )
    recomputed["confirmation_total"] = sum(
        count for name, count in phase_counts.items()
        if "confirm" in name or "sensitivity" in name
    )
    recomputed["training_total"] = sum(phase_counts.values())
    recomputed["total"] = recomputed["training_total"]
    # Validate every declared count, while allowing non-count metadata fields
    # that the runner explicitly documents.
    allowed_metadata = {"resource_runs_not_in_training_matrix", "training_started"}
    for name in phase_counts:
        if name not in declared:
            raise ValueError(f"registered_job_counts is missing phase {name}")
    for key, value in declared.items():
        if key in allowed_metadata:
            continue
        if key not in recomputed:
            raise ValueError(f"registered_job_counts has unknown key {key!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"registered_job_counts[{key}] must be an integer")
        if int(value) != recomputed[key]:
            raise ValueError(
                f"registered_job_counts[{key}]={value} disagrees with independently "
                f"recomputed value={recomputed[key]}"
            )
    if isinstance(plan.get("confirmation_contract"), Mapping):
        for key in ("selection_total", "confirmation_total", "training_total"):
            if key not in declared:
                raise ValueError(f"registered_job_counts is missing aggregate key {key}")
    return {"declared": dict(declared), "recomputed": recomputed}


def _peak_placebo_contract(
    plan: Mapping[str, Any],
    contrast: Mapping[str, str],
    treatment: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Check lag-cardinality/source eligibility for the registered placebo.

    This gate is deliberately narrower than a generic lag comparison.  It is
    activated only for the exact peak-arm versus fixed-placebo endpoints in the
    plan; missing or unqualified metadata aborts the contrast before any
    statistical result is produced.
    """

    contract = plan.get("acf_falsification_contract")
    if not isinstance(contract, Mapping):
        return None
    peak_phase = str(contract.get("peak_arm", ""))
    placebo_phase = str(contract.get("fixed_placebo_arm", ""))
    if (
        contrast.get("treatment_phase") != peak_phase
        or contrast.get("reference_phase") != placebo_phase
    ):
        return None
    placebo_lags = _strict_lag_list(
        contract.get("placebo_lags"), "registered placebo lags"
    )
    if not placebo_lags:
        raise ValueError("peak/placebo attribution is blocked: registered placebo lags are missing")

    def coordinate(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("dataset"), row.get("features"), int(row.get("seq_len", -1)),
            int(row.get("pred_len", -1)), int(row.get("seed", -1)),
        )

    left = {coordinate(row): row for row in treatment}
    right = {coordinate(row): row for row in reference}
    if set(left) != set(right):
        raise ValueError("peak/placebo attribution is blocked: lag metadata is unpaired")
    by_cell: dict[tuple[Any, ...], set[tuple[int, int]]] = defaultdict(set)
    source_summary: dict[str, Any] = {
        "peak_lag_mode": "train_peaks_available",
        "peak_source": "train_peak_selection",
        "placebo_lag_mode": "fixed",
        "placebo_source": "fixed_configured",
        "placebo_lags": placebo_lags,
    }
    for key in sorted(left):
        peak = left[key]
        placebo = right[key]
        peak_memory = peak.get("memory")
        placebo_memory = placebo.get("memory")
        if not isinstance(peak_memory, Mapping) or not isinstance(placebo_memory, Mapping):
            raise ValueError("peak/placebo attribution is blocked: memory lag metadata is missing")
        if not peak_memory.get("use_acf_memory") or not placebo_memory.get("use_acf_memory"):
            raise ValueError("peak/placebo attribution is blocked: memory branch is disabled")
        peak_lags = peak_memory.get("effective_lags")
        fixed_lags = placebo_memory.get("effective_lags")
        if not isinstance(peak_lags, list) or not peak_lags:
            raise ValueError("peak/placebo attribution is blocked: peak effective lags are missing")
        if not isinstance(fixed_lags, list) or not fixed_lags:
            raise ValueError("peak/placebo attribution is blocked: placebo effective lags are missing")
        if peak_memory.get("lag_mode") not in {"train_peaks", "train_peaks_available"}:
            raise ValueError("peak/placebo attribution is blocked: peak lag mode is not train-peak based")
        if peak_memory.get("effective_lag_source") not in {"train_peaks", "train_peaks_available"}:
            raise ValueError("peak/placebo attribution is blocked: peak lag source is not train-peak selection")
        if peak_memory.get("effective_lag_fallback"):
            raise ValueError("peak/placebo attribution is blocked: peak lag fallback is forbidden")
        peak_train_lags = peak_memory.get("train_peak_lags")
        if not isinstance(peak_train_lags, list) or not set(peak_lags).issubset(set(peak_train_lags)):
            raise ValueError("peak/placebo attribution is blocked: peak lags are not train-peak qualified")
        if placebo_memory.get("lag_mode") != "fixed":
            raise ValueError("peak/placebo attribution is blocked: placebo lag mode is not fixed")
        if placebo_memory.get("effective_lag_source") != "fixed_configured":
            raise ValueError("peak/placebo attribution is blocked: placebo lag source is not fixed_configured")
        if placebo_memory.get("effective_lag_fallback"):
            raise ValueError("peak/placebo attribution is blocked: placebo lag fallback is forbidden")
        if set(fixed_lags) != set(placebo_lags):
            raise ValueError("peak/placebo attribution is blocked: placebo lags differ from registered lags")
        cell = key[:4]
        pair_counts = (len(peak_lags), len(fixed_lags))
        if pair_counts[0] != pair_counts[1]:
            raise ValueError(
                "peak/placebo attribution is blocked: effective lag cardinality differs "
                f"for {cell}: {pair_counts}"
            )
        by_cell[cell].add(pair_counts)
    cardinality = []
    for cell, counts in sorted(by_cell.items()):
        if len(counts) != 1:
            raise ValueError(
                "peak/placebo attribution is blocked: lag cardinality varies across seeds "
                f"for {cell}"
            )
        peak_count, placebo_count = next(iter(counts))
        cardinality.append({
            "dataset": cell[0], "features": cell[1], "seq_len": int(cell[2]),
            "pred_len": int(cell[3]), "peak_count": peak_count,
            "placebo_count": placebo_count, "equal": True,
        })
    return {
        "eligible": True,
        "kind": "peak_vs_fixed_placebo",
        "source_qualification": source_summary,
        "lag_cardinality_by_cell": cardinality,
    }


_MISSING = object()


def _threshold_candidates(
    gate: Mapping[str, Any],
    primary_key: str,
    ratio_keys: Sequence[str],
    log_keys: Sequence[str],
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """Collect ratio/log threshold declarations from a statistical gate.

    A threshold may be declared as a scalar (interpreted as a multiplicative
    ratio), as an object containing ``ratio`` and/or ``log_threshold``, or via
    one of the explicit alias keys.  Keeping both representations in the plan
    makes the contract readable while allowing a log-scale-only declaration.
    """

    ratios: list[tuple[str, Any]] = []
    logs: list[tuple[str, Any]] = []
    primary = gate.get(primary_key, _MISSING)
    if primary is not _MISSING:
        if isinstance(primary, Mapping):
            for key in ("ratio", "multiplicative_ratio", "threshold_ratio", "value"):
                if key in primary:
                    ratios.append((f"{primary_key}.{key}", primary[key]))
            for key in ("log_threshold", "log_ratio", "log"):
                if key in primary:
                    logs.append((f"{primary_key}.{key}", primary[key]))
            if not ratios and not logs:
                raise ValueError(
                    f"{primary_key} object must contain ratio or log_threshold"
                )
        else:
            ratios.append((primary_key, primary))
    for key in ratio_keys:
        if key in gate:
            ratios.append((key, gate[key]))
    for key in log_keys:
        if key in gate:
            logs.append((key, gate[key]))
    return ratios, logs


def _threshold_contract(
    gate: Mapping[str, Any],
    *,
    primary_key: str,
    ratio_keys: Sequence[str],
    log_keys: Sequence[str],
    label: str,
    require_ratio_at_least_one: bool = False,
    require_ratio_at_most_one: bool = False,
) -> dict[str, float | str]:
    """Resolve and validate one frozen multiplicative/log threshold.

    If both ratio and log forms are present they must agree numerically.  The
    parser intentionally rejects prose strings: a statistical gate must be a
    machine-checkable number in the registered plan.
    """

    ratios, logs = _threshold_candidates(gate, primary_key, ratio_keys, log_keys)
    if not ratios and not logs:
        raise ValueError(
            f"statistical_gate must freeze {label} as a numeric ratio or log threshold"
        )

    def one_number(values: Sequence[tuple[str, Any]], field: str) -> tuple[float | None, str | None]:
        if not values:
            return None, None
        parsed: list[tuple[str, float]] = []
        for source, value in values:
            if isinstance(value, bool):
                raise ValueError(f"{source} must be numeric, not boolean")
            number = _finite(value, source)
            parsed.append((source, number))
        first_source, first = parsed[0]
        for source, number in parsed[1:]:
            if not math.isclose(number, first, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(
                    f"conflicting {field} declarations: {first_source}={first} vs {source}={number}"
                )
        return first, first_source

    ratio, ratio_source = one_number(ratios, f"{label} ratio")
    log_value, log_source = one_number(logs, f"{label} log threshold")
    if ratio is not None and ratio <= 0.0:
        raise ValueError(f"{label} ratio must be strictly positive")
    if ratio is not None:
        derived_log = math.log(ratio)
        if log_value is not None and not math.isclose(
            derived_log, log_value, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} ratio/log threshold disagree: log({ratio})={derived_log} vs {log_value}"
            )
        resolved_log = derived_log
    else:
        assert log_value is not None
        resolved_log = log_value
        try:
            ratio = math.exp(resolved_log)
        except OverflowError as exc:
            raise ValueError(f"{label} log threshold is too large") from exc
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(f"{label} log threshold does not yield a finite positive ratio")
    if not math.isfinite(resolved_log):
        raise ValueError(f"{label} log threshold is not finite")
    if require_ratio_at_least_one and ratio < 1.0 - 1e-12:
        raise ValueError(f"{label} ratio must be at least 1.0 for non-inferiority")
    if require_ratio_at_most_one and ratio > 1.0 + 1e-12:
        raise ValueError(f"{label} ratio must be at most 1.0 for superiority")
    source = "+".join(item for item in (ratio_source, log_source) if item)
    return {
        "ratio": float(ratio),
        "log_threshold": float(resolved_log),
        "source": source,
    }


def _phase_expected(plan: Mapping[str, Any], phase: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    global_cfg = plan.get("global", {})
    if not isinstance(global_cfg, Mapping):
        global_cfg = {}
    datasets = [str(item) for item in phase.get("datasets", [])]
    models = [str(item) for item in phase.get("models", [])]
    seeds = [int(item) for item in phase.get("seeds", [])]
    features = str(phase.get("features", global_cfg.get("features", "S")))
    seq_len = int(phase.get("seq_len", global_cfg.get("seq_len", 96)))
    horizons_map = phase.get("horizons_by_dataset")
    if horizons_map is None:
        horizons = [int(item) for item in phase.get("horizons", global_cfg.get("horizons", []))]
        horizons_map = {dataset: horizons for dataset in datasets}
    if not datasets or not models or not seeds:
        raise ValueError(f"phase {phase.get('name')!r} has empty datasets/models/seeds")
    expected: set[tuple[Any, ...]] = set()
    for dataset in datasets:
        values = horizons_map.get(dataset, [])
        if not values:
            raise ValueError(f"phase {phase.get('name')!r} has no horizons for {dataset}")
        for model in models:
            for horizon in values:
                for seed in seeds:
                    coordinate = (dataset, features, seq_len, int(horizon), int(seed), model)
                    if coordinate in expected:
                        raise ValueError(f"duplicate expected phase coordinate: {coordinate}")
                    expected.add(coordinate)
    # A confirmation phase must contain one frozen arm per coordinate.  A
    # lambda grid belongs to selection and would create duplicate coordinates.
    lambdas = phase.get("autocon_lambdas")
    if isinstance(lambdas, list) and len(lambdas) > 1:
        raise ValueError(f"phase {phase.get('name')!r} is a lambda grid, not a frozen phase")
    return expected


def _manifest_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve runner retries by last record while retaining an audit trail.

    The runner appends every attempt and resumes a failed job with the same
    ``job_id``.  Its validator uses the last record for each id, so the
    analyzer must do the same.  Earlier failed attempts are not discarded:
    their count, status, and line number remain in the returned audit.
    """

    final_by_job: dict[str, dict[str, Any]] = {}
    attempts_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            job_id = str(value.get("job_id", ""))
            if not job_id:
                raise ValueError(f"manifest line {line_number} has no job_id")
            value = dict(value)
            value["_manifest_line"] = line_number
            attempts_by_job[job_id].append(value)
            final_by_job[job_id] = value
    records = [final_by_job[job_id] for job_id in sorted(final_by_job)]
    retry_jobs = []
    failed_attempts = []
    for job_id, attempts in sorted(attempts_by_job.items()):
        if len(attempts) > 1:
            retry_jobs.append({
                "job_id": job_id,
                "attempt_count": len(attempts),
                "final_status": attempts[-1].get("status"),
                "prior_statuses": [attempt.get("status") for attempt in attempts[:-1]],
            })
        for attempt in attempts[:-1]:
            if attempt.get("status") != "ok":
                failed_attempts.append({
                    "job_id": job_id,
                    "line": attempt["_manifest_line"],
                    "status": attempt.get("status"),
                })
    return records, {
        "attempt_count": sum(len(attempts) for attempts in attempts_by_job.values()),
        "unique_job_count": len(final_by_job),
        "retry_job_count": len(retry_jobs),
        "retry_jobs": retry_jobs,
        "prior_non_ok_attempt_count": len(failed_attempts),
        "prior_non_ok_attempts": failed_attempts,
    }


def _phase_from_plan(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan has no phases list")
    for phase in phases:
        if isinstance(phase, Mapping) and str(phase.get("name")) == name:
            return phase
    raise ValueError(f"phase not found in plan: {name}")


def _bool_value(value: Any, default: bool = False) -> bool:
    """Parse persisted plan booleans without treating ``\"false\"`` as true."""

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
    raise ValueError(f"plan boolean is not canonical: {value!r}")


def _confirmation_phase_names(plan: Mapping[str, Any]) -> set[str]:
    global_cfg = plan.get("global", {})
    if not isinstance(global_cfg, Mapping):
        global_cfg = {}
    global_skip_test = _bool_value(global_cfg.get("skip_test", False))
    names: set[str] = set()
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan has no phases list")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("plan phase is not an object")
        name = str(phase.get("name", "")).strip()
        if not name:
            raise ValueError("plan phase has no name")
        if name in names:
            raise ValueError(f"duplicate plan phase name: {name}")
        if not _bool_value(phase.get("skip_test", global_skip_test), global_skip_test):
            names.add(name)
    if not names:
        raise ValueError("plan has no test-producing confirmation phases")
    return names


_LAUNCH_AUTHORIZED_STATUSES = frozenset({"approved_for_launch", "launch_authorized"})
_LAUNCH_AUTHORIZATION_KINDS = frozenset({
    "training_plan_launch_authorization",
    "mamba_acf_confirmation_launch_authorization",
})


def _normalized_launch_evidence(value: Any) -> dict[str, str]:
    """Mirror runner sidecar evidence normalization for archived artifacts."""

    if not isinstance(value, Mapping):
        raise ValueError("archived launch sidecar lacks gate-evidence hashes")
    normalized: dict[str, str] = {}
    for gate, item in value.items():
        if not isinstance(gate, str) or not gate.strip():
            raise ValueError("archived launch sidecar has an invalid gate-evidence key")
        digest = item.get("sha256") if isinstance(item, Mapping) else item
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("archived launch sidecar has an invalid gate-evidence hash")
        digest = digest.lower()
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("archived launch sidecar has an invalid gate-evidence hash")
        normalized[gate.strip()] = digest
    return normalized


def _validate_analysis_authorization(
    plan: Mapping[str, Any], plan_path: Path
) -> dict[str, Any] | None:
    """Reject statistics on an unapproved strict confirmation plan.

    Exploratory plans intentionally have no ``confirmation_contract`` and keep
    their historical analysis paths.  For the strict contract, analysis is a
    post-launch operation and therefore requires the same explicit approval
    state as the runner.
    """

    if not isinstance(plan.get("confirmation_contract"), Mapping):
        return None
    if not isinstance(plan.get("statistical_gate"), Mapping):
        raise ValueError("confirmation plan requires a statistical_gate object")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("confirmation plan requires an execution object")
    if execution.get("authorized") is not True:
        raise ValueError(
            "confirmation statistics are blocked: plan execution.authorized is not true"
        )
    execution_status = execution.get("status")
    plan_status = plan.get("status")
    if not isinstance(execution_status, str) or not execution_status.strip():
        raise ValueError("confirmation plan requires execution.status for analysis")
    if not isinstance(plan_status, str) or not plan_status.strip():
        raise ValueError("confirmation plan requires top-level status for analysis")
    if execution_status != execution_status.strip() or plan_status != plan_status.strip():
        raise ValueError("confirmation plan status must not have surrounding whitespace")
    if execution_status != plan_status:
        raise ValueError("confirmation plan execution.status disagrees with status")
    if execution_status not in _LAUNCH_AUTHORIZED_STATUSES:
        raise ValueError(
            "confirmation statistics are blocked: plan status is not launch-approved"
        )
    return {
        "plan_path": str(plan_path.expanduser().resolve()),
        "plan_sha256": _sha256(plan_path.expanduser().resolve()),
        "status": execution_status,
        "authorized": True,
    }


def _validate_dataset_binding(
    plan: Mapping[str, Any], phase: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    fingerprints = plan.get("data_fingerprints", {})
    if not isinstance(fingerprints, Mapping):
        fingerprints = {}
    expected: dict[str, str] = {}
    for dataset in phase.get("datasets", []):
        item = fingerprints.get(str(dataset))
        if not isinstance(item, Mapping):
            raise ValueError(f"plan has no fingerprint for phase dataset {dataset}")
        relative = item.get("relative_path")
        declared = item.get("sha256")
        if relative is None or declared is None:
            raise ValueError(f"incomplete fingerprint for phase dataset {dataset}")
        data_path = _resolve_path(str(relative))
        if not data_path.is_file():
            raise ValueError(f"declared dataset is missing: {data_path}")
        actual = _sha256(data_path)
        if actual != str(declared):
            raise ValueError(f"dataset hash differs from plan for {dataset}")
        expected[str(dataset)] = actual
    recorded = provenance.get("dataset_hashes", {})
    if not isinstance(recorded, Mapping):
        raise ValueError("provenance dataset_hashes is not an object")
    if expected and dict(recorded) != expected:
        raise ValueError("provenance dataset hashes do not match the plan/current data")


def _validate_selected_lambda_binding(
    root: Path,
    phase: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, float] | None:
    """Validate the immutable selection copy embedded in a confirm phase.

    The launch runner performs a full replay before creating provenance.  The
    statistics stage still checks the copied inputs and compact binding so a
    forged/partial phase validation cannot turn an unbound lambda into a
    confirmatory result.
    """

    if phase.get("autocon_lambda") != "selected":
        return None
    binding = provenance.get("selection_provenance")
    if not isinstance(binding, Mapping):
        raise ValueError("selected phase has no selection provenance binding")
    selection_copy_value = provenance.get("selection_copy")
    sidecar_copy_value = provenance.get("selection_provenance_copy")
    if not selection_copy_value or not sidecar_copy_value:
        raise ValueError("selected phase is missing copied selection artifacts")
    selection_copy = Path(str(selection_copy_value)).expanduser().resolve()
    sidecar_copy = Path(str(sidecar_copy_value)).expanduser().resolve()
    if not _inside(selection_copy, root) or not selection_copy.is_file():
        raise ValueError("selection copy is missing/outside phase root")
    if not _inside(sidecar_copy, root) or not sidecar_copy.is_file():
        raise ValueError("selection provenance copy is missing/outside phase root")
    selection_hash = _sha256(selection_copy)
    sidecar_hash = _sha256(sidecar_copy)
    if provenance.get("selection_sha256") != selection_hash:
        raise ValueError("selection copy hash differs from phase provenance")
    if provenance.get("selection_provenance_sha256") != sidecar_hash:
        raise ValueError("selection provenance copy hash differs from phase provenance")
    if provenance.get("selection_binding_sha256") != _canonical_hash(binding):
        raise ValueError("selection binding hash differs from phase provenance")

    selection = _read_json(selection_copy)
    selected = selection.get("selected_lambda_by_model")
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError("selection copy has no selected_lambda_by_model mapping")
    expected_phase = str(phase.get("selection_phase", "")).strip()
    if not expected_phase or str(selection.get("phase", "")) != expected_phase:
        raise ValueError("selection copy phase does not match confirm phase")
    expected_features = str(phase.get("features", "S"))
    if str(selection.get("features", expected_features)) != expected_features:
        raise ValueError("selection copy features do not match confirm phase")
    expected_models = {
        str(model) for model in phase.get("models", [])
        if str(model).endswith("AutoCon")
    }
    if set(str(key) for key in selected) != expected_models:
        raise ValueError("selection copy models do not match selected confirm models")
    normalized: dict[str, float] = {}
    for model, value in selected.items():
        number = _finite(value, f"selected lambda {model}")
        if number <= 0.0:
            raise ValueError(f"selected lambda {model} must be positive")
        normalized[str(model)] = number
    selected_hash = binding.get("selected_lambda_by_model_sha256")
    if selected_hash != _canonical_hash(selected):
        raise ValueError("selection binding does not hash the copied selected lambdas")
    if binding.get("selection_provenance_publication_ready") is not False:
        raise ValueError("selection binding must remain non-publication-ready")
    if not binding.get("selection_phase_validation_sha256") or not binding.get(
        "selection_manifest_sha256"
    ):
        raise ValueError("selection binding lacks validation/manifest hashes")
    return normalized


def _validate_analyzer_snapshot(
    snapshot: Path, provenance: Mapping[str, Any]
) -> dict[str, str]:
    """Bind this live analyzer to the frozen copy used for the phase.

    A valid result cannot be reinterpreted by a later edited analyzer.  The
    runner includes this file in the source snapshot, and both its manifest
    hash and the executable file hash must agree with the current process.
    """

    relative = "scripts/analyze_mamba_acf_confirmation.py"
    recorded_files = provenance.get("source_files")
    if not isinstance(recorded_files, Mapping):
        raise ValueError("source snapshot has no file hash map for analyzer binding")
    recorded = recorded_files.get(relative)
    snapshot_script = snapshot / relative
    if not isinstance(recorded, str) or not recorded or not snapshot_script.is_file():
        raise ValueError("source snapshot lacks the confirmation analyzer script hash")
    snapshot_hash = _sha256(snapshot_script)
    if snapshot_hash != recorded:
        raise ValueError("snapshot confirmation analyzer hash differs from provenance")
    live_script = Path(__file__).resolve()
    live_hash = _sha256(live_script)
    if live_hash != snapshot_hash:
        raise ValueError(
            "live confirmation analyzer differs from frozen source snapshot; "
            "run the snapshot-matched analyzer before interpreting artifacts"
        )
    return {
        "relative_path": relative,
        "snapshot_sha256": snapshot_hash,
        "live_sha256": live_hash,
    }


def _validate_phase(
    root: Path, plan_path: Path, plan: Mapping[str, Any], phase_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"phase root does not exist: {root}")
    phase = _phase_from_plan(plan, phase_name)
    registered_root = _resolve_path(str(phase.get("output_root", "")))
    if registered_root != root:
        raise ValueError(f"phase output_root mismatch for {phase_name}: {registered_root} != {root}")

    validation_path = root / "phase_validation.json"
    validation = _read_json(validation_path)
    required_validation = {
        "valid", "phase", "plan_sha256", "manifest", "manifest_sha256",
        "provenance", "provenance_id", "expected_jobs", "manifest_unique_jobs",
        "artifact_hashes_by_job",
    }
    missing = sorted(required_validation - set(validation))
    if missing:
        raise ValueError(f"phase validation missing fields: {missing}")
    if validation.get("valid") is not True:
        raise ValueError(f"phase validation is not valid: {validation_path}")
    if str(validation.get("phase")) != phase_name:
        raise ValueError("phase validation name mismatch")
    plan_hash = _sha256(plan_path)
    if validation.get("plan_sha256") != plan_hash:
        raise ValueError("plan changed after phase validation")
    if validation.get("errors") not in (None, []):
        raise ValueError("phase validation contains errors")
    if int(validation.get("malformed_manifest_lines", 0)) != 0:
        raise ValueError("phase validation recorded malformed manifest lines")

    manifest_path = Path(str(validation["manifest"])).expanduser().resolve()
    if not _inside(manifest_path, root) or not manifest_path.is_file():
        raise ValueError(f"validated manifest is missing/outside phase root: {manifest_path}")
    if validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("manifest changed after phase validation")
    records, manifest_audit = _manifest_records(manifest_path)
    expected = _phase_expected(plan, phase)
    expected_jobs = int(validation["expected_jobs"])
    if expected_jobs != len(expected):
        raise ValueError(
            f"phase validation expected_jobs={expected_jobs} disagrees with plan expansion={len(expected)}"
        )
    if len(records) != expected_jobs or int(validation["manifest_unique_jobs"]) != expected_jobs:
        raise ValueError("manifest coverage is incomplete or duplicated")

    provenance_path = Path(str(validation["provenance"])).expanduser().resolve()
    if not _inside(provenance_path, root) or not provenance_path.is_file():
        raise ValueError(f"validated provenance is missing/outside phase root: {provenance_path}")
    provenance = _read_json(provenance_path)
    if provenance.get("phase") != phase_name:
        raise ValueError("provenance phase mismatch")
    if provenance.get("provenance_id") != validation.get("provenance_id"):
        raise ValueError("validation/provenance ID mismatch")
    if provenance.get("plan_sha256") != plan_hash:
        raise ValueError("provenance plan hash mismatch")
    # Strict confirmation artifacts must carry the approved launch binding;
    # otherwise a hand-created/legacy phase cannot be promoted by statistics.
    if isinstance(plan.get("confirmation_contract"), Mapping):
        launch = provenance.get("launch_authorization")
        if not isinstance(launch, Mapping):
            raise ValueError("confirmation provenance has no launch authorization binding")
        if launch.get("authorized") is not True:
            raise ValueError("confirmation provenance launch authorization is false")
        if launch.get("plan_sha256") != plan_hash:
            raise ValueError("confirmation provenance launch plan hash mismatch")
        if launch.get("plan_status") not in _LAUNCH_AUTHORIZED_STATUSES:
            raise ValueError("confirmation provenance launch status is not approved")
        binding_hash = provenance.get("launch_authorization_binding_sha256")
        if binding_hash != _canonical_hash(launch):
            raise ValueError("confirmation launch authorization binding hash mismatch")
        copy_value = provenance.get("launch_authorization_copy")
        copy_hash = provenance.get("launch_authorization_copy_sha256")
        if not copy_value or not copy_hash:
            raise ValueError("confirmation provenance lacks archived launch sidecar binding")
        copy_path = Path(str(copy_value)).expanduser().resolve()
        if not _inside(copy_path, root) or not copy_path.is_file():
            raise ValueError("archived confirmation launch sidecar is missing/outside phase root")
        if _sha256(copy_path) != copy_hash:
            raise ValueError("archived confirmation launch sidecar hash mismatch")
        if launch.get("sidecar_sha256") != copy_hash:
            raise ValueError("archived launch sidecar hash differs from launch binding")
        archived_sidecar = _read_json(copy_path)
        if archived_sidecar.get("schema_version") != 1:
            raise ValueError("archived launch sidecar schema is unsupported")
        if archived_sidecar.get("kind") not in _LAUNCH_AUTHORIZATION_KINDS:
            raise ValueError("archived launch sidecar kind is unsupported")
        if launch.get("sidecar_kind") and archived_sidecar.get("kind") != launch.get("sidecar_kind"):
            raise ValueError("archived launch sidecar kind differs from launch binding")
        if archived_sidecar.get("plan_sha256") != plan_hash:
            raise ValueError("archived launch sidecar plan hash mismatch")
        if archived_sidecar.get("plan_status") != launch.get("plan_status"):
            raise ValueError("archived launch sidecar status differs from launch binding")
        approval = archived_sidecar.get("approval")
        bound_approval = launch.get("approval")
        if not isinstance(approval, Mapping) or approval.get("approved") is not True:
            raise ValueError("archived launch sidecar is not approved")
        if archived_sidecar.get("authorized") is False:
            raise ValueError("archived launch sidecar is explicitly unauthorized")
        if not isinstance(bound_approval, Mapping) or any(
            approval.get(key) != bound_approval.get(key)
            for key in ("approved", "approved_by", "approved_at_utc")
        ):
            raise ValueError("archived launch sidecar approval differs from launch binding")
        evidence = _normalized_launch_evidence(
            archived_sidecar.get("prelaunch_gate_evidence_hashes")
        )
        if evidence != launch.get("prelaunch_gate_evidence_hashes"):
            raise ValueError("archived launch sidecar gate evidence differs from launch binding")
        required_gates = launch.get("required_prelaunch_gates")
        if not isinstance(required_gates, list) or any(gate not in evidence for gate in required_gates):
            raise ValueError("archived launch sidecar lacks required gate evidence")
    if provenance.get("source_tree_sha256") is None or not provenance.get("source_files"):
        raise ValueError("provenance has no source snapshot attestation")
    snapshot = Path(str(provenance.get("source_snapshot", ""))).expanduser().resolve()
    if not _inside(snapshot, root) or not snapshot.is_dir():
        raise ValueError(f"source snapshot is missing/outside phase root: {snapshot}")
    source_files, source_tree_hash = _hash_file_map(snapshot)
    if source_files != provenance.get("source_files") or source_tree_hash != provenance.get("source_tree_sha256"):
        raise ValueError("source snapshot changed after phase validation")
    analyzer_binding = _validate_analyzer_snapshot(snapshot, provenance)
    plan_copy = snapshot.parent / "final_training_plan.json"
    if plan_copy.is_file() and _sha256(plan_copy) != plan_hash:
        raise ValueError("provenance plan copy changed")
    _validate_dataset_binding(plan, phase, provenance)
    selected_lambdas = _validate_selected_lambda_binding(root, phase, provenance)

    expected_hashes = validation.get("artifact_hashes_by_job")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("phase validation has no artifact hash map")
    manifest_ids = {str(record["job_id"]) for record in records}
    if set(str(key) for key in expected_hashes) != manifest_ids:
        raise ValueError("artifact hash coverage differs from manifest")
    result_root = (root / "results").resolve()
    if not result_root.is_dir():
        raise ValueError(f"phase results root is missing: {result_root}")

    rows: list[dict[str, Any]] = []
    observed: set[tuple[Any, ...]] = set()
    manifest_config_paths: set[Path] = set()
    for record in records:
        job_id = str(record["job_id"])
        if record.get("status") != "ok":
            raise ValueError(f"manifest job is not successful: {job_id}")
        if str(record.get("phase")) != phase_name:
            raise ValueError(f"manifest phase mismatch: {job_id}")
        artifact_value = record.get("expected_artifact")
        if not artifact_value:
            raise ValueError(f"manifest job has no expected_artifact: {job_id}")
        metrics_path = Path(str(artifact_value)).expanduser().resolve()
        result_dir = metrics_path.parent
        if metrics_path.name != "metrics.npy" or not _inside(result_dir, result_root) or not metrics_path.is_file():
            raise ValueError(f"invalid/missing metrics artifact for {job_id}")
        checkpoint_root_value = _command_arg(record.get("command", []), "--checkpoints")
        checkpoint_dir = None
        if checkpoint_root_value:
            checkpoint_root = Path(checkpoint_root_value).expanduser().resolve()
            if not _inside(checkpoint_root, root):
                raise ValueError(f"checkpoint root outside phase root: {job_id}")
            checkpoint_dir = checkpoint_root / result_dir.name
        actual_hashes = _artifact_hashes(result_dir, checkpoint_dir)
        if dict(actual_hashes) != dict(expected_hashes[job_id]):
            raise ValueError(f"artifact hash mismatch after validation: {job_id}")
        if record.get("artifact_hashes") is not None and dict(record["artifact_hashes"]) != actual_hashes:
            raise ValueError(f"manifest artifact hashes mismatch: {job_id}")

        config_path = result_dir / "config.json"
        if not config_path.is_file():
            raise ValueError(f"missing config artifact: {job_id}")
        config = _read_json(config_path)
        dataset = str(record.get("dataset", ""))
        features = str(record.get("features", ""))
        seq_len = int(record.get("seq_len", config.get("seq_len", -1)))
        horizon = int(record.get("pred_len", -1))
        seed = int(record.get("seed", -1))
        model = str(record.get("model", ""))
        coordinate = (dataset, features, seq_len, horizon, seed, model)
        if coordinate in observed:
            raise ValueError(f"duplicate result coordinate: {coordinate}")
        observed.add(coordinate)
        if coordinate not in expected:
            raise ValueError(f"result coordinate is not in plan phase: {coordinate}")
        if str(config.get("data")) != dataset or str(config.get("features")) != features:
            raise ValueError(f"config coordinate mismatch: {job_id}")
        if int(config.get("seq_len", -1)) != seq_len or int(config.get("pred_len", -1)) != horizon:
            raise ValueError(f"config horizon/sequence mismatch: {job_id}")
        config_seed = config.get("run_seed", config.get("seed"))
        if config_seed is None or int(config_seed) != seed:
            raise ValueError(f"config seed mismatch: {job_id}")
        if config.get("plan_job_id") not in (None, job_id):
            raise ValueError(f"config plan_job_id mismatch: {job_id}")
        if bool(config.get("skip_test", False)):
            raise ValueError(f"selection artifact passed as confirmation input: {job_id}")
        if record.get("provenance_id") != provenance.get("provenance_id"):
            raise ValueError(f"record provenance mismatch: {job_id}")
        if record.get("source_tree_sha256_before") != provenance.get("source_tree_sha256"):
            raise ValueError(f"record source hash before mismatch: {job_id}")
        if record.get("source_tree_sha256_after") != provenance.get("source_tree_sha256"):
            raise ValueError(f"record source hash after mismatch: {job_id}")
        if selected_lambdas is not None:
            if model not in selected_lambdas:
                raise ValueError(f"selected lambda is missing for {model}: {job_id}")
            expected_lambda = selected_lambdas[model]
            try:
                config_lambda = _finite(
                    config.get("AutoCon_lambda"), f"{job_id} AutoCon_lambda"
                )
                record_lambda = _finite(
                    record.get("autocon_lambda"), f"{job_id} manifest autocon_lambda"
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if not math.isclose(config_lambda, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"config selected lambda mismatch: {job_id}")
            if not math.isclose(record_lambda, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"manifest selected lambda mismatch: {job_id}")

        relation_metadata = _relation_log_contract(config, checkpoint_dir, job_id)
        memory_metadata = _memory_lag_metadata(config, job_id)
        _validate_memory_contract(
            plan, phase_name, model, config, memory_metadata, job_id
        )

        try:
            values = np.asarray(np.load(metrics_path, allow_pickle=False), dtype=float).reshape(-1)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"cannot load metrics artifact: {metrics_path}: {exc}") from exc
        if values.size <= METRIC_INDEX:
            raise ValueError(f"metrics.npy has no MSE entry: {metrics_path}")
        mse = _finite(values[METRIC_INDEX], f"{job_id} metrics.npy mse")
        if mse <= 0.0:
            raise ValueError(f"MSE must be strictly positive for log ratio: {job_id}")
        rows.append({
            "phase": phase_name,
            "dataset": dataset,
            "features": features,
            "seq_len": seq_len,
            "pred_len": horizon,
            "seed": seed,
            "model": model,
            "mse": mse,
            "job_id": job_id,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "relation": relation_metadata,
            "memory": memory_metadata,
        })
        manifest_config_paths.add(config_path.resolve())

    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"phase coordinate coverage mismatch: missing={missing[:3]}, extra={extra[:3]}")
    discovered = {path.resolve() for path in result_root.rglob("config.json")}
    if discovered != manifest_config_paths:
        raise ValueError("unregistered or missing result config files under phase results root")
    meta = {
        "phase": phase_name,
        "root": str(root),
        "plan_sha256": plan_hash,
        "validation": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_attempt_audit": manifest_audit,
        "provenance": str(provenance_path),
        "provenance_id": provenance.get("provenance_id"),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "analyzer_binding": analyzer_binding,
        "expected_jobs": expected_jobs,
        "observed_jobs": len(rows),
    }
    return rows, meta


def _split_endpoint(token: str) -> tuple[str, str | None]:
    token = token.strip()
    for separator in (":", "/"):
        if separator in token:
            phase, model = token.split(separator, 1)
            return phase.strip(), model.strip() or None
    return token, None


def _resolve_endpoint(token: str, plan: Mapping[str, Any]) -> tuple[str, str]:
    phase_token, model_token = _split_endpoint(token)
    phases = [phase for phase in plan.get("phases", []) if isinstance(phase, Mapping)]
    names = {str(phase.get("name")) for phase in phases}
    if phase_token in names:
        phase = next(phase for phase in phases if str(phase.get("name")) == phase_token)
        models = [str(item) for item in phase.get("models", [])]
        if model_token is None:
            if len(models) != 1:
                raise ValueError(f"endpoint {token!r} needs an explicit phase:model")
            model_token = models[0]
        if model_token not in models:
            raise ValueError(f"model {model_token!r} is not registered in phase {phase_token!r}")
        return phase_token, model_token
    if model_token is not None:
        raise ValueError(f"unknown phase in endpoint: {phase_token!r}")
    candidates = []
    for phase in phases:
        for model in phase.get("models", []):
            if str(model) == phase_token:
                candidates.append((str(phase.get("name")), str(model)))
    if len(candidates) != 1:
        raise ValueError(
            f"endpoint {token!r} is not a unique phase or model; use phase:model"
        )
    return candidates[0]


def _parse_contrast_spec(spec: str, plan: Mapping[str, Any]) -> dict[str, str]:
    value = spec.strip()
    label = None
    if "=" in value:
        label, value = value.split("=", 1)
        label = label.strip() or None
    parts = value.split("::")
    if len(parts) != 2:
        raise ValueError(
            "contrast must be 'phase:model::phase:model' or 'label=phase:model::phase:model'"
        )
    treatment_phase, treatment_model = _resolve_endpoint(parts[0], plan)
    reference_phase, reference_model = _resolve_endpoint(parts[1], plan)
    if label is None:
        label = f"{treatment_phase}:{treatment_model} versus {reference_phase}:{reference_model}"
    return {
        "label": label,
        "treatment_phase": treatment_phase,
        "treatment_model": treatment_model,
        "reference_phase": reference_phase,
        "reference_model": reference_model,
    }


def _plan_contrasts(plan: Mapping[str, Any], explicit: Sequence[str]) -> list[dict[str, str]]:
    specs = list(explicit)
    if not specs:
        gate = plan.get("statistical_gate", {})
        if not isinstance(gate, Mapping) or not isinstance(gate.get("primary_comparisons"), list):
            raise ValueError("plan has no statistical_gate.primary_comparisons")
        specs = [str(item) for item in gate["primary_comparisons"]]
    contrasts: list[dict[str, str]] = []
    labels: set[str] = set()
    for spec in specs:
        if " versus " in spec and "::" not in spec and "=" not in spec:
            left, right = spec.split(" versus ", 1)
            parsed = _parse_contrast_spec(f"{left.strip()}::{right.strip()}", plan)
        else:
            parsed = _parse_contrast_spec(spec, plan)
        if parsed["label"] in labels:
            raise ValueError(f"duplicate contrast label: {parsed['label']}")
        labels.add(parsed["label"])
        contrasts.append(parsed)
    return contrasts


def _holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Return Holm step-down adjusted p-values, preserving input positions."""
    finite = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    output: list[float | None] = [None] * len(p_values)
    running = 0.0
    total = len(finite)
    for rank, (index, value) in enumerate(sorted(finite, key=lambda item: (item[1], item[0]))):
        running = max(running, min(1.0, (total - rank) * value))
        output[index] = running
    return output


def _wilcoxon_less_p(values: Sequence[float]) -> float:
    """Exact Wilcoxon signed-rank p-value for ``H1: median(values) < 0``.

    Zero differences are discarded (the conventional ``zero_method=wilcox``).
    A small integer dynamic program over doubled midranks avoids asymptotic
    approximations while remaining practical for the five-seed confirmation
    contract and larger modest seed counts.
    """
    numbers = np.asarray(values, dtype=float).reshape(-1)
    if numbers.size == 0 or not np.all(np.isfinite(numbers)):
        raise ValueError("Wilcoxon input must be finite and non-empty")
    nonzero = numbers[numbers != 0.0]
    if nonzero.size == 0:
        return 1.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    doubled = np.rint(2.0 * ranks).astype(int)
    observed = int(np.sum(doubled[nonzero > 0.0]))
    total_rank = int(np.sum(doubled))
    counts = np.zeros(total_rank + 1, dtype=np.int64)
    counts[0] = 1
    for rank in doubled:
        # Descending scalar updates avoid overlapping NumPy slices and count
        # each sign assignment exactly once, including tied midranks.
        for total in range(total_rank, int(rank) - 1, -1):
            counts[total] += counts[total - int(rank)]
    denominator = float(2 ** int(doubled.size))
    return float(np.sum(counts[: observed + 1]) / denominator)


def _paired_t_upper(values: Sequence[float], alpha: float) -> dict[str, float | int | None]:
    numbers = np.asarray(values, dtype=float).reshape(-1)
    if numbers.size == 0 or not np.all(np.isfinite(numbers)):
        raise ValueError("paired t input must be finite and non-empty")
    n = int(numbers.size)
    mean = float(np.mean(numbers))
    if n < 2:
        return {"n_pairs": n, "mean": mean, "sd": None, "upper": None, "p_less": None}
    sd = float(np.std(numbers, ddof=1))
    if sd == 0.0:
        upper = mean
        p_less = 0.0 if mean < 0.0 else 1.0
    else:
        sem = sd / math.sqrt(n)
        t_stat = mean / sem
        upper = mean + float(stats.t.ppf(1.0 - alpha, n - 1)) * sem
        p_less = float(stats.t.cdf(t_stat, n - 1))
    return {
        "n_pairs": n,
        "mean": mean,
        "sd": sd,
        "upper": float(upper),
        "p_less": float(p_less),
    }


def _pair_rows(
    treatment: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row["dataset"], row["features"], int(row["seq_len"]), int(row["pred_len"]), int(row["seed"]))

    left: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    right: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in treatment:
        coordinate = key(row)
        if coordinate in left:
            raise ValueError(f"duplicate treatment pairing coordinate: {coordinate}")
        left[coordinate] = row
    for row in reference:
        coordinate = key(row)
        if coordinate in right:
            raise ValueError(f"duplicate reference pairing coordinate: {coordinate}")
        right[coordinate] = row
    if set(left) != set(right):
        missing_reference = sorted(set(left) - set(right))
        missing_treatment = sorted(set(right) - set(left))
        raise ValueError(
            "unpaired treatment/reference coordinates: "
            f"missing_reference={missing_reference[:3]}, missing_treatment={missing_treatment[:3]}"
        )
    rows = []
    for coordinate in sorted(left):
        treatment_mse = _finite(left[coordinate]["mse"], f"treatment MSE {coordinate}")
        reference_mse = _finite(right[coordinate]["mse"], f"reference MSE {coordinate}")
        if treatment_mse <= 0.0 or reference_mse <= 0.0:
            raise ValueError(f"MSE must be strictly positive for log ratio: {coordinate}")
        rows.append({
            "dataset": coordinate[0],
            "features": coordinate[1],
            "seq_len": coordinate[2],
            "pred_len": coordinate[3],
            "seed": coordinate[4],
            "treatment_mse": treatment_mse,
            "reference_mse": reference_mse,
            "log_ratio": float(math.log(treatment_mse / reference_mse)),
            "raw_delta": float(treatment_mse - reference_mse),
            "treatment_job_id": left[coordinate].get("job_id"),
            "reference_job_id": right[coordinate].get("job_id"),
        })
    return rows


def _analyze_contrast(
    contrast: Mapping[str, str],
    rows_by_phase: Mapping[str, Sequence[Mapping[str, Any]]],
    m_cells: int,
    alpha: float,
    noninferiority: Mapping[str, float | str],
    superiority: Mapping[str, float | str],
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    def select(phase: str, model: str) -> list[Mapping[str, Any]]:
        selected = [row for row in rows_by_phase[phase] if str(row["model"]) == model]
        if not selected:
            raise ValueError(f"no rows for {phase}:{model}")
        return selected

    treatment_rows = select(contrast["treatment_phase"], contrast["treatment_model"])
    reference_rows = select(contrast["reference_phase"], contrast["reference_model"])
    attribution = (
        _peak_placebo_contract(plan, contrast, treatment_rows, reference_rows)
        if plan is not None
        else None
    )
    paired = _pair_rows(treatment_rows, reference_rows)
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        cell = (row["dataset"], row["features"], row["seq_len"], row["pred_len"])
        cells[cell].append(row)
    if len(cells) != m_cells:
        raise ValueError(
            f"contrast {contrast['label']!r} has {len(cells)} cells but fixed m_cells={m_cells}"
        )
    adjusted_alpha = alpha / float(m_cells)
    if not 0.0 < adjusted_alpha < 1.0:
        raise ValueError("Bonferroni adjusted alpha is outside (0, 1)")
    prelim: list[dict[str, Any]] = []
    for cell in sorted(cells):
        observations = sorted(cells[cell], key=lambda item: int(item["seed"]))
        seeds = [int(item["seed"]) for item in observations]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seed in contrast cell: {cell}")
        logs = [float(item["log_ratio"]) for item in observations]
        raw = [float(item["raw_delta"]) for item in observations]
        t_result = _paired_t_upper(logs, adjusted_alpha)
        if t_result["n_pairs"] < 2:
            raise ValueError(f"at least two paired seeds are required for cell: {cell}")
        upper = t_result["upper"]
        if upper is None:
            raise ValueError(f"one-sided upper bound is unavailable for cell: {cell}")
        noninferiority_pass = bool(
            float(upper) <= float(noninferiority["log_threshold"])
        )
        # Superiority is strict: an upper bound exactly at zero is compatible
        # with equality and therefore cannot support a directional advantage.
        superiority_pass = bool(
            float(upper) < float(superiority["log_threshold"])
        )
        prelim.append({
            "dataset": cell[0],
            "features": cell[1],
            "seq_len": int(cell[2]),
            "pred_len": int(cell[3]),
            "coverage": {
                "expected_seed_count": len(seeds),
                "observed_seed_count": len(seeds),
                "seeds": seeds,
                "complete": True,
            },
            "n_pairs": int(t_result["n_pairs"]),
            "mean_log_ratio": float(t_result["mean"]),
            "sd_log_ratio": t_result["sd"],
            "raw_delta_mean": float(np.mean(raw)),
            "raw_delta_sd": float(np.std(raw, ddof=1)) if len(raw) > 1 else None,
            "one_sided_t_upper_bound": upper,
            "one_sided_t_p_less": t_result["p_less"],
            "wilcoxon_exact_p_less": _wilcoxon_less_p(logs),
            "noninferiority_threshold": {
                "ratio": float(noninferiority["ratio"]),
                "log_threshold": float(noninferiority["log_threshold"]),
            },
            "superiority_threshold": {
                "ratio": float(superiority["ratio"]),
                "log_threshold": float(superiority["log_threshold"]),
            },
            "noninferiority_pass": noninferiority_pass,
            "superiority_pass": superiority_pass,
            "seed_observations": observations,
        })
    holm = _holm_adjust([row["one_sided_t_p_less"] for row in prelim])
    for index, row in enumerate(prelim):
        row["holm_adjusted_p_less"] = holm[index]
        row["interval_alpha"] = float(adjusted_alpha)
        row["interval_confidence"] = float(1.0 - adjusted_alpha)
    noninferiority_pass_cells = sum(bool(row["noninferiority_pass"]) for row in prelim)
    superiority_pass_cells = sum(bool(row["superiority_pass"]) for row in prelim)
    return {
        "label": contrast["label"],
        "treatment": {
            "phase": contrast["treatment_phase"],
            "model": contrast["treatment_model"],
        },
        "reference": {
            "phase": contrast["reference_phase"],
            "model": contrast["reference_model"],
        },
        "m_cells": int(m_cells),
        "alpha": float(alpha),
        "adjusted_alpha": float(adjusted_alpha),
        "coverage": {
            "expected_cells": int(m_cells),
            "observed_cells": len(prelim),
            "complete": len(prelim) == m_cells,
            "expected_seed_count_per_cell": sorted({row["coverage"]["expected_seed_count"] for row in prelim}),
        },
        "thresholds": {
            "noninferiority": {
                "ratio": float(noninferiority["ratio"]),
                "log_threshold": float(noninferiority["log_threshold"]),
                "rule": "upper_bound <= log_threshold",
            },
            "superiority": {
                "ratio": float(superiority["ratio"]),
                "log_threshold": float(superiority["log_threshold"]),
                "rule": "upper_bound < log_threshold",
            },
        },
        "gate_coverage": {
            "cell_count": len(prelim),
            "noninferiority_pass_cells": noninferiority_pass_cells,
            "noninferiority_failed_cells": len(prelim) - noninferiority_pass_cells,
            "noninferiority_all_cells_pass": noninferiority_pass_cells == len(prelim),
            "superiority_pass_cells": superiority_pass_cells,
            "superiority_failed_cells": len(prelim) - superiority_pass_cells,
            "superiority_all_cells_pass": superiority_pass_cells == len(prelim),
            "complete_cell_coverage": len(prelim) == m_cells,
        },
        **({"lag_attribution": attribution} if attribution is not None else {}),
        "cells": prelim,
    }


def _statistical_contract(
    plan: Mapping[str, Any], cell_count: int
) -> tuple[float, int, str, dict[str, float | str], dict[str, float | str]]:
    gate = plan.get("statistical_gate", {})
    if not isinstance(gate, Mapping):
        raise ValueError("plan has no statistical_gate object")
    alpha_value = gate.get("interval_alpha", gate.get("alpha", 0.05))
    alpha = _finite(alpha_value, "statistical_gate.interval_alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("interval alpha must be in (0, 1)")
    m_value = gate.get("simultaneous_m_cells", cell_count)
    try:
        m_cells = int(m_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("simultaneous_m_cells must be a positive integer") from exc
    if m_cells <= 0:
        raise ValueError("simultaneous_m_cells must be positive")
    if m_cells != cell_count:
        raise ValueError(
            f"fixed simultaneous_m_cells={m_cells} disagrees with plan cell count={cell_count}"
        )
    rule = str(gate.get("interval_construction_rule", "")).strip()
    if not rule:
        raise ValueError("plan must freeze statistical_gate.interval_construction_rule")
    if "Bonferroni" not in rule and "bonferroni" not in rule:
        raise ValueError("interval construction rule must explicitly identify Bonferroni adjusted alpha")
    noninferiority = _threshold_contract(
        gate,
        primary_key="noninferiority_margin",
        ratio_keys=(
            "noninferiority_ratio",
            "noninferiority_margin_ratio",
            "noninferiority_threshold_ratio",
        ),
        log_keys=(
            "noninferiority_log_threshold",
            "noninferiority_margin_log_threshold",
        ),
        label="noninferiority margin",
        require_ratio_at_least_one=True,
    )
    superiority = _threshold_contract(
        gate,
        primary_key="superiority_threshold",
        ratio_keys=("superiority_ratio", "superiority_threshold_ratio"),
        log_keys=("superiority_log_threshold",),
        label="superiority threshold",
        require_ratio_at_most_one=True,
    )
    return alpha, m_cells, rule, noninferiority, superiority


def build_analysis(
    plan_path: Path,
    phase_roots: Mapping[str, Path],
    explicit_contrasts: Sequence[str] = (),
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    plan = _read_json(plan_path)
    launch_authorization = _validate_analysis_authorization(plan, plan_path)
    registered_counts = _registered_job_counts_contract(plan)
    if not phase_roots:
        raise ValueError("at least one --phase-root is required")
    expected_confirmation = _confirmation_phase_names(plan)
    supplied_phases = {str(name) for name in phase_roots}
    missing_phases = sorted(expected_confirmation - supplied_phases)
    unexpected_phases = sorted(supplied_phases - expected_confirmation)
    if missing_phases or unexpected_phases:
        raise ValueError(
            "statistics input must include exactly every test-producing confirmation phase; "
            f"missing={missing_phases}, unexpected={unexpected_phases}"
        )
    rows_by_phase: dict[str, list[dict[str, Any]]] = {}
    phase_meta: dict[str, dict[str, Any]] = {}
    for phase_name, root in phase_roots.items():
        if phase_name in rows_by_phase:
            raise ValueError(f"duplicate phase root: {phase_name}")
        rows, meta = _validate_phase(root, plan_path, plan, phase_name)
        rows_by_phase[phase_name] = rows
        phase_meta[phase_name] = meta
    source_hashes = {str(meta.get("source_tree_sha256")) for meta in phase_meta.values()}
    if len(source_hashes) != 1:
        raise ValueError(
            "phase source snapshots differ; paired contrasts require one immutable source tree"
        )
    plan_hashes = {str(meta.get("plan_sha256")) for meta in phase_meta.values()}
    if len(plan_hashes) != 1:
        raise ValueError("phase plan hashes differ")
    contrasts = _plan_contrasts(plan, explicit_contrasts)
    # A strict confirmation plan owns its registered primary comparison set.
    # Allowing a command-line subset/replacement would turn the same frozen
    # artifacts into a different hypothesis family and invalidate the gate.
    if explicit_contrasts and isinstance(plan.get("confirmation_contract"), Mapping):
        raise ValueError(
            "confirmation analysis does not accept --contrast; use the registered "
            "statistical_gate.primary_comparisons set"
        )
    # Determine the fixed family size from the registered plan, rather than
    # from whichever rows happen to be supplied to this invocation.
    all_cells: set[tuple[Any, ...]] = set()
    for rows in rows_by_phase.values():
        all_cells.update((r["dataset"], r["features"], r["seq_len"], r["pred_len"]) for r in rows)
    alpha, m_cells, rule, noninferiority, superiority = _statistical_contract(
        plan, len(all_cells)
    )
    for contrast in contrasts:
        for endpoint in ("treatment_phase", "reference_phase"):
            if contrast[endpoint] not in rows_by_phase:
                raise ValueError(
                    f"contrast {contrast['label']!r} requires phase root {contrast[endpoint]!r}"
                )
    comparison_results = [
        _analyze_contrast(
            contrast, rows_by_phase, m_cells, alpha, noninferiority, superiority, plan
        )
        for contrast in contrasts
    ]
    return {
        "schema_version": 1,
        "status": "validated_statistics_only",
        "analysis": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "plan": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "metric": "metrics.npy[1] (scaled MSE)",
            "inferential_unit": "paired random seed",
            "interval_construction_rule": rule,
            "interval_alpha": alpha,
            "simultaneous_m_cells": m_cells,
            "adjusted_alpha": alpha / m_cells,
            "noninferiority_threshold": noninferiority,
            "superiority_threshold": superiority,
            "launch_authorization": launch_authorization,
            "registered_job_counts": registered_counts,
            "holm_scope": "one-sided paired-t p-values across cells within each named comparison",
            "wilcoxon": "exact signed-rank less alternative; zero differences discarded",
        },
        "phases": phase_meta,
        "contrasts": comparison_results,
        "limitations": [
            "Bonferroni adjusted-alpha intervals are not Holm-inverted confidence intervals",
            "five confirmation seeds remain a small inferential sample",
            "forecast windows and channels are not independent replicates",
            "this output does not authorize a publication claim or alter the frozen plan",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value).replace("|", "\\|")


def _markdown(payload: Mapping[str, Any]) -> str:
    analysis = payload["analysis"]
    lines = [
        "# MambaACF Confirmation Statistics",
        "",
        "This report is emitted only after strict phase-validation, provenance, manifest, and artifact-hash checks.",
        "The inferential unit is the paired random seed.",
        "",
        "## Interval rule",
        "",
        f"- Rule: `{analysis['interval_construction_rule']}`",
        f"- Family size: `m={analysis['simultaneous_m_cells']}`; adjusted alpha: `{_fmt(analysis['adjusted_alpha'])}`.",
        (
            "- Non-inferiority: upper bound <= "
            f"`{_fmt(analysis['noninferiority_threshold']['log_threshold'])}` "
            f"(ratio `{_fmt(analysis['noninferiority_threshold']['ratio'])}`)."
        ),
        (
            "- Superiority: upper bound < "
            f"`{_fmt(analysis['superiority_threshold']['log_threshold'])}` "
            f"(ratio `{_fmt(analysis['superiority_threshold']['ratio'])}`)."
        ),
        "- Holm-adjusted p-values are reported separately; the Bonferroni interval is not a Holm inversion.",
        "",
        "## Cell-level contrasts",
        "",
    ]
    for contrast in payload["contrasts"]:
        lines.extend([
            f"### {contrast['label']}",
            "",
            f"Treatment: `{contrast['treatment']['phase']}:{contrast['treatment']['model']}`; reference: `{contrast['reference']['phase']}:{contrast['reference']['model']}`.",
            "",
        ])
        attribution = contrast.get("lag_attribution")
        if attribution is not None:
            lines.extend([
                "Lag attribution eligibility: `{} ({})`.".format(
                    _fmt(attribution.get("eligible")), attribution.get("kind", "unknown")
                ),
                "",
            ])
        lines.extend([
            "| Dataset | Horizon | n | Mean log ratio | Raw MSE delta | One-sided upper bound | NI pass | Superior pass | Holm p (less) | Exact Wilcoxon p (less) | Coverage |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
        ])
        for row in contrast["cells"]:
            coverage = row["coverage"]
            lines.append(
                "| " + " | ".join([
                    _fmt(row["dataset"]), _fmt(row["pred_len"]), _fmt(row["n_pairs"]),
                    _fmt(row["mean_log_ratio"]), _fmt(row["raw_delta_mean"]),
                    _fmt(row["one_sided_t_upper_bound"]), _fmt(row["noninferiority_pass"]),
                    _fmt(row["superiority_pass"]), _fmt(row["holm_adjusted_p_less"]),
                    _fmt(row["wilcoxon_exact_p_less"]),
                    f"{coverage['observed_seed_count']}/{coverage['expected_seed_count']}",
                ]) + " |"
            )
        gates = contrast["gate_coverage"]
        lines.extend([
            "",
            f"Coverage complete: `{contrast['coverage']['complete']}`. "
            f"Non-inferiority cells: `{gates['noninferiority_pass_cells']}/{gates['cell_count']}`; "
            f"superiority cells: `{gates['superiority_pass_cells']}/{gates['cell_count']}`.",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "This is a reproducibility/statistics artifact, not evidence that the method passes the registered non-inferiority or superiority gate.",
        "Interpretation remains blocked until all confirmation phases are human-approved, complete, and reviewed.",
        "",
    ])
    return "\n".join(lines)


def _parse_phase_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--phase-root must be phase=path")
    phase, path = value.split("=", 1)
    phase = phase.strip()
    if not phase or not path.strip():
        raise ValueError("--phase-root must be phase=path")
    return phase, _resolve_path(path.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="registered confirmation plan JSON")
    parser.add_argument(
        "--phase-root", action="append", required=True,
        help="validated phase root as phase=path; repeat for all phases used by contrasts",
    )
    parser.add_argument(
        "--contrast", action="append", default=[],
        help="optional contrast label=phase:model::phase:model; defaults to plan primary_comparisons",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        roots: dict[str, Path] = {}
        for value in args.phase_root:
            phase, root = _parse_phase_root(value)
            if phase in roots:
                raise ValueError(f"duplicate --phase-root phase: {phase}")
            roots[phase] = root
        payload = build_analysis(_resolve_path(args.plan), roots, args.contrast)
        output_dir = _resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "mamba_acf_confirmation_statistics.json"
        markdown_path = output_dir / "mamba_acf_confirmation_statistics.md"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps({
            "contrasts": len(payload["contrasts"]),
            "m_cells": payload["analysis"]["simultaneous_m_cells"],
            "json": str(json_path),
            "markdown": str(markdown_path),
        }, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"analyze_mamba_acf_confirmation: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
