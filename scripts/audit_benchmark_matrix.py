#!/usr/bin/env python
"""Audit benchmark plans and result coverage without selecting a model.

The repository contains several historical and exploratory plans.  This tool
keeps their coordinates separate and reports which planned jobs have a usable
metric artifact.  It is deliberately read-only with respect to source data and
existing result directories; only the requested output directory is written.

Example::

    python scripts/audit_benchmark_matrix.py \
        --plan-dir docs/research \
        --artifact-root artifacts \
        --output-dir artifacts/benchmark_audit

The resulting JSON/CSV/Markdown files are an inventory, not a statistical
comparison.  Results from different plans must not be pooled without checking
their provenance and training contracts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_training_plan import DATASETS, MODEL_SPECS  # noqa: E402
from utils.model_identity import canonical_model  # noqa: E402


# These are the minimum reference families used by the maintained extension
# plans.  They are reported separately from whatever a historical phase chose
# to include, so an omitted baseline is visible rather than silently accepted.
REQUIRED_BASELINE_MODELS = (
    "DLinear",
    "NLinear",
    "PatchTST",
    "TimesNet",
    "Mamba1",
    "Mamba2",
)


def _bool(value, default=False):
    """Parse persisted CLI booleans without treating ``"false"`` as true."""

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


def normalise_lambda(value):
    """Canonicalize a lambda value while preserving unresolved selection."""

    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "selected":
            return "selected"
        if token in {"", "none", "null"}:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(parsed):
        return str(value)
    return parsed


def _lambda_equal(left, right):
    left = normalise_lambda(left)
    right = normalise_lambda(right)
    if left == "selected":
        return True
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _phase_features(plan, phase):
    return str(phase.get("features", plan.get("global", {}).get("features", "S")))


def phase_horizons(plan, phase, dataset):
    mapping = phase.get("horizons_by_dataset")
    if mapping is not None:
        values = mapping.get(dataset, [])
    else:
        values = phase.get("horizons", plan.get("global", {}).get("horizons", []))
    return [int(value) for value in values]


def _phase_lambda_values(plan, phase, model):
    """Expand the lambda axis used by a phase, retaining ``selected``."""

    spec = MODEL_SPECS.get(model, {})
    if not _bool(spec.get("autocon", False)):
        return [None]
    # Phase values take precedence over global values, matching the runner.
    if "autocon_lambda" in phase:
        setting = phase["autocon_lambda"]
    elif "autocon_lambdas" in phase:
        setting = phase["autocon_lambdas"]
    elif "autocon_lambda" in plan.get("global", {}):
        setting = plan["global"]["autocon_lambda"]
    elif "autocon_lambdas" in plan.get("global", {}):
        setting = plan["global"]["autocon_lambdas"]
    else:
        # This is the default used by ``run_training_plan._job_specs`` for a
        # phase that forgot to state a candidate lambda explicitly.
        setting = [0.03]
    if isinstance(setting, (list, tuple)):
        values = list(setting)
    else:
        values = [setting]
    return [normalise_lambda(value) for value in values]


def expand_phase(plan, phase):
    """Expand one phase into expected coordinate records."""

    datasets = list(phase.get("datasets", []))
    models = list(phase.get("models", plan.get("global", {}).get("models", [])))
    seeds = [int(value) for value in phase.get("seeds", plan.get("global", {}).get("seeds", []))]
    features = _phase_features(plan, phase)
    rows = []
    for dataset in datasets:
        for model in models:
            for horizon in phase_horizons(plan, phase, dataset):
                for seed in seeds:
                    for lambda_value in _phase_lambda_values(plan, phase, model):
                        rows.append({
                            "dataset": str(dataset),
                            "model": canonical_model(model),
                            "features": features,
                            "pred_len": int(horizon),
                            "seed": int(seed),
                            "autocon_lambda": lambda_value,
                        })
    return rows


def _coordinate(row):
    return (
        str(row.get("dataset")),
        canonical_model(row.get("model", "unknown"), row.get("autocon")),
        str(row.get("features", "")),
        int(row.get("pred_len", 0)),
        int(row.get("seed", 0)),
        normalise_lambda(row.get("autocon_lambda")),
    )


def _base_coordinate(row):
    return _coordinate({**row, "autocon_lambda": None})[:5]


def _metric_status(result_dir):
    """Inspect compact metric artifacts without loading large arrays."""

    metrics_path = result_dir / "metrics.npy"
    stream_path = result_dir / "stream_metrics.json"
    metrics_present = False
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        try:
            values = np.asarray(np.load(metrics_path, allow_pickle=False), dtype=float).reshape(-1)
            metrics_present = values.size >= 3 and bool(np.all(np.isfinite(values[:3])))
        except (OSError, ValueError, TypeError, EOFError):
            metrics_present = False
    stream_present = False
    stream_valid = False
    if stream_path.is_file() and stream_path.stat().st_size > 0:
        stream_present = True
        try:
            payload = json.loads(stream_path.read_text(encoding="utf-8"))
            scaled = payload.get("scaled", {})
            stream_valid = isinstance(scaled, dict) and math.isfinite(float(scaled["mse"]))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            stream_valid = False
    training_complete = result_dir / "training_complete.json"
    validation_complete = False
    if training_complete.is_file():
        try:
            status = json.loads(training_complete.read_text(encoding="utf-8")).get("status")
            validation_complete = status == "trained_without_test"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            validation_complete = False
    return {
        "metrics_present": bool(metrics_present or stream_valid),
        "metrics_npy": bool(metrics_present),
        "stream_metrics": bool(stream_present),
        "stream_valid": bool(stream_valid),
        "training_complete": training_complete.is_file(),
        "validation_complete": validation_complete,
    }


def discover_results(artifact_root):
    """Read all persisted run configs below ``artifact_root``."""

    artifact_root = Path(artifact_root).resolve()
    rows = []
    if not artifact_root.is_dir():
        return rows
    for config_path in sorted(artifact_root.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            pred_len = int(config.get("pred_len"))
            seed = int(config.get("run_seed", config.get("seed")))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        row = {
            "dataset": str(config.get("data", "")),
            "model": canonical_model(config),
            "features": str(config.get("features", "")),
            "pred_len": pred_len,
            "seed": seed,
            "autocon": _bool(config.get("AutoCon", False)),
            "autocon_lambda": (
                normalise_lambda(config.get("AutoCon_lambda"))
                if _bool(config.get("AutoCon", False)) else None
            ),
            "config_path": str(config_path),
            "artifact_root": str(artifact_root),
        }
        row.update(_metric_status(config_path.parent))
        rows.append(row)
    return rows


def _dataset_inventory(hash_datasets=False):
    rows = []
    registered_paths = set()
    for name, spec in sorted(DATASETS.items()):
        path = (REPO_ROOT / spec["root"] / spec["path"]).resolve()
        registered_paths.add(path)
        row = {
            "name": name,
            "relative_path": path.relative_to(REPO_ROOT).as_posix(),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "expected_channels": int(spec["channels"]),
            "actual_channels": None,
            "header_valid": False,
            "time_column": spec["time_col"],
            "target": spec["target"],
            "frequency": spec["freq"],
        }
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    header = next(csv.reader(handle))
                row["actual_channels"] = max(0, len(header) - 1)
                row["header_valid"] = bool(
                    header and header[0] == spec["time_col"] and spec["target"] in header
                    and len(header) - 1 == int(spec["channels"])
                )
            except (OSError, StopIteration, csv.Error, UnicodeError):
                pass
            if hash_datasets:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                row["sha256"] = digest.hexdigest()
        rows.append(row)

    unknown_files = []
    dataset_root = REPO_ROOT / "dataset"
    if dataset_root.is_dir():
        for path in sorted(dataset_root.rglob("*.csv")):
            resolved = path.resolve()
            if resolved not in registered_paths:
                unknown_files.append(path.relative_to(REPO_ROOT).as_posix())
    return rows, unknown_files


def _phase_root(plan, phase):
    root = phase.get("output_root", f"artifacts/paper_final/{phase.get('name', 'unnamed')}")
    return (REPO_ROOT / root).resolve()


def _baseline_check_applicable(phase):
    """Whether a phase is expected to carry the full baseline matrix.

    Validation-only selection phases intentionally contain candidate models or
    AutoCon lambda arms rather than every forecasting baseline.  Confirmatory,
    transfer, and explicit extension phases remain subject to the baseline
    completeness check.
    """

    if "baseline_check" in phase:
        return _bool(phase.get("baseline_check"), False)
    name = str(phase.get("name", "")).lower()
    purpose = str(phase.get("purpose", "")).lower()
    return not ("selection" in name or "validation-only" in purpose)


def _is_under(path, root):
    """Return whether ``path`` is below ``root`` on Python 3.8+."""

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _expected_matches(expected, observed):
    """Match one expected coordinate against observed metric rows.

    A phase with ``autocon_lambda=selected`` intentionally leaves lambda
    unresolved until selection.  Its coordinate therefore matches any finite
    lambda for the same dataset/model/features/horizon/seed.
    """

    for row in observed:
        if _base_coordinate(row) != _base_coordinate(expected):
            continue
        expected_lambda = normalise_lambda(expected.get("autocon_lambda"))
        if expected_lambda == "selected" or _lambda_equal(expected_lambda, row.get("autocon_lambda")):
            yield row


def audit_phase(plan, phase, all_results):
    """Return a machine-readable coverage audit for one plan phase."""

    expected = expand_phase(plan, phase)
    root = _phase_root(plan, phase)
    under_root = [row for row in all_results if _is_under(row["config_path"], root)]
    observed = [row for row in under_root if row.get("metrics_present", False)]
    config_only = [row for row in under_root if not row.get("metrics_present", False)]
    validation_complete = [row for row in config_only if row.get("validation_complete", False)]
    matched = []
    missing = []
    duplicate = []
    for item in expected:
        matches = list(_expected_matches(item, observed))
        if not matches:
            missing.append(item)
        else:
            matched.append(item)
            if len(matches) > 1:
                duplicate.append({**item, "observed_count": len(matches)})

    expected_keys = {
        (row["dataset"], row["model"], row["features"], row["pred_len"], row["seed"])
        for row in expected
    }
    extra = []
    seen_extra = set()
    for row in observed:
        key = _base_coordinate(row)
        has_match = any(bool(list(_expected_matches(item, [row]))) for item in expected)
        if key not in expected_keys or not has_match:
            if _coordinate(row) not in seen_extra:
                extra.append(row)
                seen_extra.add(_coordinate(row))

    models = [canonical_model(value) for value in phase.get("models", plan.get("global", {}).get("models", []))]
    baseline_check = _baseline_check_applicable(phase)
    missing_baselines = (
        [model for model in REQUIRED_BASELINE_MODELS if model not in models]
        if baseline_check else []
    )
    coverage = float(len(matched) / len(expected)) if expected else 0.0
    return {
        "plan": str(plan.get("name", "")),
        "plan_status": str(plan.get("status", "")),
        "phase": str(phase.get("name", "")),
        "output_root": str(root.relative_to(REPO_ROOT)) if _is_under(root, REPO_ROOT) else str(root),
        "datasets": [str(value) for value in phase.get("datasets", [])],
        "models": models,
        "baseline_check_applicable": baseline_check,
        "features": _phase_features(plan, phase),
        "expected_jobs": int(len(expected)),
        "matched_jobs": int(len(matched)),
        "missing_jobs": int(len(missing)),
        "coverage": coverage,
        "observed_metric_rows": int(len(observed)),
        "config_only_rows": int(len(config_only)),
        "validation_complete_rows": int(len(validation_complete)),
        "extra_metric_rows": int(len(extra)),
        "duplicate_matches": int(len(duplicate)),
        "missing_required_baselines": missing_baselines,
        "expected": expected,
        "missing": missing,
        "duplicates": duplicate,
        "extra": [
            {key: row[key] for key in (
                "dataset", "model", "features", "pred_len", "seed", "autocon_lambda", "config_path"
            )}
            for row in extra
        ],
    }


def model_inventory():
    """Describe registered model keys and whether their source module exists."""

    module_by_model = {
        "DLinear": "models/DLinear.py",
        "NLinear": "models/NLinear.py",
        "PatchTST": "models/PatchTST.py",
        "TimesNet": "models/TimesNet.py",
        "iTransformer": "models/iTransformer.py",
        "Autoformer": "models/Autoformer.py",
        "FEDformer": "models/FEDformer.py",
        "ETSformer": "models/ETSformer.py",
        "Transformer": "models/Transformer.py",
        "Crossformer": "models/Crossformer.py",
        "AutoConNet": "models/AutoConNet.py",
        "Mamba1": "models/Mamba1AutoCon.py",
        "Mamba2": "models/Mamba2AutoCon.py",
        "MambaTS": "models/MambaTS.py",
        "MambaTSv2": "models/MambaTSv2.py",
        "MambaTSv3": "models/MambaTSv3.py",
        "MambaTSv4": "models/MambaTSv4.py",
        "MambaTSv5": "models/MambaTSv5.py",
        "MambaTM1": "models/MambaTM1AutoCon.py",
        "MambaTM2": "models/MambaTM2AutoCon.py",
        "MambaACF": "models/MambaACF.py",
        # Runner-only paired arm alias; it executes the same MambaACF module
        # with a variable-token configuration override.
        "MambaACFVariable": "models/MambaACF.py",
        "MambaACFVariableAutoCon": "models/MambaACF.py",
        # ``model_inventory`` strips the AutoCon suffix when deriving the
        # base key, so retain the matching lite base name as well.
        "MambaACFLite": "models/MambaACF.py",
        "MambaACFLiteAutoCon": "models/MambaACF.py",
        "MambaACF1": "models/MambaACF1.py",
    }
    rows = []
    for name, spec in sorted(MODEL_SPECS.items()):
        base = name[:-7] if name.endswith("AutoCon") else name
        module = module_by_model.get(base)
        rows.append({
            "name": name,
            "base_model": base,
            "autocon": bool(spec.get("autocon", False)),
            "source_module": module,
            "source_exists": bool(module and (REPO_ROOT / module).is_file()),
            "required_baseline": name in REQUIRED_BASELINE_MODELS,
        })
    return rows


def load_plans(plan_dir, explicit_paths=()):
    """Load executable plans and candidate matrices from JSON files.

    A candidate-only document may expose ``arms`` without an executable
    ``phases`` list (as the current MambaACF extension document does).  It is
    retained in the inventory with zero executable phases instead of being
    mistaken for a malformed training plan.
    """

    paths = [Path(path).resolve() for path in explicit_paths]
    if not paths:
        # Plans commonly carry a date suffix (for example,
        # ``mamba_acf_screen_plan_20260903.json``).  A few exploratory pilots
        # use ``pilot`` in the filename without the literal ``plan`` token;
        # include that explicit family while keeping unrelated JSON sidecars
        # out of the default scan.
        plan_root = Path(plan_dir).resolve()
        paths = sorted(
            set(plan_root.glob("*_plan*.json"))
            | set(plan_root.glob("*_pilot*.json"))
        )
    plans = []
    errors = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        if not isinstance(payload, dict):
            errors.append({"path": str(path), "error": "top-level JSON must be an object"})
            continue
        if not isinstance(payload.get("phases"), list):
            if isinstance(payload.get("arms"), list):
                payload = dict(payload)
                payload["phases"] = []
                payload["candidate_only"] = True
            else:
                errors.append({"path": str(path), "error": "missing phases list"})
                continue
        if not isinstance(payload.get("phases"), list):
            continue
        payload = dict(payload)
        payload["name"] = path.stem
        payload["path"] = str(path)
        plans.append(payload)
    return plans, errors


def _execution_consistency(plans, phase_audits):
    """Compare launch snapshots with the artifacts actually on disk.

    A plan is intentionally immutable after launch, so its ``execution``
    fields can describe the pre-launch snapshot even when a later launch
    sidecar produced completed artifacts.  This audit is informational: it
    surfaces that distinction for human review instead of mutating the plan
    or silently treating either source as authoritative.
    """

    by_plan = {}
    for phase in phase_audits:
        by_plan.setdefault(phase["plan"], []).append(phase)

    rows = []
    for plan in plans:
        phases = by_plan.get(plan["name"], [])
        # A metric or validation-complete row means that some training/eval
        # work produced a persisted artifact under this plan's phase root.
        actual_metric_jobs = sum(int(item.get("matched_jobs", 0)) for item in phases)
        actual_validation_jobs = sum(
            int(item.get("validation_complete_rows", 0)) for item in phases
        )
        actual_started = bool(actual_metric_jobs or actual_validation_jobs)
        execution = plan.get("execution")
        registered = plan.get("registered_job_counts")
        has_snapshot = isinstance(execution, dict)
        has_registered = isinstance(registered, dict) and "training_started" in registered
        declared_started = (
            _bool(execution.get("training_started", False))
            if has_snapshot else None
        )
        registered_started = (
            _bool(registered.get("training_started", False))
            if has_registered else None
        )
        warnings = []
        if has_snapshot and has_registered and declared_started != registered_started:
            warnings.append("execution_vs_registered_training_started_mismatch")
        if has_snapshot and actual_started != declared_started:
            warnings.append("artifacts_vs_execution_training_started_mismatch")
        if has_snapshot and actual_started and not _bool(execution.get("authorized", False)):
            warnings.append("artifacts_under_unauthorized_execution_snapshot")
        rows.append({
            "plan": plan["name"],
            "execution_snapshot_present": has_snapshot,
            "authorized_snapshot": (
                _bool(execution.get("authorized", False)) if has_snapshot else None
            ),
            "execution_training_started": declared_started,
            "registered_training_started": registered_started,
            "actual_metric_jobs": int(actual_metric_jobs),
            "actual_validation_jobs": int(actual_validation_jobs),
            "actual_artifacts_indicate_training": actual_started,
            "warnings": warnings,
        })
    return rows


def build_audit(plans, artifact_root, hash_datasets=False):
    """Build the complete inventory payload used by all output formats."""

    results = discover_results(artifact_root)
    phase_audits = []
    for plan in plans:
        for phase in plan.get("phases", []):
            phase_audits.append(audit_phase(plan, phase, results))
    datasets, unknown_files = _dataset_inventory(hash_datasets=hash_datasets)
    metric_rows = [row for row in results if row.get("metrics_present", False)]
    observed_dataset_counts = {}
    observed_model_counts = {}
    for row in metric_rows:
        observed_dataset_counts[row["dataset"]] = observed_dataset_counts.get(row["dataset"], 0) + 1
        observed_model_counts[row["model"]] = observed_model_counts.get(row["model"], 0) + 1
    # A result may be valid yet outside every registered phase (for example a
    # pilot under ``artifacts/mamba_acf_pilot_*``).  Keep a grouped count so
    # those rows cannot silently enter a paper matrix.
    planned_roots = [_phase_root(plan, phase) for plan in plans for phase in plan.get("phases", [])]
    unplanned = [row for row in metric_rows if not any(_is_under(row["config_path"], root) for root in planned_roots)]
    return {
        "schema_version": 1,
        "artifact_root": str(Path(artifact_root).resolve()),
        "plans": [
            {"name": plan["name"], "path": plan["path"], "status": plan.get("status", ""),
             "version": plan.get("version", plan.get("schema_version")),
             "candidate_only": bool(plan.get("candidate_only", False)),
             "candidate_arm_count": len(plan.get("arms", [])) if isinstance(plan.get("arms"), list) else 0}
            for plan in plans
        ],
        "datasets": datasets,
        "unknown_dataset_csv": unknown_files,
        "models": model_inventory(),
        "phases": phase_audits,
        "execution_consistency": _execution_consistency(plans, phase_audits),
        "result_config_count": len(results),
        "result_metric_count": len(metric_rows),
        "observed_metric_datasets": dict(sorted(observed_dataset_counts.items())),
        "observed_metric_models": dict(sorted(observed_model_counts.items())),
        "unplanned_metric_count": len(unplanned),
        "plan_load_errors": [],
    }


def _write_csv(payload, output_dir):
    fields = [
        "plan", "plan_status", "phase", "output_root", "features", "datasets", "models",
        "expected_jobs", "matched_jobs", "missing_jobs", "coverage", "baseline_check_applicable",
        "observed_metric_rows",
        "config_only_rows", "validation_complete_rows", "extra_metric_rows", "duplicate_matches",
        "missing_required_baselines",
    ]
    with (output_dir / "phase_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["phases"]:
            writer.writerow({
                key: (";".join(str(value) for value in row[key]) if isinstance(row[key], list) else row[key])
                for key in fields
            })
    missing_fields = ["plan", "phase", "dataset", "model", "features", "pred_len", "seed", "autocon_lambda"]
    with (output_dir / "missing_jobs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_fields)
        writer.writeheader()
        for phase in payload["phases"]:
            for row in phase["missing"]:
                writer.writerow({"plan": phase["plan"], "phase": phase["phase"], **{key: row.get(key) for key in missing_fields[2:]}})


def _markdown(payload):
    lines = [
        "# Benchmark Matrix Audit",
        "",
        "This is a read-only inventory of registered plans and persisted metric artifacts. It does not select a model and does not pool results across protocols.",
        "",
        f"Discovered `{payload['result_config_count']}` config files and `{payload['result_metric_count']}` usable metric artifacts below `{payload['artifact_root']}`.",
        f"`{payload['unplanned_metric_count']}` usable artifacts are outside registered phase roots and remain historical/exploratory.",
        "",
        "## Datasets",
        "",
        "| dataset | path | channels | header | exists |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["datasets"]:
        lines.append("| {name} | `{relative_path}` | {expected_channels} | {header_valid} | {exists} |".format(**row))
    if payload["unknown_dataset_csv"]:
        lines.extend(["", "Unregistered CSV files:", ""])
        lines.extend(f"- `{path}`" for path in payload["unknown_dataset_csv"])
    lines.extend(["", "## Model Registry", "", "| model | AutoCon | source | exists | required baseline |", "|---|---|---|---|---|"])
    for row in payload["models"]:
        lines.append("| {name} | {autocon} | `{source_module}` | {source_exists} | {required_baseline} |".format(**row))
    lines.extend(["", "## Phase Coverage", "", "| plan | phase | expected | matched | coverage | baseline check | validation-complete | config-only | missing baseline |", "|---|---|---:|---:|---:|---|---:|---:|---|"])
    for row in payload["phases"]:
        missing = ", ".join(row["missing_required_baselines"]) or "none"
        lines.append(f"| {row['plan']} | {row['phase']} | {row['expected_jobs']} | {row['matched_jobs']} | {row['coverage']:.1%} | {row['baseline_check_applicable']} | {row['validation_complete_rows']} | {row['config_only_rows']} | {missing} |")
    execution_rows = payload.get("execution_consistency", [])
    if execution_rows:
        lines.extend([
            "",
            "## Execution-state Consistency",
            "",
            "This section compares immutable launch snapshots with artifacts on disk. A warning is informational and requires sidecar/provenance review; it does not rewrite or invalidate a historical plan.",
            "",
            "| plan | authorized snapshot | execution started | registered started | metric jobs | validation jobs | artifact evidence | warnings |",
            "|---|---|---|---|---:|---:|---|---|",
        ])
        for row in execution_rows:
            warnings = ", ".join(row["warnings"]) or "none"
            lines.append(
                f"| {row['plan']} | {row['authorized_snapshot']} | "
                f"{row['execution_training_started']} | {row['registered_training_started']} | "
                f"{row['actual_metric_jobs']} | {row['actual_validation_jobs']} | "
                f"{row['actual_artifacts_indicate_training']} | {warnings} |"
            )
    lines.extend(["", "## Missing Coordinates", "", "See `missing_jobs.csv` for the complete list.", ""])
    for phase in payload["phases"]:
        if not phase["missing"]:
            continue
        lines.append(f"### {phase['plan']} / {phase['phase']}")
        lines.append("")
        for row in phase["missing"][:20]:
            lines.append("- `{dataset}` / `{model}` / H`{pred_len}` / seed `{seed}` / lambda `{autocon_lambda}`".format(**row))
        if len(phase["missing"]) > 20:
            lines.append(f"- ... {len(phase['missing']) - 20} more; see CSV")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=REPO_ROOT / "docs/research")
    parser.add_argument("--plan", type=Path, action="append", default=[],
                        help="explicit plan path; may be repeated")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-datasets", action="store_true",
                        help="include SHA256 for registered CSV files")
    args = parser.parse_args(argv)
    plans, errors = load_plans(args.plan_dir, args.plan)
    if not plans:
        raise SystemExit("no valid training plans found")
    payload = build_audit(plans, args.artifact_root, hash_datasets=args.hash_datasets)
    payload["plan_load_errors"] = errors
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    _write_csv(payload, output_dir)
    (output_dir / "benchmark_audit.md").write_text(_markdown(payload), encoding="utf-8")
    print("Audited {} plans, {} phases, {} metric artifacts; output={}".format(
        len(plans), len(payload["phases"]), payload["result_metric_count"], output_dir
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
