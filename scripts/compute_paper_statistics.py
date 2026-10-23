#!/usr/bin/env python
"""Compute seed-paired paper statistics from one confirmatory result phase.

The script reads the immutable result artifacts directly (``config.json`` and
``metrics.npy``); CSV indexes are deliberately not used as a numeric source.
For each registered backbone pair it computes the contrastive-minus-plain
delta at every dataset/feature/horizon cell, a two-sided paired Student-t
test, a Student-t confidence interval, and Holm/BH-FDR adjusted p-values.

The inferential unit is the random seed. Forecast windows are already reduced
to one scalar metric per run by the training code and are never treated as
additional replicates here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_identity import as_bool, canonical_model


MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("AutoConNet", "AutoConNetAutoCon"),
    ("Mamba1", "Mamba1AutoCon"),
    ("Mamba2", "Mamba2AutoCon"),
)
METRIC_INDEX = {"mae": 0, "mse": 1, "rmse": 2, "mape": 3, "mspe": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, str]:
    """Hash the same result/checkpoint files recorded by the phase validator."""
    names = (
        "config.json", "metrics.npy", "original_metrics.npy", "channel_metrics.npy",
        "stream_metrics.json", "stream_samples.npz",
        "pred.npy", "true.npy", "training_complete.json", "acf_prior.npz",
    )
    paths = [result_dir / name for name in names]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _command_arg(command: Sequence[str], name: str) -> str | None:
    try:
        index = list(command).index(name)
    except ValueError:
        return None
    return str(command[index + 1]) if index + 1 < len(command) else None


def _latest_manifest(path: Path) -> tuple[dict[str, dict], int]:
    records: dict[str, dict] = {}
    malformed = 0
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
                records[str(record["job_id"])] = record
    return records, malformed


def _validate_artifacts(
    results_dir: Path,
    validation_path: Path,
    plan_path: Path | None,
    phase_name: str | None,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind loaded metrics to a completed phase-validation hash snapshot."""
    with validation_path.open(encoding="utf-8") as handle:
        validation = json.load(handle)
    required = {
        "valid", "phase", "manifest", "manifest_sha256", "expected_jobs",
        "manifest_unique_jobs", "artifact_hashes_by_job",
    }
    missing = sorted(required - set(validation))
    if missing:
        raise ValueError(f"phase validation is missing strict fields: {missing}")
    if validation.get("valid") is not True:
        raise ValueError("phase validation is not valid")
    if phase_name is not None and validation.get("phase") != phase_name:
        raise ValueError(
            f"phase validation mismatch: expected {phase_name!r}, got {validation.get('phase')!r}"
        )
    if plan_path is not None and validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError("training plan changed after phase validation")

    # A plan declaring a confirmation contract can only be summarized when the
    # phase provenance contains the exact approved launch sidecar and binding.
    # This check is intentionally absent for legacy/exploratory plans so their
    # historical statistics entry point remains usable.
    launch_binding = None
    if plan_path is not None:
        try:
            with plan_path.open(encoding="utf-8") as plan_handle:
                plan_payload = json.load(plan_handle)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"cannot read training plan: {plan_path}: {exc}") from exc
        if isinstance(plan_payload, Mapping) and isinstance(
            plan_payload.get("confirmation_contract"), Mapping
        ):
            provenance_value = validation.get("provenance")
            if not isinstance(provenance_value, str) or not provenance_value.strip():
                raise ValueError("confirmation phase validation has no provenance binding")
            provenance_path = Path(provenance_value).expanduser().resolve()
            phase_root = validation_path.expanduser().resolve().parent
            if provenance_path.parent != phase_root or not provenance_path.is_file():
                raise ValueError(
                    "confirmation provenance is missing or outside the validation phase root"
                )
            if provenance_path != (phase_root / "provenance.json").resolve():
                raise ValueError(
                    "confirmation provenance is not the phase-root provenance.json"
                )
            try:
                with provenance_path.open(encoding="utf-8") as provenance_handle:
                    provenance_payload = json.load(provenance_handle)
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f"cannot read confirmation provenance: {exc}") from exc
            if provenance_payload.get("provenance_id") != validation.get("provenance_id"):
                raise ValueError(
                    "confirmation validation/provenance id mismatch"
                )
            if provenance_payload.get("phase") != validation.get("phase"):
                raise ValueError("confirmation validation/provenance phase mismatch")
            try:
                try:
                    from scripts.run_training_plan import _validate_confirmation_launch_binding
                except ModuleNotFoundError:
                    from run_training_plan import _validate_confirmation_launch_binding

                launch_binding = _validate_confirmation_launch_binding(
                    plan_payload, plan_path, provenance_payload, phase_root
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"confirmation launch binding is missing or invalid: {exc}"
                ) from exc

    manifest_path = Path(str(validation["manifest"])).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"validated manifest is missing: {manifest_path}")
    if validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("manifest changed after phase validation")
    manifest_records, malformed = _latest_manifest(manifest_path)
    if malformed:
        raise ValueError(f"validated manifest contains {malformed} malformed JSONL rows")
    expected_jobs = int(validation["expected_jobs"])
    if len(manifest_records) != expected_jobs or int(validation["manifest_unique_jobs"]) != expected_jobs:
        raise ValueError("validated manifest coverage is incomplete")
    expected_hashes = validation.get("artifact_hashes_by_job", {})
    if set(expected_hashes) != set(manifest_records):
        raise ValueError("phase-validation artifact hash coverage differs from manifest")

    result_root = results_dir / "results" if (results_dir / "results").is_dir() else results_dir
    manifest_config_paths: set[Path] = set()
    for job_id, record in manifest_records.items():
        if record.get("status") != "ok":
            raise ValueError(f"validated manifest contains non-successful job: {job_id}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"manifest job has no expected_artifact: {job_id}")
        artifact = Path(str(expected_artifact)).resolve()
        result_dir = artifact.parent
        config_path = result_dir / "config.json"
        if not config_path.is_file() or result_root not in config_path.parents:
            raise ValueError(f"validated artifact is outside results root or missing: {job_id}")
        checkpoint_root = _command_arg(record.get("command", []), "--checkpoints")
        checkpoint_dir = None
        if checkpoint_root:
            checkpoint_dir = Path(checkpoint_root).resolve() / result_dir.name
        actual_hashes = _artifact_hashes(result_dir, checkpoint_dir)
        if actual_hashes != expected_hashes[job_id]:
            raise ValueError(f"artifact changed after phase validation: {job_id}")
        manifest_config_paths.add(config_path.resolve())

    loaded_paths = {Path(record["config_path"]).resolve() for record in records}
    if loaded_paths != manifest_config_paths:
        raise ValueError(
            "loaded result configs do not exactly match the validated manifest "
            f"(loaded={len(loaded_paths)}, manifest={len(manifest_config_paths)})"
        )
    result = {
        "path": str(validation_path),
        "sha256": _sha256(validation_path),
        "phase": validation.get("phase"),
        "provenance_id": validation.get("provenance_id"),
        "manifest": str(manifest_path),
        "manifest_sha256": validation.get("manifest_sha256"),
        "expected_jobs": expected_jobs,
        "manifest_unique_jobs": int(validation["manifest_unique_jobs"]),
    }
    if launch_binding is not None:
        result["launch_authorization_binding_sha256"] = launch_binding[
            "launch_authorization_binding_sha256"
        ]
        result["launch_authorization_copy_sha256"] = launch_binding[
            "launch_authorization_copy_sha256"
        ]
    return result


def _repo_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository, like other scripts."""
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _normalise_model(config: Mapping[str, Any]) -> str:
    return canonical_model(config)


def _seed(config: Mapping[str, Any]) -> int:
    value = config.get("run_seed", config.get("seed"))
    if value is None:
        raise ValueError("config has no run_seed/seed")
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _load_plan(path: Path, phase_name: str | None, results_dir: Path) -> tuple[dict | None, dict | None, str | None]:
    if path is None:
        return None, None, None
    with path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    phases = plan.get("phases", [])
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"plan has no phases: {path}")
    phase = None
    if phase_name is not None:
        phase = next((item for item in phases if item.get("name") == phase_name), None)
        if phase is None:
            raise ValueError(f"phase not found in plan: {phase_name}")
    else:
        # A plan can be used without naming the phase only when its output root
        # identifies exactly one phase.  This avoids silently mixing S and M.
        matches = []
        for item in phases:
            output_root = item.get("output_root")
            if output_root and _repo_path(Path(output_root)) == results_dir:
                matches.append(item)
        if len(matches) != 1:
            raise ValueError("--phase is required when the plan does not identify one phase")
        phase = matches[0]
        phase_name = str(phase["name"])
    return plan, phase, phase_name


def _discover_config_paths(results_dir: Path) -> tuple[Path, list[Path]]:
    if not results_dir.is_dir():
        raise ValueError(f"results directory does not exist: {results_dir}")
    result_root = results_dir / "results" if (results_dir / "results").is_dir() else results_dir
    paths = sorted(result_root.rglob("config.json"))
    if not paths:
        raise ValueError(f"no result config.json files found below: {result_root}")
    return result_root, paths


def _load_records(results_dir: Path, metrics: Sequence[str]) -> tuple[list[dict], dict[str, int]]:
    result_root, config_paths = _discover_config_paths(results_dir)
    records: list[dict] = []
    seen: dict[tuple, Path] = {}
    metric_indices = [METRIC_INDEX[name] for name in metrics]

    for config_path in config_paths:
        result_dir = config_path.parent
        metrics_path = result_dir / "metrics.npy"
        if not metrics_path.is_file():
            raise ValueError(f"config has no metrics.npy; refusing partial/stale input: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if as_bool(config.get("skip_test", False)):
            raise ValueError(f"selection artifact passed as confirmatory input: {config_path}")
        required = ("data", "features", "seq_len", "pred_len", "model")
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"config missing {missing}: {config_path}")

        dataset = str(config["data"])
        features = str(config["features"])
        seq_len = int(config["seq_len"])
        pred_len = int(config["pred_len"])
        seed = _seed(config)
        model = _normalise_model(config)
        dimensions = tuple(int(config.get(name, 1)) for name in ("enc_in", "dec_in", "c_out"))
        coordinate = (dataset, features, seq_len, pred_len, seed, model, *dimensions)
        if coordinate in seen:
            raise ValueError(
                "duplicate result coordinate:\n  "
                f"{seen[coordinate]}\n  {config_path}"
            )
        seen[coordinate] = config_path

        try:
            values = np.asarray(np.load(metrics_path, allow_pickle=False), dtype=float).reshape(-1)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"cannot load metrics.npy: {metrics_path}: {exc}") from exc
        if values.size <= max(metric_indices) or not np.all(np.isfinite(values[metric_indices])):
            raise ValueError(f"requested metric values are missing/non-finite: {metrics_path}")

        original_path = result_dir / "original_metrics.npy"
        original_values = None
        if original_path.is_file():
            try:
                original_values = np.asarray(
                    np.load(original_path, allow_pickle=False), dtype=float
                ).reshape(-1)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"cannot load original_metrics.npy: {original_path}: {exc}") from exc
            if original_values.size <= max(metric_indices) or not np.all(
                np.isfinite(original_values[metric_indices])
            ):
                raise ValueError(f"requested original metric values are invalid: {original_path}")

        records.append({
            "dataset": dataset,
            "features": features,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "seed": seed,
            "model": model,
            "raw_model": str(config.get("model")),
            "autocon": as_bool(config.get("AutoCon", False)),
            "autocon_lambda": (
                _finite_float(config["AutoCon_lambda"], "AutoCon_lambda")
                if as_bool(config.get("AutoCon", False)) and config.get("AutoCon_lambda") is not None
                else None
            ),
            "enc_in": dimensions[0],
            "dec_in": dimensions[1],
            "c_out": dimensions[2],
            "values": values,
            "original_values": original_values,
            "config_path": config_path,
            "metrics_path": metrics_path,
            "config_sha256": _sha256(config_path),
            "metrics_sha256": _sha256(metrics_path),
            "result_root": result_root,
        })

    if not records:
        raise ValueError("no usable result records found")
    return records, {"config_files": len(config_paths), "records": len(records)}


def _plan_expected(plan: dict | None, phase: dict | None) -> tuple[set[tuple], dict]:
    if plan is None or phase is None:
        return set(), {}
    global_cfg = dict(plan.get("global", {}))
    features = str(phase.get("features", global_cfg.get("features", "S")))
    seq_len = int(phase.get("seq_len", global_cfg.get("seq_len", 96)))
    datasets = [str(item) for item in phase.get("datasets", [])]
    models = [str(item) for item in phase.get("models", [])]
    seeds = [int(item) for item in phase.get("seeds", [])]
    horizons_map = phase.get("horizons_by_dataset")
    if horizons_map is None:
        horizons = [int(item) for item in phase.get("horizons", global_cfg.get("horizons", []))]
        horizons_map = {dataset: horizons for dataset in datasets}
    expected = set()
    for dataset in datasets:
        for model in models:
            for pred_len in horizons_map.get(dataset, []):
                for seed in seeds:
                    expected.add((dataset, features, seq_len, int(pred_len), seed, model))
    if not expected:
        raise ValueError("plan phase expands to no expected result coordinates")
    metadata = {
        "phase": str(phase.get("name")),
        "features": features,
        "seq_len": seq_len,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "horizons_by_dataset": {str(k): [int(v) for v in values] for k, values in horizons_map.items()},
    }
    return expected, metadata


def _projection(record: Mapping[str, Any]) -> tuple:
    return (
        record["dataset"], record["features"], int(record["seq_len"]),
        int(record["pred_len"]), int(record["seed"]), record["model"],
    )


def _check_coverage(
    records: Sequence[Mapping[str, Any]],
    expected: set[tuple],
    metadata: Mapping[str, Any],
    allow_incomplete: bool,
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    observed = {_projection(record) for record in records}
    if len(observed) != len(records):
        raise ValueError("duplicate dataset/features/seq_len/horizon/seed/model coordinates")
    missing = sorted(expected - observed)
    extra = sorted(observed - expected) if expected else []
    if expected and (missing or extra):
        message = f"plan coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        if not allow_incomplete:
            examples = []
            if missing:
                examples.append("missing=" + repr(missing[:3]))
            if extra:
                examples.append("extra=" + repr(extra[:3]))
            raise ValueError(message + "; " + "; ".join(examples))
        warnings.append(message + " (continuing because --allow-incomplete was set)")

    observed_models = sorted({record["model"] for record in records})
    ignored_models = sorted(set(observed_models) - set(metadata.get("models", observed_models)))
    coverage = {
        "expected_coordinates": len(expected) if expected else None,
        "observed_coordinates": len(observed),
        "missing_coordinates": len(missing) if expected else None,
        "extra_coordinates": len(extra) if expected else None,
        "observed_models": observed_models,
        "ignored_models": ignored_models,
        "expected_models": metadata.get("models"),
        "missing_examples": [list(item) for item in missing[:10]],
        "extra_examples": [list(item) for item in extra[:10]],
    }
    return coverage, warnings


def _paired_test(delta: np.ndarray, alpha: float) -> dict[str, Any]:
    n = int(delta.size)
    mean = float(np.mean(delta)) if n else float("nan")
    std = float(np.std(delta, ddof=1)) if n > 1 else float("nan")
    if n < 2:
        return {
            "n_pairs": n,
            "mean_delta": mean,
            "std_delta": None if n < 2 else std,
            "sem_delta": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
        }
    sem = std / math.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    half_width = tcrit * sem
    if std == 0.0:
        # The paired t statistic is undefined at zero variance.  A constant
        # non-zero delta is decisive; an exact tie has no evidence either way.
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        t_stat = mean / sem
        p_value = float(2.0 * stats.t.sf(abs(t_stat), n - 1))
    return {
        "n_pairs": n,
        "mean_delta": mean,
        "std_delta": std,
        "sem_delta": sem,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "p_value": p_value,
    }


def _holm_bonferroni(p_values: Sequence[float | None]) -> list[float | None]:
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    adjusted: list[float | None] = [None] * len(p_values)
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda item: (item[1], item[0]))
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def _benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    adjusted: list[float | None] = [None] * len(p_values)
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda item: (item[1], item[0]))
    total = len(ordered)
    running = 1.0
    for rank in range(total - 1, -1, -1):
        index, value = ordered[rank]
        running = min(running, value * total / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def _relative_change(delta_mean: float, plain_mean: float) -> float | None:
    if not math.isfinite(plain_mean) or plain_mean == 0.0:
        return None
    return float(100.0 * delta_mean / plain_mean)


def _make_comparisons(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    alpha: float,
    allow_incomplete: bool,
) -> tuple[list[dict], list[str]]:
    # Build a model/seed index for each cell, retaining dimensions in the
    # record so a mismatched S/M pair cannot be paired accidentally.
    by_cell_model: dict[tuple, dict[str, dict[int, Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        cell = (record["dataset"], record["features"], record["seq_len"], record["pred_len"])
        model_map = by_cell_model[cell][record["model"]]
        seed = int(record["seed"])
        if seed in model_map:
            raise ValueError(f"duplicate seed in cell/model: {cell}, {record['model']}, seed={seed}")
        model_map[seed] = record

    warnings: list[str] = []
    rows: list[dict] = []
    for cell in sorted(by_cell_model):
        dataset, features, seq_len, pred_len = cell
        model_map = by_cell_model[cell]
        for plain_model, auto_model in MODEL_PAIRS:
            plain = model_map.get(plain_model, {})
            auto = model_map.get(auto_model, {})
            plain_seeds, auto_seeds = set(plain), set(auto)
            common = sorted(plain_seeds & auto_seeds)
            missing_plain = sorted(auto_seeds - plain_seeds)
            missing_auto = sorted(plain_seeds - auto_seeds)
            if missing_plain or missing_auto:
                message = (
                    f"unpaired seeds for {dataset}/{features}/h{pred_len} "
                    f"{auto_model} vs {plain_model}: "
                    f"missing_plain={missing_plain}, missing_auto={missing_auto}"
                )
                if not allow_incomplete:
                    raise ValueError(message)
                warnings.append(message + " (using intersection)")
            if not common:
                if allow_incomplete:
                    warnings.append(f"no paired seeds for {dataset}/{features}/h{pred_len} {auto_model} vs {plain_model}")
                    continue
                raise ValueError(f"no paired seeds for {dataset}/{features}/h{pred_len} {auto_model} vs {plain_model}")

            lambda_values = {
                auto[seed].get("autocon_lambda") for seed in common
                if auto[seed].get("autocon_lambda") is not None
            }
            if len(lambda_values) > 1:
                raise ValueError(
                    f"contrastive lambda varies within cell {dataset}/{features}/h{pred_len} "
                    f"{auto_model}: {sorted(lambda_values)}"
                )
            lambda_value = next(iter(lambda_values)) if lambda_values else None

            for metric in metrics:
                index = METRIC_INDEX[metric]
                plain_values = np.asarray([plain[seed]["values"][index] for seed in common], dtype=float)
                auto_values = np.asarray([auto[seed]["values"][index] for seed in common], dtype=float)
                delta = auto_values - plain_values
                test = _paired_test(delta, alpha)
                plain_mean = float(np.mean(plain_values))
                auto_mean = float(np.mean(auto_values))
                row = {
                    "metric": metric,
                    "scale": "train-standardized",
                    "dataset": dataset,
                    "features": features,
                    "seq_len": int(seq_len),
                    "pred_len": int(pred_len),
                    "plain_model": plain_model,
                    "autocon_model": auto_model,
                    "comparison": f"{auto_model} - {plain_model}",
                    "autocon_lambda": lambda_value,
                    "n_pairs": test["n_pairs"],
                    "mean_plain": plain_mean,
                    "mean_autocon": auto_mean,
                    "mean_delta": test["mean_delta"],
                    "std_delta": test["std_delta"],
                    "sem_delta": test["sem_delta"],
                    "ci95_low": test["ci_low"],
                    "ci95_high": test["ci_high"],
                    "relative_change_pct": _relative_change(test["mean_delta"], plain_mean),
                    "p_value_two_sided": test["p_value"],
                    "direction": (
                        "improvement" if test["mean_delta"] < 0 else
                        "degradation" if test["mean_delta"] > 0 else "tie"
                    ),
                    "seed_deltas": [
                        {
                            "seed": int(seed),
                            "plain": float(plain[seed]["values"][index]),
                            "autocon": float(auto[seed]["values"][index]),
                            "delta": float(auto[seed]["values"][index] - plain[seed]["values"][index]),
                            "plain_config": os.path.relpath(plain[seed]["config_path"], records[0]["result_root"]),
                            "autocon_config": os.path.relpath(auto[seed]["config_path"], records[0]["result_root"]),
                            "plain_config_sha256": plain[seed]["config_sha256"],
                            "plain_metrics_sha256": plain[seed]["metrics_sha256"],
                            "autocon_config_sha256": auto[seed]["config_sha256"],
                            "autocon_metrics_sha256": auto[seed]["metrics_sha256"],
                        }
                        for seed in common
                    ],
                }
                rows.append(row)

    rows.sort(key=lambda row: (
        row["metric"], row["dataset"], row["features"], row["pred_len"],
        row["plain_model"], row["autocon_model"],
    ))
    for metric in metrics:
        indices = [index for index, row in enumerate(rows) if row["metric"] == metric]
        p_values = [rows[index]["p_value_two_sided"] for index in indices]
        holm = _holm_bonferroni(p_values)
        fdr = _benjamini_hochberg(p_values)
        family_size = sum(value is not None and math.isfinite(float(value)) for value in p_values)
        for local_index, row_index in enumerate(indices):
            row = rows[row_index]
            row["correction_family"] = f"{metric}:all_dataset_horizon_pairs"
            row["correction_family_size"] = int(family_size)
            row["holm_p_value"] = holm[local_index]
            row["fdr_bh_p_value"] = fdr[local_index]
            row["holm_significant"] = bool(holm[local_index] is not None and holm[local_index] <= alpha)
            row["fdr_significant"] = bool(fdr[local_index] is not None and fdr[local_index] <= alpha)
    return rows, warnings


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["metric"], row["features"], row["plain_model"], row["autocon_model"])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        metric, features, plain_model, auto_model = key
        deltas = np.asarray([row["mean_delta"] for row in group], dtype=float)
        relative = np.asarray(
            [row["relative_change_pct"] for row in group if row["relative_change_pct"] is not None],
            dtype=float,
        )
        output.append({
            "metric": metric,
            "features": features,
            "plain_model": plain_model,
            "autocon_model": auto_model,
            "comparison": f"{auto_model} - {plain_model}",
            "n_cells": int(len(group)),
            "n_improved_cells": int(np.sum(deltas < 0)),
            "n_degraded_cells": int(np.sum(deltas > 0)),
            "mean_cell_delta": float(np.mean(deltas)),
            "median_cell_delta": float(np.median(deltas)),
            "mean_relative_change_pct": float(np.mean(relative)) if relative.size else None,
            "holm_significant_cells": int(sum(bool(row["holm_significant"]) for row in group)),
            "fdr_significant_cells": int(sum(bool(row["fdr_significant"]) for row in group)),
        })
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def _table(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> str:
    rows = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _markdown(payload: Mapping[str, Any]) -> str:
    analysis = payload["analysis"]
    coverage = payload["coverage"]
    rows = payload["comparisons"]
    aggregate = payload["aggregate_by_comparison"]
    lines = [
        "# Corrected Confirmatory Statistics",
        "",
        "This report reads `config.json` and `metrics.npy` directly from one result phase.",
        "The paired unit is the random seed; forecast windows are not treated as independent replicates.",
        "A negative delta means the AutoCon model has lower error than its plain backbone.",
        "",
        "## Analysis",
        "",
        f"- Phase: `{analysis.get('phase') or 'unspecified'}`",
        f"- Results: `{analysis['results_dir']}`",
        f"- Metrics: `{', '.join(analysis['metrics'])}` in `{analysis['scale']}` space",
        f"- Two-sided alpha: `{_format(analysis['alpha'])}`",
        "- Holm and Benjamini-Hochberg FDR are corrected separately for each metric over all dataset/horizon/backbone cells.",
        "",
        "## Coverage",
        "",
        _table([coverage], [
            "expected_coordinates", "observed_coordinates", "missing_coordinates",
            "extra_coordinates", "expected_models", "observed_models",
        ]),
        "",
        "## Cell-Level Paired Results",
        "",
    ]
    if rows:
        lines.append(_table(rows, [
            "metric", "dataset", "features", "pred_len", "comparison", "autocon_lambda",
            "n_pairs", "mean_delta", "relative_change_pct", "ci95_low", "ci95_high",
            "p_value_two_sided", "holm_p_value", "fdr_bh_p_value", "direction",
        ]))
    else:
        lines.append("No complete paired cells were available.")
    lines.extend(["", "## Descriptive Backbone Aggregates", ""])
    if aggregate:
        lines.append(_table(aggregate, [
            "metric", "features", "comparison", "n_cells", "n_improved_cells",
            "n_degraded_cells", "mean_cell_delta", "median_cell_delta",
            "mean_relative_change_pct", "holm_significant_cells", "fdr_significant_cells",
        ]))
    else:
        lines.append("No aggregate rows were available.")
    warnings = payload.get("warnings", [])
    lines.extend(["", "## Interpretation Notes", "",
                  "- Confidence intervals are Student-t intervals for paired seed deltas; they are not intervals over sliding windows.",
                  "- P-values are two-sided. A corrected p-value below alpha does not by itself establish a universal improvement; inspect the sign and cell coverage.",
                  "- Backbone aggregate rows are descriptive summaries across cells and are not additional independent tests.",
                  "- `seed_deltas` and artifact hashes are retained in the JSON for auditability."])
    if warnings:
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in warnings])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="One confirmatory phase root (for example corrected_confirmatory_S).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory receiving paper_statistics.json and paper_statistics.md.")
    parser.add_argument("--plan", type=Path, default=None,
                        help="Optional training plan used for exact coordinate coverage checks.")
    parser.add_argument("--phase", default=None,
                        help="Plan phase name; required when --plan cannot identify one phase.")
    parser.add_argument("--validation", type=Path, default=None,
                        help="phase_validation.json; defaults to <results_dir>/phase_validation.json.")
    parser.add_argument("--metric", dest="metrics", action="append", choices=sorted(METRIC_INDEX),
                        help="Metric to analyze; repeat for more metrics (default: mse).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Two-sided confidence/test alpha (default: 0.05).")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Summarize available intersections instead of failing on missing coordinates/seeds.")
    parser.add_argument("--allow-unvalidated", action="store_true",
                        help="Permit analysis without a valid phase_validation.json (exploratory use only).")
    args = parser.parse_args()

    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between 0 and 1")
    metrics = list(dict.fromkeys(args.metrics or ["mse"]))
    results_dir = _repo_path(args.results_dir)
    plan_path = _repo_path(args.plan) if args.plan is not None else None
    if plan_path is None and (args.allow_unvalidated or args.allow_incomplete):
        raise ValueError(
            "--plan is required with --allow-unvalidated or --allow-incomplete; "
            "unbound summary mode is not permitted"
        )
    plan, phase, phase_name = _load_plan(plan_path, args.phase, results_dir) if plan_path else (None, None, args.phase)
    strict_confirmation = isinstance(plan, Mapping) and isinstance(
        plan.get("confirmation_contract"), Mapping
    )
    validation_path = _repo_path(args.validation) if args.validation is not None else results_dir / "phase_validation.json"
    if strict_confirmation:
        if args.allow_unvalidated:
            raise ValueError(
                "--allow-unvalidated is forbidden for a confirmation contract"
            )
        if args.allow_incomplete:
            raise ValueError(
                "--allow-incomplete is forbidden for a confirmation contract"
            )
        if not validation_path.is_file():
            raise ValueError(
                f"valid phase_validation.json is required for confirmation statistics: {validation_path}"
            )
    records, load_counts = _load_records(results_dir, metrics)
    expected, plan_metadata = _plan_expected(plan, phase)
    coverage, coverage_warnings = _check_coverage(records, expected, plan_metadata, args.allow_incomplete)
    comparisons, pairing_warnings = _make_comparisons(records, metrics, args.alpha, args.allow_incomplete)
    validation_info = None
    validation_warnings: list[str] = []
    if validation_path.is_file():
        validation_info = _validate_artifacts(results_dir, validation_path, plan_path, phase_name, records)
    elif not args.allow_unvalidated:
        raise ValueError(
            f"valid phase_validation.json is required for paper statistics: {validation_path}; "
            "use --allow-unvalidated only for exploratory checks"
        )
    else:
        validation_warnings.append(f"no phase validation was used: {validation_path}")
    warnings = coverage_warnings + pairing_warnings + validation_warnings
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "phase": phase_name,
            "results_dir": str(results_dir),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "plan": str(plan_path) if plan_path else None,
            "plan_sha256": _sha256(plan_path) if plan_path else None,
            "metrics": metrics,
            "scale": "train-standardized",
            "alpha": float(args.alpha),
            "test": "two-sided paired Student-t on seed-level deltas",
            "correction": "Holm-Bonferroni and Benjamini-Hochberg FDR per metric family",
            "allow_incomplete": bool(args.allow_incomplete),
            "allow_unvalidated": bool(args.allow_unvalidated),
            "argv": sys.argv,
        },
        "validation": validation_info,
        "load_counts": load_counts,
        "coverage": coverage,
        "warnings": warnings,
        "comparisons": comparisons,
        "aggregate_by_comparison": _aggregate(comparisons),
    }
    payload = _json_safe(payload)
    json_path = output_dir / "paper_statistics.json"
    markdown_path = output_dir / "paper_statistics.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "records": load_counts["records"],
        "paired_cells": len(comparisons),
        "metrics": metrics,
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
