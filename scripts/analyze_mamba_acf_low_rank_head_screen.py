#!/usr/bin/env python3
"""Analyze the paired exploratory MambaACF low-rank-head screen.

The reader verifies phase validation, manifest/artifact hashes, the shared
train-only ACF sidecar, and the actual checkpoint state-dict layout before
computing paired descriptive contrasts.  It is intentionally independent of
the channel-gate readout and never changes training artifacts.
"""

from __future__ import annotations

import argparse
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
    RELATION_LOG_KEYS,
    _finite,
    _hash_map_for_record,
    _manifest_records,
    _read_json,
    _relative,
    _resolve,
    _sha256,
    _summary_series,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    ("low_rank_head_full_raw_h96", "full", False),
    ("low_rank_head_r16_raw_h96", "rank16", True),
)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)
EXPECTED_RANK = 16
EXPECTED_LENGTH = 96


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    """Load a checkpoint without allowing arbitrary pickle execution."""

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
    # A DataParallel checkpoint may prefix every key with ``module.``.  The
    # analyzer compares architecture keys after removing that transport prefix.
    normalized = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): tensor
        for key, tensor in state.items()
    }
    return normalized


def _tensor_shape(state: Mapping[str, Any], key: str, path: Path) -> Tuple[int, ...]:
    value = state.get(key)
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"checkpoint is missing tensor {key}: {path}")
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid tensor shape for {key}: {path}") from exc


def _tensor_numel(state: Mapping[str, Any], key: str, path: Path) -> int:
    value = state.get(key)
    numel = getattr(value, "numel", None)
    if not callable(numel):
        raise ValueError(f"checkpoint tensor has no numel for {key}: {path}")
    try:
        return int(numel())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid tensor size for {key}: {path}") from exc


def _head_contract(
    state: Mapping[str, Any], path: Path, factorized: bool
) -> Dict[str, Any]:
    """Check the learned head layout, not just the persisted config flag."""

    if factorized:
        required = {
            "temporal_head.input_projection.weight": (EXPECTED_RANK, EXPECTED_LENGTH),
            "temporal_head.output_projection.weight": (EXPECTED_LENGTH, EXPECTED_RANK),
            "linear_residual.input_projection.weight": (EXPECTED_RANK, EXPECTED_LENGTH),
            "linear_residual.output_projection.weight": (EXPECTED_LENGTH, EXPECTED_RANK),
        }
        forbidden = ("temporal_head.weight", "linear_residual.weight")
    else:
        required = {
            "temporal_head.weight": (EXPECTED_LENGTH, EXPECTED_LENGTH),
            "linear_residual.weight": (EXPECTED_LENGTH, EXPECTED_LENGTH),
        }
        forbidden = (
            "temporal_head.input_projection.weight",
            "temporal_head.output_projection.weight",
            "linear_residual.input_projection.weight",
            "linear_residual.output_projection.weight",
        )
    for key, expected_shape in required.items():
        actual_shape = _tensor_shape(state, key, path)
        if actual_shape != expected_shape:
            raise ValueError(
                f"head tensor shape mismatch for {key}: {actual_shape} != {expected_shape}: {path}"
            )
    present_forbidden = [key for key in forbidden if key in state]
    if present_forbidden:
        raise ValueError(
            f"checkpoint has incompatible head tensors {present_forbidden}: {path}"
        )
    head_keys = list(required)
    parameter_count = sum(_tensor_numel(state, key, path) for key in head_keys)
    # Include the projection biases in the reported count when present.  The
    # bias shape is intentionally checked only as a consistency aid, because
    # older checkpoints may omit it for a custom linear implementation.
    for key, shape in (
        ("temporal_head.bias", (EXPECTED_LENGTH,)),
        ("linear_residual.bias", (EXPECTED_LENGTH,)),
    ):
        if key in state:
            if _tensor_shape(state, key, path) != shape:
                raise ValueError(f"invalid head bias shape for {key}: {path}")
            parameter_count += _tensor_numel(state, key, path)
    if factorized:
        for key, shape in (
            ("temporal_head.input_projection.bias", (EXPECTED_RANK,)),
            ("temporal_head.output_projection.bias", (EXPECTED_LENGTH,)),
            ("linear_residual.input_projection.bias", (EXPECTED_RANK,)),
            ("linear_residual.output_projection.bias", (EXPECTED_LENGTH,)),
        ):
            if key in state:
                if _tensor_shape(state, key, path) != shape:
                    raise ValueError(f"invalid factorized head bias shape for {key}: {path}")
                parameter_count += _tensor_numel(state, key, path)
    return {
        "factorized": bool(factorized),
        "rank": EXPECTED_RANK if factorized else None,
        "input_length": EXPECTED_LENGTH,
        "output_length": EXPECTED_LENGTH,
        "parameter_count": int(parameter_count),
        "required_keys": head_keys,
    }


def _check_config(config: Mapping[str, Any], factorized: bool, job_id: str) -> None:
    expected: Dict[str, Any] = {
        "model": "MambaACFAutoCon",
        "features": "M",
        "pred_len": EXPECTED_LENGTH,
        "AutoCon": True,
        "AutoCon_lambda": 0.03,
        "AutoCon_acf_source": "raw",
        "mamba_variable_mixer": True,
        "mamba_variable_group_size": 1,
        "mamba_use_acf_memory": True,
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_lag_mask": True,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_min_lag": EXPECTED_LENGTH,
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_token_stride": 1,
        "mamba_scales": [1],
        "mamba_channel_gate": False,
        "mamba_acf_channel_gate": False,
        "mamba_low_rank_head": factorized,
        "mamba_temporal_rank": EXPECTED_RANK,
    }
    for key, wanted in expected.items():
        if key not in config:
            raise ValueError(f"config {key} is missing for {job_id}")
        actual = config[key]
        if isinstance(wanted, bool):
            if isinstance(actual, bool):
                pass
            elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual in (0, 1):
                actual = bool(actual)
            else:
                raise ValueError(f"config {key} is not an explicit boolean for {job_id}")
        elif isinstance(wanted, float):
            try:
                if not math.isclose(float(actual), wanted, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} mismatch for {job_id}") from exc
            continue
        if actual != wanted:
            raise ValueError(
                f"config {key} mismatch for {job_id}: {actual!r} != {wanted!r}"
            )


def _load_phase(
    plan: Mapping[str, Any], phase_name: str, arm: str, factorized: bool
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
        if not math.isclose(float(record.get("autocon_lambda")), 0.03, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"unexpected AutoCon lambda: {job_id}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected artifact: {job_id}")
        metrics_path = _resolve(str(expected_artifact))
        if (
            metrics_path.name != "metrics.npy"
            or not metrics_path.is_file()
            or not metrics_path.is_relative_to(result_root)
        ):
            raise ValueError(f"metrics artifact outside result root: {job_id}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        config = _read_json(result_dir / "config.json")
        _check_config(config, factorized, job_id)
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {job_id}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        state = _checkpoint_state(checkpoint_path)
        head = _head_contract(state, checkpoint_path, factorized)
        diagnostics = {
            key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS
        }
        sidecar_hash = hashes.get("acf_prior.npz")
        if not sidecar_hash:
            raise ValueError(f"missing sidecar hash: {job_id}")
        rows.append(
            {
                "phase": phase_name,
                "arm": arm,
                "dataset": dataset,
                "seed": seed,
                "mse": _finite(scaled.get("mse"), f"{job_id} mse"),
                "mae": _finite(scaled.get("mae"), f"{job_id} mae"),
                "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed"),
                "epochs_ran": int(record.get("epochs_ran")),
                "best_epoch": int(record.get("best_epoch")),
                "job_id": job_id,
                "sidecar_sha256": sidecar_hash,
                "head": head,
                "diagnostics": diagnostics,
                "autocon_loss_mean": _summary_series(logs, "autocon_loss", logs_path)["mean"],
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
        t_statistic, p_value = float(result.statistic), float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_statistic, p_value = mean / (sd / math.sqrt(len(numbers))), None
    else:
        t_statistic = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p_value = None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_statistic,
        "two_sided_p": p_value,
    }


def _contrast(
    metric: str,
    pairs: Sequence[Mapping[str, Any]],
    reference: str = "full",
    treatment: str = "rank16",
) -> Dict[str, Any]:
    deltas = [
        float(item[f"{treatment}_{metric}"] - item[f"{reference}_{metric}"])
        for item in pairs
    ]
    result = _paired_test(deltas)
    reference_mean = statistics.fmean(item[f"{reference}_{metric}"] for item in pairs)
    treatment_mean = statistics.fmean(item[f"{treatment}_{metric}"] for item in pairs)
    result.update(
        {
            "reference": reference,
            "treatment": treatment,
            "reference_mean": reference_mean,
            "treatment_mean": treatment_mean,
            "relative_delta_mean": _relative(result["mean"], reference_mean),
        }
    )
    return result


def _build(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {
        (str(row["arm"]), str(row["dataset"]), int(row["seed"])): row
        for row in rows
    }
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            full = indexed[("full", dataset, seed)]
            rank16 = indexed[("rank16", dataset, seed)]
            if full["sidecar_sha256"] != rank16["sidecar_sha256"]:
                raise ValueError(f"sidecar differs between arms: {dataset} seed {seed}")
            item: Dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "sidecar_sha256": full["sidecar_sha256"],
            }
            for label, row in (("full", full), ("rank16", rank16)):
                for metric in ("mse", "mae", "elapsed_sec"):
                    item[f"{label}_{metric}"] = row[metric]
                item[f"{label}_head_parameter_count"] = row["head"]["parameter_count"]
                item[f"{label}_job_id"] = row["job_id"]
            pairs.append(item)

    contrasts = {
        metric: _contrast(metric, pairs) for metric in ("mse", "mae", "elapsed_sec")
    }
    by_dataset: Dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(subset),
            **{
                metric: _contrast(metric, subset)
                for metric in ("mse", "mae", "elapsed_sec")
            },
        }

    diagnostics: Dict[str, Any] = {}
    for arm in ("full", "rank16"):
        subset = [row for row in rows if row["arm"] == arm]
        diagnostics[arm] = {
            key: {
                "mean_over_runs": statistics.fmean(
                    row["diagnostics"][key]["mean"] for row in subset
                )
            }
            for key in RELATION_LOG_KEYS
        }
        diagnostics[arm]["autocon_loss_mean_over_runs"] = statistics.fmean(
            row["autocon_loss_mean"] for row in subset
        )
        diagnostics[arm]["head_parameter_count_values"] = [
            int(row["head"]["parameter_count"]) for row in subset
        ]
        diagnostics[arm]["head_parameter_count_mean"] = statistics.fmean(
            row["head"]["parameter_count"] for row in subset
        )
        diagnostics[arm]["epochs_ran_values"] = [int(row["epochs_ran"]) for row in subset]
        diagnostics[arm]["epochs_ran_total"] = sum(diagnostics[arm]["epochs_ran_values"])
        diagnostics[arm]["epochs_ran_min"] = min(diagnostics[arm]["epochs_ran_values"])
        diagnostics[arm]["epochs_ran_max"] = max(diagnostics[arm]["epochs_ran_values"])
    return {
        "schema_version": 1,
        "status": "exploratory_acf_low_rank_head_screen_not_confirmatory",
        "plan": str(plan["_path"]),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "phases": [name for name, *_ in PHASES],
        "rows": list(rows),
        "pairs": pairs,
        "contrasts": {"rank16_minus_full": contrasts, "by_dataset": by_dataset},
        "diagnostics": diagnostics,
        "sidecar_pair_count": len(pairs),
        "head_contract": {
            "reference": "full-rank 96x96 temporal and linear-residual projections",
            "treatment": "rank-16 96x16x96 factorized temporal and linear-residual projections",
        },
        "evidence_boundary": "Paired exploratory low-rank-head evidence; not confirmatory or publication-ready.",
    }


def _markdown(result: Mapping[str, Any]) -> str:
    contrasts = result["contrasts"]["rank16_minus_full"]
    lines = [
        "# MambaACF Low-Rank Head Screen",
        "",
        "Status: **exploratory; not confirmatory**.",
        "",
        "The two arms share the MambaACF variable-token AutoCon model, raw train-only ACF sidecar, memory branch, global hard relation target, sampler, data, seeds, and eight-epoch budget. They differ only in whether both temporal forecast projections are full-rank or rank-16 factorized. This is provenance-snapshot exploratory evidence, not a current-tree retraining or standalone replay.",
        "",
        "Negative accuracy deltas favor rank-16. Negative elapsed-time deltas mean the factorized arm finished sooner. The paired sample is six dataset/seed coordinates and is descriptive only; all p-values below are unadjusted two-sided paired t-tests.",
        "",
        "## Paired Readout",
        "",
        "| Contrast (rank16 - full) | n | Mean delta | Relative delta | Wins lower | Two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, label in (("mse", "MSE"), ("mae", "MAE"), ("elapsed_sec", "Elapsed seconds")):
        values = contrasts[metric]
        p = values.get("two_sided_p")
        lines.append(
            f"| {label} | {values['n']} | {values['mean']:.6g} | {values['relative_delta_mean']:.3%} | {values['wins_lower']} | {'n/a' if p is None else f'{p:.4g}'} |"
        )
    lines += [
        "",
        "## Dataset Splits",
        "",
        "| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative | Elapsed delta | Elapsed relative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in result["contrasts"]["by_dataset"].items():
        mse, mae, elapsed = values["mse"], values["mae"], values["elapsed_sec"]
        lines.append(
            f"| {dataset} | {values['n']} | {mse['mean']:.6g} | {mse['relative_delta_mean']:.3%} | {mae['mean']:.6g} | {mae['relative_delta_mean']:.3%} | {elapsed['mean']:.6g} | {elapsed['relative_delta_mean']:.3%} |"
        )
    lines += [
        "",
        "## Head Contract",
        "",
        "The analyzer checks the learned checkpoint tensors, not only the config flag. The reference contains two 96x96 projections; the treatment contains two 96x16x96 factorized projections. AutoCon receives the same representation tensor before these forecast heads.",
        "",
        "| Arm | Mean head parameter count | Factorization |",
        "|---|---:|---|",
    ]
    for arm, label in (("full", "full"), ("rank16", "rank16")):
        values = result["diagnostics"][arm]
        lines.append(
            f"| {label} | {values['head_parameter_count_mean']:.1f} | {'yes' if arm == 'rank16' else 'no'} |"
        )
    lines += [
        "",
        "## Training Run Length",
        "",
        "Early stopping makes elapsed time partly a training-budget outcome rather than a pure forward-speed measure.",
        "",
        "| Arm | Total epochs | Epochs ran (min-max) |",
        "|---|---:|---:|",
    ]
    for arm, label in (("full", "full"), ("rank16", "rank16")):
        values = result["diagnostics"][arm]
        lines.append(f"| {label} | {values['epochs_ran_total']} | {values['epochs_ran_min']}--{values['epochs_ran_max']} |")
    lines += [
        "",
        "## Relation Diagnostics",
        "",
        "| Arm | AutoCon loss mean | Global valid fraction | Active-anchor fraction | Positive count / anchor | Sign accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (("full", "full"), ("rank16", "rank16")):
        diag = result["diagnostics"][arm]
        lines.append(
            f"| {label} | {diag['autocon_loss_mean_over_runs']:.6g} | "
            f"{diag['acf_relation_global_valid_fraction']['mean_over_runs']:.3%} | "
            f"{diag['acf_relation_global_active_anchor_fraction']['mean_over_runs']:.3%} | "
            f"{diag['acf_relation_global_positive_count_mean']['mean_over_runs']:.3f} | "
            f"{diag['acf_relation_global_sign_accuracy']['mean_over_runs']:.3%} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- This H96, three-seed screen does not establish a universal accuracy or speed advantage.",
        "- Elapsed time is a short-budget training readout and is not an end-to-end throughput benchmark with repeated synchronized resource measurements.",
        "- Non-inferiority, longer horizons, parameter-matched external baselines, and independent seeds require a separate registered experiment.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    plan_path = _resolve(args.plan)
    plan = dict(_read_json(plan_path))
    plan["_path"] = str(plan_path)
    rows: List[Mapping[str, Any]] = []
    for phase_name, arm, factorized in PHASES:
        rows.extend(_load_phase(plan, phase_name, arm, factorized))
    result = _build(plan, rows)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pairs": len(result["pairs"]),
                "output_json": str(output_json),
                "output_md": str(output_md),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
