#!/usr/bin/env python3
"""Audit the exploratory MambaACF relation-coverage screen.

The reader is deliberately read-only.  It verifies phase validation, manifest
and artifact hashes, train-only ACF sidecars, exact three-arm configuration,
and relation objective v3 before reporting paired forecasting metrics and the
realized per-lag sampler exposure saved in ``loss_logs.json``.
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
except Exception:  # pragma: no cover
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


PHASES = (
    ("relation_coverage_reference_random_h96", "reference", "random", False),
    ("relation_coverage_balanced_h96", "balanced", "balanced", False),
    ("relation_coverage_active_norm_h96", "active_norm", "random", True),
)
DATASETS = ("ETTh1", "traffic")
SEEDS = (2021, 2022, 2023)


def _pair_count(value: Any, label: str) -> int:
    number = _finite(value, label)
    integer = int(number)
    if integer < 0 or not math.isclose(number, integer, abs_tol=1e-9):
        raise ValueError(f"{label} must be a non-negative integer: {value!r}")
    return integer


def _positive_lag_set(value: Any, label: str, path: Path) -> Tuple[int, ...]:
    """Normalize a persisted lag list and reject ambiguous/invalid values."""

    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list: {path}")
    normalized = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} contains a non-positive lag: {path}")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-finite lag: {path}") from exc
        if not math.isfinite(number) or number <= 0 or not number.is_integer():
            raise ValueError(f"{label} contains a non-positive lag: {path}")
        normalized.append(int(number))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must contain unique positive lags: {path}")
    return tuple(sorted(normalized))


def _lag_exposure_summary(
    logs: Mapping[str, Any], path: Path, expected_lags: Sequence[int] | None = None
) -> Dict[str, Any]:
    """Validate epoch exposure telemetry and aggregate it by selected lag."""

    count_epochs = logs.get("acf_relation_lag_pair_counts")
    exposure_epochs = logs.get("acf_relation_lag_pair_exposure")
    if not isinstance(count_epochs, list) or not count_epochs:
        raise ValueError(f"missing lag-pair counts: {path}")
    if not isinstance(exposure_epochs, list) or len(exposure_epochs) != len(count_epochs):
        raise ValueError(f"missing or misaligned lag-pair exposure: {path}")
    expected_lag_set = None if expected_lags is None else {str(int(lag)) for lag in expected_lags}
    aggregate: Dict[str, int] = {}
    epoch_spans: List[float] = []
    for epoch, (raw_counts, raw_exposure) in enumerate(
        zip(count_epochs, exposure_epochs), start=1
    ):
        if not isinstance(raw_counts, Mapping) or not isinstance(raw_exposure, Mapping):
            raise ValueError(f"lag-pair telemetry epoch {epoch} is not an object: {path}")
        keys = {str(key) for key in raw_counts}
        if not keys or keys != {str(key) for key in raw_exposure}:
            raise ValueError(f"lag-pair telemetry has inconsistent lag keys at epoch {epoch}: {path}")
        if expected_lag_set is not None and keys != expected_lag_set:
            raise ValueError(
                f"lag-pair telemetry lag keys do not match effective lags at epoch {epoch}: {path}"
            )
        counts: Dict[str, int] = {}
        for lag in sorted(keys, key=int):
            if int(lag) <= 0:
                raise ValueError(f"lag-pair telemetry has non-positive lag {lag!r}: {path}")
            counts[lag] = _pair_count(raw_counts[lag], f"{path.name}:lag={lag}")
        total = sum(counts.values())
        if total <= 0:
            raise ValueError(f"lag-pair telemetry has no accepted pair at epoch {epoch}: {path}")
        shares = []
        for lag, count in counts.items():
            observed = _finite(raw_exposure[lag], f"{path.name}:exposure lag={lag}")
            expected = count / total
            if observed < 0 or observed > 1 or not math.isclose(
                observed, expected, rel_tol=1e-7, abs_tol=1e-9
            ):
                raise ValueError(f"lag-pair exposure does not match counts at epoch {epoch}: {path}")
            shares.append(observed)
            aggregate[lag] = aggregate.get(lag, 0) + count
        epoch_spans.append(max(shares) - min(shares))
    total_pairs = sum(aggregate.values())
    per_lag = {
        lag: {
            "pair_count": count,
            "aggregate_exposure": count / total_pairs,
        }
        for lag, count in sorted(aggregate.items(), key=lambda item: int(item[0]))
    }
    aggregate_shares = [item["aggregate_exposure"] for item in per_lag.values()]
    return {
        "epochs": len(count_epochs),
        "total_pairs": total_pairs,
        "per_lag": per_lag,
        "aggregate_balance_span": max(aggregate_shares) - min(aggregate_shares),
        "mean_epoch_balance_span": statistics.fmean(epoch_spans),
        "max_epoch_balance_span": max(epoch_spans),
    }


def _expected_config(balance: str, active_norm: bool) -> Dict[str, Any]:
    return {
        "model": "MambaACFAutoCon",
        "features": "M",
        "pred_len": 96,
        "AutoCon": True,
        "AutoCon_lambda": 0.03,
        "AutoCon_acf_source": "raw",
        "mamba_variable_mixer": True,
        "mamba_variable_group_size": 1,
        "mamba_use_acf_memory": True,
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_batching": "lag_pairs",
        "mamba_acf_relation_lag_balance": balance,
        "mamba_acf_relation_active_anchor_normalization": active_norm,
        "mamba_acf_relation_lag_mask": True,
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_low_rank_head": False,
    }


def _assert_config(config: Mapping[str, Any], expected: Mapping[str, Any], job_id: str) -> None:
    for key, value in expected.items():
        if key not in config:
            raise ValueError(f"config {key} is missing for {job_id}")
        actual = config[key]
        if isinstance(value, bool):
            if isinstance(actual, bool):
                pass
            elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual in (0, 1):
                actual = bool(actual)
            else:
                raise ValueError(f"config {key} is not an explicit boolean for {job_id}")
        elif isinstance(value, float):
            try:
                actual = float(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} is not numeric: {job_id}") from exc
            if math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12):
                continue
        if actual != value:
            raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {value!r}")


def _load_phase(
    plan: Mapping[str, Any], phase_name: str, arm: str, balance: str, active_norm: bool
) -> List[Dict[str, Any]]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    phase = next((item for item in phases if item.get("name") == phase_name), None)
    if not isinstance(phase, Mapping):
        raise ValueError(f"plan has no phase {phase_name!r}")
    phase_root = _resolve(str(phase.get("output_root", "")))
    validation_path = phase_root / "phase_validation.json"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True or validation.get("phase") != phase_name:
        raise ValueError(f"invalid phase validation: {validation_path}")
    records = _manifest_records(phase_root, validation)
    result_root, checkpoint_root = phase_root / "results", phase_root / "checkpoints"
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for record in records:
        job_id = str(record.get("job_id"))
        if record.get("status") != "ok" or record.get("phase") != phase_name:
            raise ValueError(f"unsuccessful manifest record: {job_id}")
        dataset, seed = str(record.get("dataset")), int(record.get("seed"))
        coordinate = (dataset, seed)
        if dataset not in DATASETS or seed not in SEEDS or coordinate in seen:
            raise ValueError(f"unexpected or duplicate coordinate: {phase_name} {coordinate}")
        seen.add(coordinate)
        if not math.isclose(_finite(record.get("autocon_lambda"), f"{job_id} lambda"), 0.03, abs_tol=1e-12):
            raise ValueError(f"unexpected AutoCon lambda: {job_id}")
        metrics_path = _resolve(str(record.get("expected_artifact", "")))
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not metrics_path.is_relative_to(result_root):
            raise ValueError(f"metrics artifact outside result root: {job_id}")
        result_dir, checkpoint_dir = metrics_path.parent, checkpoint_root / metrics_path.parent.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        sidecar_hash = hashes.get("acf_prior.npz")
        if not sidecar_hash:
            raise ValueError(f"missing ACF sidecar hash: {job_id}")
        config = _read_json(result_dir / "config.json")
        _assert_config(config, _expected_config(balance, active_norm), job_id)
        if str(config.get("mamba_acf_prior_sha256", "")).lower() != sidecar_hash:
            raise ValueError(f"config ACF sidecar hash mismatch: {job_id}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        if str(logs.get("acf_relation_lag_balance", "")).lower() != balance:
            raise ValueError(f"loss-log lag balance mismatch: {logs_path}")
        effective_lags = _positive_lag_set(
            logs.get("acf_relation_effective_lags"),
            "acf_relation_effective_lags",
            logs_path,
        )
        selected_lags = _positive_lag_set(
            logs.get("acf_relation_selected_lags"),
            "acf_relation_selected_lags",
            logs_path,
        )
        if selected_lags != effective_lags:
            raise ValueError(
                f"selected and effective relation lags differ for {job_id}: "
                f"{selected_lags} != {effective_lags}"
            )
        lag_exposure = _lag_exposure_summary(logs, logs_path, effective_lags)
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics lacks scaled metrics: {job_id}")
        rows.append({
            "phase": phase_name,
            "arm": arm,
            "dataset": dataset,
            "seed": seed,
            "job_id": job_id,
            "mse": _finite(scaled.get("mse"), f"{job_id} mse"),
            "mae": _finite(scaled.get("mae"), f"{job_id} mae"),
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed"),
            "epochs_ran": int(record.get("epochs_ran")),
            "best_epoch": int(record.get("best_epoch")),
            "sidecar_sha256": sidecar_hash,
            "diagnostics": {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS},
            "autocon_loss_mean": _summary_series(logs, "autocon_loss", logs_path)["mean"],
            "effective_lags": list(effective_lags),
            "selected_lags": list(selected_lags),
            "lag_exposure": lag_exposure,
        })
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if seen != expected:
        raise ValueError(f"incomplete coordinate matrix for {phase_name}")
    return rows


def _paired(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    if scipy_stats is not None and len(values) > 1:
        test = scipy_stats.ttest_1samp(values, 0.0)
        t_statistic, p_value = float(test.statistic), float(test.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_statistic, p_value = mean / (sd / math.sqrt(len(values))), None
    else:
        t_statistic, p_value = (0.0 if mean == 0 else math.copysign(math.inf, mean)), None
    return {"n": len(values), "mean": mean, "sd": sd, "wins_lower": sum(value < 0 for value in values), "ties": sum(value == 0 for value in values), "t_statistic": t_statistic, "two_sided_p": p_value}


def _contrast(reference: str, treatment: str, metric: str, pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    deltas = [float(item[f"{treatment}_{metric}"] - item[f"{reference}_{metric}"]) for item in pairs]
    result = _paired(deltas)
    reference_mean = statistics.fmean(float(item[f"{reference}_{metric}"]) for item in pairs)
    result.update({
        "reference": reference,
        "treatment": treatment,
        "reference_mean": reference_mean,
        "treatment_mean": statistics.fmean(float(item[f"{treatment}_{metric}"]) for item in pairs),
        "relative_delta_mean": _relative(result["mean"], reference_mean),
    })
    return result


def _build(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {(row["arm"], row["dataset"], row["seed"]): row for row in rows}
    pairs: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            reference, balanced, active_norm = (
                indexed[(arm, dataset, seed)] for arm in ("reference", "balanced", "active_norm")
            )
            if len({reference["sidecar_sha256"], balanced["sidecar_sha256"], active_norm["sidecar_sha256"]}) != 1:
                raise ValueError(f"sidecar mismatch across arms: {dataset} seed {seed}")
            pair: Dict[str, Any] = {"dataset": dataset, "seed": seed, "sidecar_sha256": reference["sidecar_sha256"]}
            for arm, row in (("reference", reference), ("balanced", balanced), ("active_norm", active_norm)):
                for metric in ("mse", "mae", "elapsed_sec"):
                    pair[f"{arm}_{metric}"] = row[metric]
                pair[f"{arm}_job_id"] = row["job_id"]
            pairs.append(pair)
    contrasts: Dict[str, Any] = {}
    for name, treatment in (("balanced_minus_reference", "balanced"), ("active_norm_minus_reference", "active_norm")):
        contrasts[name] = {metric: _contrast("reference", treatment, metric, pairs) for metric in ("mse", "mae", "elapsed_sec")}
        contrasts[name]["by_dataset"] = {
            dataset: {metric: _contrast("reference", treatment, metric, [pair for pair in pairs if pair["dataset"] == dataset]) for metric in ("mse", "mae", "elapsed_sec")}
            for dataset in DATASETS
        }
    exposure_rows = []
    for row in rows:
        summary = row["lag_exposure"]
        for lag, values in summary["per_lag"].items():
            exposure_rows.append({
                "arm": row["arm"], "dataset": row["dataset"], "seed": row["seed"], "lag": int(lag),
                "pair_count": values["pair_count"], "aggregate_exposure": values["aggregate_exposure"],
                "aggregate_balance_span": summary["aggregate_balance_span"],
                "mean_epoch_balance_span": summary["mean_epoch_balance_span"],
            })
    diagnostics = {
        arm: {
            "autocon_loss_mean_over_runs": statistics.fmean(row["autocon_loss_mean"] for row in rows if row["arm"] == arm),
            "active_anchor_fraction_mean_over_runs": statistics.fmean(row["diagnostics"]["acf_relation_global_active_anchor_fraction"]["mean"] for row in rows if row["arm"] == arm),
            "epochs_ran_values": [int(row["epochs_ran"]) for row in rows if row["arm"] == arm],
        }
        for arm in ("reference", "balanced", "active_norm")
    }
    for values in diagnostics.values():
        values["epochs_ran_total"] = sum(values["epochs_ran_values"])
        values["epochs_ran_min"] = min(values["epochs_ran_values"])
        values["epochs_ran_max"] = max(values["epochs_ran_values"])
    return {
        "schema_version": 1,
        "status": "exploratory_relation_coverage_screen_not_confirmatory",
        "plan": str(plan["_path"]),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "phases": [name for name, *_ in PHASES],
        "rows": list(rows), "pairs": pairs, "contrasts": contrasts,
        "exposure_rows": exposure_rows, "diagnostics": diagnostics,
        "sidecar_triplet_count": len(pairs),
        "evidence_boundary": "Paired exploratory relation-coverage evidence only; no confirmatory, universal accuracy, or speed claim.",
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MambaACF Relation-Coverage Screen", "",
        "Status: **exploratory; not confirmatory**.", "",
        "All arms use raw train-only ACF sidecars, the variable-token MambaACF AutoCon alias, memory, global hard-selected relation supervision, full-rank heads, the same H96 matrix, and lambda 0.03. They differ only in realized lag-pair policy or global active-anchor normalization. The readout is provenance-snapshot exploratory evidence, not a current-tree retraining or confirmatory result.", "",
        "| Contrast | Metric | n | Mean delta | Relative delta | Wins lower | Two-sided p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("balanced_minus_reference", "balanced - reference"), ("active_norm_minus_reference", "active norm - reference")):
        for metric, metric_label in (("mse", "MSE"), ("mae", "MAE"), ("elapsed_sec", "Elapsed seconds")):
            values = result["contrasts"][name][metric]
            p = values["two_sided_p"]
            lines.append(f"| {label} | {metric_label} | {values['n']} | {values['mean']:.6g} | {values['relative_delta_mean']:.3%} | {values['wins_lower']} | {'n/a' if p is None else f'{p:.4g}'} |")
    lines += ["", "## Realized Lag Exposure", "", "Counts are accepted disjoint pairs aggregated over epochs; exposure is each lag's share. Balance span is max exposure minus min exposure for the run, where lower is more even.", "", "| Arm | Dataset | Seed | Lag | Pair count | Exposure | Aggregate span | Mean epoch span |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(result["exposure_rows"], key=lambda item: (item["arm"], item["dataset"], item["seed"], item["lag"])):
        lines.append(f"| {row['arm']} | {row['dataset']} | {row['seed']} | {row['lag']} | {row['pair_count']} | {row['aggregate_exposure']:.3%} | {row['aggregate_balance_span']:.3%} | {row['mean_epoch_balance_span']:.3%} |")
    lines += ["", "## Relation Scale", "", "| Arm | AutoCon loss mean | Global active-anchor fraction | Total epochs | Epochs ran (min-max) |", "|---|---:|---:|---:|---:|"]
    for arm in ("reference", "balanced", "active_norm"):
        values = result["diagnostics"][arm]
        lines.append(f"| {arm} | {values['autocon_loss_mean_over_runs']:.6g} | {values['active_anchor_fraction_mean_over_runs']:.3%} | {values['epochs_ran_total']} | {values['epochs_ran_min']}--{values['epochs_ran_max']} |")
    lines += ["", "## Boundaries", "", "- Balanced pairing is a least-exposed-lag greedy policy over feasible disjoint pairs, not a strict stratified sampler.", "- The six paired coordinates and eight-epoch budget support descriptive diagnostics only; p-values are unadjusted two-sided paired t-tests, not confirmatory evidence.", "- Elapsed time includes the full training job and should not be read as a hardware-independent speed claim."]
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
    for phase_name, arm, balance, active_norm in PHASES:
        rows.extend(_load_phase(plan, phase_name, arm, balance, active_norm))
    result = _build(plan, rows)
    output_json, output_md = _resolve(args.output_json), _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pairs": len(result["pairs"]), "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
