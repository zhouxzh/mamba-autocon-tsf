#!/usr/bin/env python3
"""Audit the sidecar-backed relation-target semantics screen.

The reader is intentionally read-only.  It verifies the immutable phase
contracts, train-only sidecar hashes, objective-v3 telemetry, and the
independent high-ACF diagnostic threshold before producing paired descriptive
metrics.  The inferential unit is dataset x seed; the screen is exploratory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Keep both ``python -m scripts...`` and ``python scripts/...`` usable.
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mamba_acf_relation_coverage_screen import (
    _lag_exposure_summary,
    _positive_lag_set,
)
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


PHASES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("relation_target_hard_raw_h96", "hard", "hard_selected", True),
    ("relation_target_soft05_raw_h96", "soft05", "soft_multi_peak", False),
)
DATASETS = ("ETTh1", "traffic")
SEEDS = (2021, 2022, 2023)
EXPECTED_SOFT_THRESHOLD = 0.5
EXPECTED_DIAGNOSTIC_THRESHOLD = 0.1


def _assert_config(config: Mapping[str, Any], expected: Mapping[str, Any], job_id: str) -> None:
    """Require every contract key and reject ambiguous boolean strings."""

    for key, value in expected.items():
        if key not in config:
            raise ValueError(f"config {key} is missing for {job_id}")
        actual = config[key]
        if isinstance(value, bool):
            if isinstance(actual, bool):
                normalized = actual
            elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual in (0, 1):
                normalized = bool(actual)
            else:
                raise ValueError(f"config {key} is not an explicit boolean for {job_id}")
            actual = normalized
        elif isinstance(value, float):
            try:
                actual = float(actual)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {key} is not numeric for {job_id}") from exc
            if math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12):
                continue
        if actual != value:
            raise ValueError(f"config {key} mismatch for {job_id}: {actual!r} != {value!r}")


def _expected_config(target_mode: str, lag_mask: bool) -> Dict[str, Any]:
    return {
        "model": "MambaACFAutoCon",
        "features": "M",
        "pred_len": 96,
        "AutoCon": True,
        "AutoCon_lambda": 0.03,
        "AutoCon_acf_source": "raw",
        "mamba_variable_mixer": True,
        "mamba_variable_group_size": 1,
        "mamba_use_acf_memory": False,
        "mamba_dynamic_acf": False,
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_batching": "lag_pairs",
        "mamba_acf_relation_lag_balance": "random",
        "mamba_acf_relation_active_anchor_normalization": False,
        "mamba_acf_relation_channel_reduce": "mean",
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_lag_mask": lag_mask,
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_relation_target_mode": target_mode,
        "mamba_acf_relation_soft_min_abs": EXPECTED_SOFT_THRESHOLD,
        "mamba_acf_relation_diagnostic_min_abs": EXPECTED_DIAGNOSTIC_THRESHOLD,
        "mamba_acf_relation_selected_lag_boost": 0.0,
        "mamba_low_rank_head": False,
    }


def _validate_lag_metadata(
    effective_lags: Sequence[int],
    selected_lags: Sequence[int],
    lag_mask: bool,
    job_id: str,
) -> None:
    """Validate the two lag metadata fields under their live semantics.

    ``effective_lags`` is the relation sampler/denominator universe.  With a
    selected-positive mask enabled it is also the persisted selected set.  A
    soft target without that mask still records the broader forecast-memory
    peak list, so the only valid relation is ``effective subset selected``.
    """

    effective = set(int(lag) for lag in effective_lags)
    selected = set(int(lag) for lag in selected_lags)
    if lag_mask:
        if selected != effective:
            raise ValueError(
                f"selected/effective lag mismatch for {job_id}: "
                f"{tuple(sorted(selected))} != {tuple(sorted(effective))}"
            )
    elif not effective.issubset(selected):
        raise ValueError(
            f"effective lags are not contained in selected metadata for {job_id}: "
            f"{tuple(sorted(effective))} not in {tuple(sorted(selected))}"
        )


def _load_phase(plan: Mapping[str, Any], phase_name: str, arm: str,
                target_mode: str, lag_mask: bool) -> List[Dict[str, Any]]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan phases must be a list")
    phase = next((item for item in phases if isinstance(item, Mapping) and item.get("name") == phase_name), None)
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
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        sidecar_hash = hashes.get("acf_prior.npz")
        if not sidecar_hash:
            raise ValueError(f"missing ACF sidecar hash: {job_id}")
        config = _read_json(result_dir / "config.json")
        _assert_config(config, _expected_config(target_mode, lag_mask), job_id)
        if str(config.get("mamba_acf_prior_sha256", "")).lower() != sidecar_hash:
            raise ValueError(f"config ACF sidecar hash mismatch: {job_id}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        if logs.get("acf_relation_target_mode") != target_mode:
            raise ValueError(f"loss-log target mode mismatch: {job_id}")
        if not math.isclose(_finite(logs.get("acf_relation_soft_min_abs"), f"{job_id} soft threshold"), EXPECTED_SOFT_THRESHOLD, abs_tol=1e-12):
            raise ValueError(f"loss-log soft threshold mismatch: {job_id}")
        if not math.isclose(_finite(logs.get("acf_relation_diagnostic_min_abs"), f"{job_id} diagnostic threshold"), EXPECTED_DIAGNOSTIC_THRESHOLD, abs_tol=1e-12):
            raise ValueError(f"loss-log diagnostic threshold mismatch: {job_id}")
        effective_lags = _positive_lag_set(logs.get("acf_relation_effective_lags"), "acf_relation_effective_lags", logs_path)
        selected_lags = _positive_lag_set(logs.get("acf_relation_selected_lags"), "acf_relation_selected_lags", logs_path)
        _validate_lag_metadata(effective_lags, selected_lags, lag_mask, job_id)
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
            "effective_lags": list(effective_lags),
            "selected_lags": list(selected_lags),
            "lag_exposure": lag_exposure,
            "diagnostics": {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS},
            "autocon_loss_mean": _summary_series(logs, "autocon_loss", logs_path)["mean"],
        })
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if seen != expected:
        raise ValueError(f"incomplete coordinate matrix for {phase_name}")
    return rows


def _paired(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    if scipy_stats is not None and len(numbers) > 1:
        test = scipy_stats.ttest_1samp(numbers, 0.0)
        t_stat, p_value = float(test.statistic), float(test.pvalue)
    else:
        t_stat, p_value = (mean / (sd / math.sqrt(len(numbers))) if sd > 0 else 0.0), None
    if len(numbers) > 1 and math.isfinite(sd):
        margin = 1.96 * sd / math.sqrt(len(numbers))
        ci95 = [mean - margin, mean + margin]
    else:
        ci95 = [math.nan, math.nan]
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
        "ci95": ci95,
    }


def _contrast(pairs: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    deltas = [float(item[f"soft05_{metric}"] - item[f"hard_{metric}"]) for item in pairs]
    result = _paired(deltas)
    reference_mean = statistics.fmean(float(item[f"hard_{metric}"]) for item in pairs)
    treatment_mean = statistics.fmean(float(item[f"soft05_{metric}"]) for item in pairs)
    result.update({
        "reference_mean": reference_mean,
        "treatment_mean": treatment_mean,
        "relative_delta_mean": _relative(result["mean"], reference_mean),
    })
    return result


def _stop_decision(
    dataset_contrasts: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the registered screen gates without tuning on test metrics."""

    stop_reasons: List[str] = []
    for dataset in DATASETS:
        contrast = dataset_contrasts[dataset]
        mse = contrast["mse"]
        if mse["wins_lower"] <= 1:
            stop_reasons.append(
                f"{dataset}: soft05 has {mse['wins_lower']}/3 lower-MSE seeds (<=1/3)"
            )
        if mse["relative_delta_mean"] > 0.01:
            stop_reasons.append(
                f"{dataset}: mean MSE worsens by {mse['relative_delta_mean']:.3%} (>1%)"
            )

    expansion_reasons: List[str] = []
    for dataset in DATASETS:
        contrast = dataset_contrasts[dataset]
        mse = contrast["mse"]
        mae = contrast["mae"]
        if mse["relative_delta_mean"] > -0.005:
            expansion_reasons.append(
                f"{dataset}: mean MSE improvement is {-mse['relative_delta_mean']:.3%} (<0.5%)"
            )
        if mse["wins_lower"] < 2:
            expansion_reasons.append(f"{dataset}: fewer than 2/3 lower-MSE seeds")
        if mae["relative_delta_mean"] > 0.005:
            expansion_reasons.append(
                f"{dataset}: MAE worsens by {mae['relative_delta_mean']:.3%} (>0.5%)"
            )
    hard_exclusion = diagnostics["hard"]["high_acf_conditional_excluded_mean_over_runs"]
    soft_exclusion = diagnostics["soft05"]["high_acf_conditional_excluded_mean_over_runs"]
    reduction = (
        0.0
        if hard_exclusion <= 0
        else (hard_exclusion - soft_exclusion) / hard_exclusion
    )
    if reduction < 0.5:
        expansion_reasons.append(
            f"diagnostic high-ACF exclusion reduction is {reduction:.3%} (<50%)"
        )
    return {
        "stop": bool(stop_reasons),
        "stop_reasons": stop_reasons,
        "expansion_eligible": not expansion_reasons,
        "expansion_reasons": expansion_reasons,
    }


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {(row["arm"], row["dataset"], row["seed"]): row for row in rows}
    pairs: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            hard = indexed[("hard", dataset, seed)]
            soft = indexed[("soft05", dataset, seed)]
            if hard["sidecar_sha256"] != soft["sidecar_sha256"]:
                raise ValueError(f"sidecar mismatch for {dataset} seed {seed}")
            pair = {"dataset": dataset, "seed": seed, "sidecar_sha256": hard["sidecar_sha256"]}
            for arm, row in (("hard", hard), ("soft05", soft)):
                for metric in ("mse", "mae", "elapsed_sec"):
                    pair[f"{arm}_{metric}"] = row[metric]
                pair[f"{arm}_job_id"] = row["job_id"]
            pairs.append(pair)
    diagnostics = {}
    for arm in ("hard", "soft05"):
        subset = [row for row in rows if row["arm"] == arm]
        diagnostics[arm] = {
            "autocon_loss_mean_over_runs": statistics.fmean(row["autocon_loss_mean"] for row in subset),
            "active_anchor_fraction_mean_over_runs": statistics.fmean(row["diagnostics"]["acf_relation_global_active_anchor_fraction"]["mean"] for row in subset),
            "positive_count_mean_over_runs": statistics.fmean(row["diagnostics"]["acf_relation_global_positive_count_mean"]["mean"] for row in subset),
            "high_acf_candidate_count_mean_over_runs": statistics.fmean(row["diagnostics"]["acf_relation_global_high_acf_candidate_count"]["mean"] for row in subset),
            "high_acf_conditional_excluded_mean_over_runs": statistics.fmean(row["diagnostics"]["acf_relation_global_high_acf_conditional_excluded_fraction"]["mean"] for row in subset),
            "epochs_ran_values": [int(row["epochs_ran"]) for row in subset],
        }
        diagnostics[arm]["epochs_ran_total"] = sum(diagnostics[arm]["epochs_ran_values"])
        diagnostics[arm]["epochs_ran_min"] = min(diagnostics[arm]["epochs_ran_values"])
        diagnostics[arm]["epochs_ran_max"] = max(diagnostics[arm]["epochs_ran_values"])
    dataset_contrasts = {
        dataset: {
            metric: _contrast(
                [pair for pair in pairs if pair["dataset"] == dataset], metric
            )
            for metric in ("mse", "mae", "elapsed_sec")
        }
        for dataset in DATASETS
    }
    return {
        "schema_version": 1,
        "status": "exploratory_relation_target_screen_not_confirmatory",
        "plan": str(plan["_path"]),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "rows": list(rows),
        "pairs": pairs,
        "contrasts": {metric: _contrast(pairs, metric) for metric in ("mse", "mae", "elapsed_sec")},
        "dataset_contrasts": dataset_contrasts,
        "diagnostics": diagnostics,
        "decision": _stop_decision(dataset_contrasts, diagnostics),
        "sidecar_pair_count": len(pairs),
        "evidence_boundary": "Exploratory paired target-support evidence only; no confirmatory, universal accuracy, or speed claim.",
    }


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MambaACF Relation-Target Screen", "",
        "Status: **exploratory; not confirmatory**.", "",
        "Both arms use the current sidecar-backed raw train-only prior, variable-token MambaACF AutoCon, global relation loss, the same lag-pair sampler, full-rank head, H96 matrix, lambda 0.03, and eight-epoch budget. The only registered intervention is hard selected-peak support versus soft strong-ACF support at `abs(ACF)>0.5`; selected-lag boost is zero. High-ACF telemetry uses an independent `abs(ACF)>0.1` threshold.", "",
        "| Contrast (soft05 - hard) | n | Mean delta | Relative delta | 95% CI | Wins lower | Two-sided p |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for metric, label in (("mse", "MSE"), ("mae", "MAE"), ("elapsed_sec", "Elapsed seconds")):
        value = result["contrasts"][metric]
        ci = value["ci95"]
        p = value["two_sided_p"]
        lines.append(f"| {label} | {value['n']} | {value['mean']:.6g} | {value['relative_delta_mean']:.3%} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['wins_lower']} | {'n/a' if p is None else f'{p:.4g}'} |")
    lines += ["", "## Paired Seeds", "", "| Dataset | Seed | MSE delta | MAE delta | Hard MSE | Soft05 MSE |", "|---|---:|---:|---:|---:|---:|"]
    for pair in result["pairs"]:
        lines.append(f"| {pair['dataset']} | {pair['seed']} | {pair['soft05_mse'] - pair['hard_mse']:.6g} | {pair['soft05_mae'] - pair['hard_mae']:.6g} | {pair['hard_mse']:.6g} | {pair['soft05_mse']:.6g} |")
    lines += ["", "## Target-Support Diagnostics", "", "| Arm | AutoCon loss | Active anchors | Positive count / anchor | High-ACF candidates | High-ACF conditional excluded | Total epochs | Epochs (min-max) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("hard", "soft05"):
        value = result["diagnostics"][arm]
        lines.append(f"| {arm} | {value['autocon_loss_mean_over_runs']:.6g} | {value['active_anchor_fraction_mean_over_runs']:.3%} | {value['positive_count_mean_over_runs']:.3f} | {value['high_acf_candidate_count_mean_over_runs']:.1f} | {value['high_acf_conditional_excluded_mean_over_runs']:.3%} | {value['epochs_ran_total']} | {value['epochs_ran_min']}--{value['epochs_ran_max']} |")
    lines += ["", "## Dataset-Level Gates", "", "| Dataset | MSE delta | Relative MSE | MSE wins | MAE relative delta |", "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        value = result["dataset_contrasts"][dataset]
        lines.append(
            f"| {dataset} | {value['mse']['mean']:.6g} | {value['mse']['relative_delta_mean']:.3%} | "
            f"{value['mse']['wins_lower']}/3 | {value['mae']['relative_delta_mean']:.3%} |"
        )
    decision = result["decision"]
    lines += [
        "", "## Registered Decision", "",
        f"- Stop rule triggered: **{'yes' if decision['stop'] else 'no'}**.",
    ]
    lines.extend(f"- Stop reason: {reason}." for reason in decision["stop_reasons"])
    lines.append(f"- Expansion eligible: **{'yes' if decision['expansion_eligible'] else 'no'}**.")
    lines.extend(f"- Expansion gate: {reason}." for reason in decision["expansion_reasons"])
    lines += [
        "", "## Boundaries", "",
        "- The independent diagnostic threshold prevents `soft_multi_peak` from making its own exclusion rate tautologically zero.",
        "- The soft arm changes target support and relation-loss density together; it is a mechanism/sensitivity screen, not a pure optimization or false-negative proof.",
        "- Pairing is at dataset x seed (`n=6`); p-values are unadjusted descriptive tests and cannot select a model.",
        "- No memory branch, longer horizon, cross-channel relation reduction, or runtime claim is tested.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/mamba_acf_relation_target_screen_plan.json"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    plan = _read_json(plan_path)
    plan = dict(plan)
    plan["_path"] = str(plan_path.resolve())
    rows: List[Dict[str, Any]] = []
    for phase_name, arm, target_mode, lag_mask in PHASES:
        rows.extend(_load_phase(plan, phase_name, arm, target_mode, lag_mask))
    result = build_readout(plan, rows)
    output_json = args.output_json or (REPO_ROOT / "artifacts/paper_final/mamba_acf_relation_target_screen_20260901/relation_target_screen_readout.json")
    output_markdown = args.output_markdown or (REPO_ROOT / "docs/research/mamba_acf_relation_target_screen_readout_20260901.md")
    output_json = output_json if output_json.is_absolute() else REPO_ROOT / output_json
    output_markdown = output_markdown if output_markdown.is_absolute() else REPO_ROOT / output_markdown
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_markdown.write_text(markdown(result), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
