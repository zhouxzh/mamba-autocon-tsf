#!/usr/bin/env python3
"""Analyze the paired three-arm MambaACF channel-gate screen.

The reader verifies phase validation, manifest/artifact hashes, exact gate
configuration, and shared train-only ACF sidecars before computing paired
descriptive contrasts.  It is intentionally exploratory and never changes
training artifacts.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    ("channel_gate_off_raw_h96", "off", False, False),
    ("channel_gate_on_raw_h96", "gate", True, False),
    ("acf_channel_gate_on_raw_h96", "acf_gate", True, True),
)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)


def _checkpoint_state(path: Path) -> Mapping[str, Any]:
    try:
        import torch

        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            value = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - surfaced as an audit error
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    state = value.get("model", value)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint model state is not a mapping: {path}")
    return state


def _scalar_tensor(state: Mapping[str, Any], key: str, path: Path) -> float:
    value = state.get(key)
    if value is None:
        raise ValueError(f"checkpoint is missing {key}: {path}")
    try:
        result = float(value.detach().cpu().reshape(-1)[0].item())
    except Exception as exc:
        raise ValueError(f"checkpoint value is not scalar {key}: {path}") from exc
    return _finite(result, f"{path.name}:{key}")


def _load_phase(
    plan: Mapping[str, Any],
    phase_name: str,
    arm: str,
    gate_expected: bool,
    acf_gate_expected: bool,
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
        if not math.isclose(float(record.get("autocon_lambda")), 0.03, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"unexpected AutoCon lambda: {record.get('job_id')}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected artifact: {record.get('job_id')}")
        metrics_path = _resolve(str(expected_artifact))
        if (
            metrics_path.name != "metrics.npy"
            or not metrics_path.is_file()
            or not metrics_path.is_relative_to(result_root)
        ):
            raise ValueError(f"metrics artifact outside result root: {record.get('job_id')}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        config = _read_json(result_dir / "config.json")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {record.get('job_id')}")
        expected_config = {
            "model": "MambaACFAutoCon",
            "features": "M",
            "pred_len": 96,
            "AutoCon": True,
            "AutoCon_lambda": 0.03,
            "AutoCon_acf_source": "raw",
            "mamba_variable_mixer": True,
            "mamba_use_acf_memory": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_target_mode": "hard_selected",
            "mamba_channel_gate": gate_expected,
            "mamba_channel_gate_rank": 8,
            "mamba_acf_channel_gate": acf_gate_expected,
        }
        for key, expected in expected_config.items():
            actual = config.get(key)
            if isinstance(expected, bool):
                actual = bool(actual)
            elif isinstance(expected, float):
                try:
                    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"config {key} mismatch: {record.get('job_id')}") from exc
                continue
            if actual != expected:
                raise ValueError(
                    f"config {key} mismatch: {record.get('job_id')}: {actual!r} != {expected!r}"
                )
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        state = _checkpoint_state(checkpoint_path)
        if gate_expected:
            gate_scale = _scalar_tensor(state, "channel_gate.residual_scale", checkpoint_path)
            acf_latent_scale = (
                _scalar_tensor(state, "channel_gate.acf_latent_scale", checkpoint_path)
                if acf_gate_expected
                else None
            )
        else:
            if any(str(key).startswith("channel_gate.") for key in state):
                raise ValueError(f"disabled gate has checkpoint parameters: {checkpoint_path}")
            gate_scale = 0.0
            acf_latent_scale = None
        diagnostics = {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS}
        sidecar_hash = hashes.get("acf_prior.npz")
        if not sidecar_hash:
            raise ValueError(f"missing sidecar hash: {record.get('job_id')}")
        rows.append(
            {
                "phase": phase_name,
                "arm": arm,
                "dataset": dataset,
                "seed": seed,
                "mse": _finite(scaled.get("mse"), f"{record.get('job_id')} mse"),
                "mae": _finite(scaled.get("mae"), f"{record.get('job_id')} mae"),
                "elapsed_sec": _finite(record.get("elapsed_sec"), f"{record.get('job_id')} elapsed"),
                "job_id": str(record.get("job_id")),
                "sidecar_sha256": sidecar_hash,
                "gate_residual_scale": gate_scale,
                "acf_latent_scale": acf_latent_scale,
                "diagnostics": diagnostics,
                "autocon_loss_mean": _summary_series(logs, "autocon_loss", logs_path)["mean"],
            }
        )
    expected_coordinates = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    if seen != expected_coordinates:
        raise ValueError(f"phase coordinate matrix is incomplete: {phase_name}")
    return rows


def _paired_test(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    if scipy_stats is not None and len(numbers) > 1:
        result = scipy_stats.ttest_1samp(numbers, 0.0)
        t_stat, p_value = float(result.statistic), float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat, p_value = mean / (sd / math.sqrt(len(numbers))), None
    else:
        t_stat, p_value = (0.0 if mean == 0 else math.copysign(math.inf, mean)), None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
    }


def _contrast(metric: str, reference: str, treatment: str, pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    deltas = [float(item[f"{treatment}_{metric}"] - item[f"{reference}_{metric}"]) for item in pairs]
    result = _paired_test(deltas)
    result["reference_mean"] = statistics.fmean(item[f"{reference}_{metric}"] for item in pairs)
    result["treatment_mean"] = statistics.fmean(item[f"{treatment}_{metric}"] for item in pairs)
    result["relative_delta_mean"] = _relative(result["mean"], result["reference_mean"])
    result["reference"] = reference
    result["treatment"] = treatment
    return result


def _build(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {(str(row["arm"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            off = indexed[("off", dataset, seed)]
            gate = indexed[("gate", dataset, seed)]
            acf = indexed[("acf_gate", dataset, seed)]
            sidecars = {off["sidecar_sha256"], gate["sidecar_sha256"], acf["sidecar_sha256"]}
            if len(sidecars) != 1:
                raise ValueError(f"sidecar differs between gate arms: {dataset} seed {seed}")
            item: Dict[str, Any] = {"dataset": dataset, "seed": seed, "sidecar_sha256": off["sidecar_sha256"]}
            for label, row in (("off", off), ("gate", gate), ("acf_gate", acf)):
                for metric in ("mse", "mae", "elapsed_sec"):
                    item[f"{label}_{metric}"] = row[metric]
                item[f"{label}_gate_residual_scale"] = row["gate_residual_scale"]
                item[f"{label}_acf_latent_scale"] = row["acf_latent_scale"]
                item[f"{label}_job_id"] = row["job_id"]
            pairs.append(item)

    contrasts: Dict[str, Any] = {}
    for name, reference, treatment in (
        ("gate_minus_off", "off", "gate"),
        ("acf_gate_minus_gate", "gate", "acf_gate"),
        ("acf_gate_minus_off", "off", "acf_gate"),
    ):
        contrasts[name] = {
            "mse": _contrast("mse", reference, treatment, pairs),
            "mae": _contrast("mae", reference, treatment, pairs),
            "elapsed_sec": _contrast("elapsed_sec", reference, treatment, pairs),
        }
        by_dataset = {}
        for dataset in EXPECTED_DATASETS:
            subset = [item for item in pairs if item["dataset"] == dataset]
            by_dataset[dataset] = {
                "n": len(subset),
                "mse": _contrast("mse", reference, treatment, subset),
                "mae": _contrast("mae", reference, treatment, subset),
            }
        contrasts[name]["by_dataset"] = by_dataset

    diagnostics: Dict[str, Any] = {}
    for arm in ("off", "gate", "acf_gate"):
        subset = [row for row in rows if row["arm"] == arm]
        diagnostics[arm] = {
            key: {"mean_over_runs": statistics.fmean(row["diagnostics"][key]["mean"] for row in subset)}
            for key in RELATION_LOG_KEYS
        }
        diagnostics[arm]["autocon_loss_mean_over_runs"] = statistics.fmean(row["autocon_loss_mean"] for row in subset)
        diagnostics[arm]["gate_residual_scale_mean"] = statistics.fmean(row["gate_residual_scale"] for row in subset)
        active_scales = [abs(float(row["gate_residual_scale"])) for row in subset]
        diagnostics[arm]["gate_residual_scale_abs_max"] = max(active_scales)
        acf_scales = [float(row["acf_latent_scale"]) for row in subset if row["acf_latent_scale"] is not None]
        diagnostics[arm]["acf_latent_scale_mean"] = statistics.fmean(acf_scales) if acf_scales else None
    return {
        "schema_version": 1,
        "status": "exploratory_acf_channel_gate_screen_not_confirmatory",
        "plan": str(plan["_path"]),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "phases": [name for name, *_ in PHASES],
        "rows": list(rows),
        "pairs": pairs,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
        "sidecar_triplet_count": len(pairs),
        "evidence_boundary": "Paired exploratory raw-source channel-gate evidence; not confirmatory or publication-ready.",
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MambaACF Channel-Gate Screen",
        "",
        "Status: **exploratory; not confirmatory**.",
        "",
        "The three arms share the raw-source train-only ACF sidecar, variable-token MambaACF AutoCon model, relation target, sampler, data, seeds, and budget. They differ only in the low-rank channel gate: disabled, ordinary, or additionally conditioned on the train-only ACF summary.",
        "",
        "Negative deltas favor the treatment arm. The paired sample is six dataset/seed coordinates and is descriptive only.",
        "",
        "| Contrast | Metric | n | Mean delta | Relative delta | Wins lower | Two-sided p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for contrast_name, title in (
        ("gate_minus_off", "gate - off"),
        ("acf_gate_minus_gate", "ACF gate - gate"),
        ("acf_gate_minus_off", "ACF gate - off"),
    ):
        for metric, label in (("mse", "MSE"), ("mae", "MAE")):
            values = result["contrasts"][contrast_name][metric]
            p = values.get("two_sided_p")
            lines.append(
                f"| {title} | {label} | {values['n']} | {values['mean']:.6g} | {values['relative_delta_mean']:.3%} | {values['wins_lower']} | {'n/a' if p is None else f'{p:.4g}'} |"
            )
    lines += [
        "",
        "## Dataset Splits",
        "",
        "| Contrast | Dataset | MSE delta | MSE relative | MAE delta | MAE relative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for contrast_name, title in (
        ("gate_minus_off", "gate - off"),
        ("acf_gate_minus_gate", "ACF gate - gate"),
        ("acf_gate_minus_off", "ACF gate - off"),
    ):
        for dataset, values in result["contrasts"][contrast_name]["by_dataset"].items():
            lines.append(
                f"| {title} | {dataset} | {values['mse']['mean']:.6g} | {values['mse']['relative_delta_mean']:.3%} | {values['mae']['mean']:.6g} | {values['mae']['relative_delta_mean']:.3%} |"
            )
    lines += [
        "",
        "## Gate Activation",
        "",
        "Checkpoint scalars are reported to distinguish an ineffective architecture from an adapter that remained closed behind ReZero.",
        "",
        "| Arm | Mean gate residual scale | Max absolute gate scale | Mean ACF latent scale | AutoCon loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("off", "gate", "acf_gate"):
        values = result["diagnostics"][arm]
        acf_scale = values["acf_latent_scale_mean"]
        lines.append(
            f"| {arm} | {values['gate_residual_scale_mean']:.6g} | {values['gate_residual_scale_abs_max']:.6g} | {'n/a' if acf_scale is None else f'{acf_scale:.6g}'} | {values['autocon_loss_mean_over_runs']:.6g} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- This screen does not establish a universal accuracy or speed advantage; elapsed time is especially noisy at this short budget.",
        "- It does not test longer horizons, parameter-matched baselines, alternative channel orderings, or balanced relation sampling.",
        "- The ACF-conditioned arm uses a train-only descriptor, but the broader residual/trend source path is not part of this raw-source causality claim.",
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
    for phase_name, arm, gate_expected, acf_gate_expected in PHASES:
        rows.extend(_load_phase(plan, phase_name, arm, gate_expected, acf_gate_expected))
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
