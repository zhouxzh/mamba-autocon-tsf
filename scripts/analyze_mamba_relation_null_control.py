#!/usr/bin/env python3
"""Read out the paired MambaACF relation-loss null-control screen.

The reader is deliberately independent of the three-arm live-screen reader:
both phases here use the same AutoCon model and sampler, and differ only in
the registered relation-loss weight.  It verifies the phase validation,
manifest/artifact hashes, train-only sidecar identity, and the per-seed pairing
before computing descriptive deltas.  It never changes training artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PHASES = (
    ("relation_lambda_zero_null_h96", "lambda_zero"),
    ("relation_lambda_point003_h96", "lambda_point003"),
)
RELATION_LOG_KEYS = (
    "acf_relation_global_valid_fraction",
    "acf_relation_global_active_anchor_fraction",
    "acf_relation_global_candidate_fraction",
    "acf_relation_global_candidate_active_anchor_fraction",
    "acf_relation_global_positive_count_mean",
    "acf_relation_global_high_acf_excluded_fraction",
    "acf_relation_global_high_acf_candidate_count",
    "acf_relation_global_high_acf_excluded_count",
    "acf_relation_global_high_acf_conditional_excluded_fraction",
    "acf_relation_global_sign_target_count",
    "acf_relation_global_sign_accuracy",
    "acf_relation_global_sign_hinge",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _hash_map_for_record(record: Mapping[str, Any], result_dir: Path, checkpoint_dir: Path) -> Dict[str, str]:
    raw = record.get("artifact_hashes")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"manifest record has no artifact hash map: {record.get('job_id')}")
    output: Dict[str, str] = {}
    result_names = {
        "config.json", "metrics.npy", "original_metrics.npy", "channel_metrics.npy",
        "stream_metrics.json", "stream_samples.npz", "pred.npy", "true.npy",
        "training_complete.json", "acf_prior.npz",
    }
    checkpoint_names = {"checkpoint.pth", "loss_logs.json"}
    for name, expected in raw.items():
        name = str(name)
        if name in result_names:
            path = result_dir / name
            root = result_dir
        elif name in checkpoint_names:
            path = checkpoint_dir / name
            root = checkpoint_dir
        else:
            raise ValueError(f"unknown artifact hash {name!r}: {record.get('job_id')}")
        if not path.is_file() or not _inside(path, root):
            raise ValueError(f"missing artifact {path}: {record.get('job_id')}")
        actual = _sha256(path)
        if actual != str(expected):
            raise ValueError(f"artifact hash mismatch {path}: {record.get('job_id')}")
        output[name] = actual
    return output


def _manifest_records(phase_root: Path, validation: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    if validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"manifest hash changed after validation: {manifest_path}")
    records: List[Mapping[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest line {line_number}: {exc}") from exc
            if not isinstance(item, Mapping):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(item)
    if not records:
        raise ValueError(f"empty manifest: {manifest_path}")
    expected = int(validation.get("expected_jobs", -1))
    if len(records) != expected or validation.get("manifest_unique_jobs") != expected:
        raise ValueError(f"manifest coverage mismatch: {phase_root}")
    if validation.get("malformed_manifest_lines", 0) != 0:
        raise ValueError(f"validation reports malformed manifest lines: {phase_root}")
    if validation.get("errors") not in (None, []):
        raise ValueError(f"phase validation contains errors: {phase_root}")
    if validation.get("status_counts") != {"ok": expected}:
        raise ValueError(f"phase is not all successful: {phase_root}")
    return records


def _summary_series(logs: Mapping[str, Any], key: str, path: Path) -> Dict[str, float]:
    values = logs.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"missing relation diagnostic {key}: {path}")
    numbers = [_finite(value, f"{path.name}:{key}") for value in values]
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "last": numbers[-1],
    }


def _load_phase(plan: Mapping[str, Any], phase_name: str, kind: str) -> List[Dict[str, Any]]:
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
    expected_lambda = 0.0 if kind == "lambda_zero" else 0.03
    rows: List[Dict[str, Any]] = []
    coordinates: set[Tuple[str, int]] = set()
    for record in records:
        if record.get("status") != "ok" or record.get("phase") != phase_name:
            raise ValueError(f"unsuccessful record: {record.get('job_id')}")
        dataset = str(record.get("dataset"))
        seed = int(record.get("seed"))
        coordinate = (dataset, seed)
        if coordinate in coordinates:
            raise ValueError(f"duplicate coordinate: {phase_name} {coordinate}")
        coordinates.add(coordinate)
        if not math.isclose(float(record.get("autocon_lambda")), expected_lambda, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"manifest lambda mismatch: {record.get('job_id')}")
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"missing expected artifact: {record.get('job_id')}")
        metrics_path = _resolve(str(expected_artifact))
        if metrics_path.name != "metrics.npy" or not metrics_path.is_file() or not _inside(metrics_path, result_root):
            raise ValueError(f"metrics artifact outside result root: {record.get('job_id')}")
        result_dir = metrics_path.parent
        checkpoint_dir = checkpoint_root / result_dir.name
        hashes = _hash_map_for_record(record, result_dir, checkpoint_dir)
        config = _read_json(result_dir / "config.json")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"stream metrics has no scaled section: {record.get('job_id')}")
        mse = _finite(scaled.get("mse"), f"{record.get('job_id')} mse")
        mae = _finite(scaled.get("mae"), f"{record.get('job_id')} mae")
        if not math.isclose(float(config.get("AutoCon_lambda")), expected_lambda, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"config lambda mismatch: {record.get('job_id')}")
        for key, expected in {
            "model": "MambaACFAutoCon",
            "features": "M",
            "pred_len": 96,
            "mamba_variable_mixer": True,
            "mamba_use_acf_memory": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_target_mode": "hard_selected",
        }.items():
            actual = config.get(key)
            if isinstance(expected, bool):
                actual = bool(actual)
            if actual != expected:
                raise ValueError(f"config {key} mismatch for {record.get('job_id')}: {actual!r}")
        logs_path = checkpoint_dir / "loss_logs.json"
        logs = _read_json(logs_path)
        if logs.get("acf_relation_objective_version") != 3:
            raise ValueError(f"relation objective is not v3: {logs_path}")
        diagnostics = {key: _summary_series(logs, key, logs_path) for key in RELATION_LOG_KEYS}
        sidecar_hash = hashes.get("acf_prior.npz")
        if not sidecar_hash:
            raise ValueError(f"missing ACF sidecar hash: {record.get('job_id')}")
        rows.append({
            "phase": phase_name,
            "kind": kind,
            "dataset": dataset,
            "seed": seed,
            "mse": mse,
            "mae": mae,
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{record.get('job_id')} elapsed"),
            "job_id": str(record.get("job_id")),
            "sidecar_sha256": sidecar_hash,
            "diagnostics": diagnostics,
            "autocon_loss_mean": _summary_series(logs, "autocon_loss", logs_path)["mean"],
            "effective_lambda_last": _summary_series(logs, "effective_lambda", logs_path)["last"],
        })
    if len(rows) != 6:
        raise ValueError(f"expected six rows in {phase_name}, got {len(rows)}")
    return rows


def _paired_test(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else math.nan
    if scipy_stats is not None and len(numbers) > 1:
        result = scipy_stats.ttest_1samp(numbers, 0.0)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
    elif math.isfinite(sd) and sd > 0:
        t_stat = mean / (sd / math.sqrt(len(numbers)))
        p_value = None
    else:
        t_stat = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p_value = None
    return {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0 for value in numbers),
        "ties": sum(value == 0 for value in numbers),
        "t_statistic": t_stat,
        "two_sided_p": p_value,
    }


def _relative(delta: float, reference: float) -> float:
    return delta / reference if reference else math.nan


def _build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    indexed = {(str(row["kind"]), str(row["dataset"]), int(row["seed"])): row for row in rows}
    coordinates = sorted({(str(row["dataset"]), int(row["seed"])) for row in rows})
    pairs = []
    for dataset, seed in coordinates:
        null = indexed[("lambda_zero", dataset, seed)]
        treatment = indexed[("lambda_point003", dataset, seed)]
        sidecar_match = null["sidecar_sha256"] == treatment["sidecar_sha256"]
        pairs.append({
            "dataset": dataset,
            "seed": seed,
            "null_mse": null["mse"],
            "treatment_mse": treatment["mse"],
            "delta_mse": treatment["mse"] - null["mse"],
            "relative_delta_mse": _relative(treatment["mse"] - null["mse"], null["mse"]),
            "null_mae": null["mae"],
            "treatment_mae": treatment["mae"],
            "delta_mae": treatment["mae"] - null["mae"],
            "relative_delta_mae": _relative(treatment["mae"] - null["mae"], null["mae"]),
            "null_elapsed_sec": null["elapsed_sec"],
            "treatment_elapsed_sec": treatment["elapsed_sec"],
            "sidecar_sha256_equal": sidecar_match,
            "null_job_id": null["job_id"],
            "treatment_job_id": treatment["job_id"],
        })
    if not pairs or not all(pair["sidecar_sha256_equal"] for pair in pairs):
        raise ValueError("paired phases do not share identical ACF sidecars")

    def contrast(items: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
        deltas = [float(item[f"delta_{metric}"]) for item in items]
        null_values = [float(item[f"null_{metric}"]) for item in items]
        treatment_values = [float(item[f"treatment_{metric}"]) for item in items]
        result = _paired_test(deltas)
        result.update({
            "reference_mean": statistics.fmean(null_values),
            "treatment_mean": statistics.fmean(treatment_values),
            "relative_delta_mean": _relative(result["mean"], statistics.fmean(null_values)),
        })
        return result

    by_dataset = {}
    for dataset in sorted({pair["dataset"] for pair in pairs}):
        subset = [pair for pair in pairs if pair["dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(subset),
            "mse": contrast(subset, "mse"),
            "mae": contrast(subset, "mae"),
        }

    diagnostic_summary: Dict[str, Any] = {}
    for kind in ("lambda_zero", "lambda_point003"):
        subset = [row for row in rows if row["kind"] == kind]
        diagnostic_summary[kind] = {
            key: {
                "mean_over_runs": statistics.fmean(
                    row["diagnostics"][key]["mean"] for row in subset
                )
            }
            for key in RELATION_LOG_KEYS
        }
        diagnostic_summary[kind]["autocon_loss_mean_over_runs"] = statistics.fmean(
            row["autocon_loss_mean"] for row in subset
        )
        diagnostic_summary[kind]["effective_lambda_last_values"] = [
            row["effective_lambda_last"] for row in subset
        ]

    return {
        "schema_version": 1,
        "status": "exploratory_relation_null_control_not_confirmatory",
        "plan": str(plan.get("_path", "")),
        "plan_sha256": _sha256(Path(str(plan["_path"]))),
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "phases": [name for name, _ in EXPECTED_PHASES],
        "rows": list(rows),
        "pairs": pairs,
        "contrasts": {
            "lambda_0.03_minus_lambda_0.0": {
                "mse": contrast(pairs, "mse"),
                "mae": contrast(pairs, "mae"),
            },
            "by_dataset": by_dataset,
        },
        "diagnostics": diagnostic_summary,
        "sidecar_pair_count": sum(pair["sidecar_sha256_equal"] for pair in pairs),
        "evidence_boundary": "Paired exploratory null-control evidence; not a confirmatory or publication claim.",
    }


def _markdown(result: Mapping[str, Any]) -> str:
    contrast = result["contrasts"]["lambda_0.03_minus_lambda_0.0"]
    lines = [
        "# MambaACF Relation Null Control",
        "",
        "Status: **exploratory; not confirmatory**.",
        "",
        "The two arms use the same variable-token MambaACF AutoCon model, train-only ACF sidecars, relation target, lag-pair sampler, data, seeds, and budget. They differ only in relation-loss weight (`lambda=0.0` versus `0.03`).",
        "",
        "## Paired Readout",
        "",
        "| Contrast (0.03 - 0.0) | n | Mean delta | Relative delta | Wins lower | Two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in (("MSE", contrast["mse"]), ("MAE", contrast["mae"])):
        p = values.get("two_sided_p")
        p_text = "n/a" if p is None else f"{p:.4g}"
        lines.append(
            f"| {label} | {values['n']} | {values['mean']:.6g} | {values['relative_delta_mean']:.3%} | {values['wins_lower']} | {p_text} |"
        )
    lines += [
        "",
        "Negative deltas favor the non-zero relation loss. The result is descriptive and does not establish a general accuracy or speed claim.",
        "",
        "## Dataset Splits",
        "",
        "| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in result["contrasts"]["by_dataset"].items():
        mse, mae = values["mse"], values["mae"]
        lines.append(
            f"| {dataset} | {values['n']} | {mse['mean']:.6g} | {mse['relative_delta_mean']:.3%} | {mae['mean']:.6g} | {mae['relative_delta_mean']:.3%} |"
        )
    lines += [
        "",
        "## Objective Diagnostics",
        "",
        "The active-anchor fraction and relation-loss scale must be reported because global inactive anchors are currently averaged as zeros. These diagnostics are not evidence of representation quality by themselves.",
        "",
        "| Arm | AutoCon loss mean | Global valid fraction | Active-anchor fraction | Positive count / anchor | High-ACF candidates | High-ACF conditional excluded | Sign accuracy | Sign hinge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, label in (("lambda_zero", "lambda=0.0"), ("lambda_point003", "lambda=0.03")):
        diag = result["diagnostics"][kind]
        lines.append(
            f"| {label} | {diag['autocon_loss_mean_over_runs']:.6g} | "
            f"{diag['acf_relation_global_valid_fraction']['mean_over_runs']:.3%} | "
            f"{diag['acf_relation_global_active_anchor_fraction']['mean_over_runs']:.3%} | "
            f"{diag['acf_relation_global_positive_count_mean']['mean_over_runs']:.3f} | "
            f"{diag['acf_relation_global_high_acf_candidate_count']['mean_over_runs']:.1f} | "
            f"{diag['acf_relation_global_high_acf_conditional_excluded_fraction']['mean_over_runs']:.3%} | "
            f"{diag['acf_relation_global_sign_accuracy']['mean_over_runs']:.3%} | "
            f"{diag['acf_relation_global_sign_hinge']['mean_over_runs']:.4f} |"
        )
    lines += [
        "",
        "## Boundaries",
        "",
        "- This control isolates the relation-loss weight from the AutoCon representation head and sampler only for this registered H96 matrix.",
        "- It does not test raw-source strict causality, longer horizons, alternative target modes, per-channel relation modeling, or runtime advantage.",
        "- The current residual ACF transform uses a centered window smoother; it is not an end-to-end strict-causal transform.",
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
    for phase_name, kind in EXPECTED_PHASES:
        rows.extend(_load_phase(plan, phase_name, kind))
    result = _build_readout(plan, rows)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pairs": len(result["pairs"]), "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
