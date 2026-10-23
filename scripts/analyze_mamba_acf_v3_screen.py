#!/usr/bin/env python3
"""Read-only statistical readout for the MambaACF v3 relation screen.

The script reads the validated JSONL manifests and the immutable
``stream_metrics.json`` artifacts under one explicitly provided phase root.
It does not select a model, rewrite result files, or treat forecast windows as
independent observations.  The inferential unit is one seed/run.

The default output is a Markdown report.  Three registered contrasts are
reported with treatment-minus-reference deltas (negative means lower error):

* available memory versus plain MambaACF;
* soft relation versus hard relation under the non-overlap control;
* memory plus soft relation versus available memory.

Example::

    PYTHONPATH=. conda run --no-capture-output -n mamba python \
        scripts/analyze_mamba_acf_v3_screen.py \
        --output docs/research/mamba_acf_v3_screen_readout.md

SciPy is optional.  If it is unavailable, the descriptive summaries are still
written and the t/Wilcoxon columns are shown as ``n/a``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Optional so that the descriptive report remains portable.
    import numpy as np
except Exception:  # pragma: no cover - exercised only in minimal environments.
    np = None  # type: ignore[assignment]

try:  # Optional inference dependency.
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - exercised only when SciPy is absent.
    scipy_stats = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
PHASE_ORDER: Tuple[str, ...] = (
    "plain_screen",
    "available_memory_screen",
    "hard_relation_nonoverlap_screen",
    "soft_relation_nonoverlap_screen",
    "available_memory_soft_relation_nonoverlap_screen",
)
PHASE_DIRECTORIES: Mapping[str, str] = {
    "plain_screen": "plain",
    "available_memory_screen": "available_memory",
    "hard_relation_nonoverlap_screen": "hard_relation_nonoverlap",
    "soft_relation_nonoverlap_screen": "soft_relation_nonoverlap",
    "available_memory_soft_relation_nonoverlap_screen": "available_memory_soft_relation_nonoverlap",
}

# (label, treatment phase, reference phase).  Keep this list explicit: it is
# the comparison contract in mamba_acf_v3_relation_screen_plan.json.
CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    (
        "available memory vs plain",
        "available_memory_screen",
        "plain_screen",
    ),
    (
        "soft relation vs hard relation",
        "soft_relation_nonoverlap_screen",
        "hard_relation_nonoverlap_screen",
    ),
    (
        "memory + soft relation vs available memory",
        "available_memory_soft_relation_nonoverlap_screen",
        "available_memory_screen",
    ),
)


def _finite(value: Any) -> Optional[float]:
    """Convert a persisted value to a finite float, or return ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any, default: bool = False) -> bool:
    """Parse JSON/CLI boolean representations without ``bool('false')``."""

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


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _inside(path: Path, root: Path) -> bool:
    """Return whether a resolved path is below (or equal to) ``root``."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _manifest_records(phase_dir: Path) -> List[Mapping[str, Any]]:
    path = phase_dir / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    records: List[Mapping[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, Mapping):
                malformed += 1
                continue
            records.append(record)
    if malformed:
        raise ValueError(f"{path} contains {malformed} malformed JSONL rows")
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def _metric_artifact(record: Mapping[str, Any], phase_dir: Path) -> Tuple[Path, Path]:
    expected = record.get("expected_artifact")
    if not expected:
        raise ValueError(f"manifest job has no expected_artifact: {record.get('job_id')}")
    metrics_path = Path(str(expected)).expanduser().resolve()
    if not metrics_path.is_file():
        raise ValueError(f"missing expected metrics artifact for {record.get('job_id')}: {metrics_path}")
    result_dir = metrics_path.parent
    result_root = (phase_dir / "results").resolve()
    if not _inside(result_dir, result_root):
        raise ValueError(
            f"artifact outside phase results root for {record.get('job_id')}: {result_dir}"
        )
    return result_dir, result_root


def _read_scaled_metrics(result_dir: Path) -> Tuple[float, float]:
    """Read scaled MSE/MAE from the streaming artifact.

    ``stream_metrics.json`` is the canonical memory-safe artifact for this
    screen.  A small ``metrics.npy`` fallback keeps the reader useful for an
    older complete run while preserving the same metric order (MAE, MSE, ...).
    """

    stream_path = result_dir / "stream_metrics.json"
    if stream_path.is_file():
        payload = _read_json(stream_path)
        scaled = payload.get("scaled")
        if isinstance(scaled, Mapping):
            mse = _finite(scaled.get("mse"))
            mae = _finite(scaled.get("mae"))
            if mse is not None and mae is not None:
                return mse, mae
        raise ValueError(f"stream metrics missing finite scaled mse/mae: {stream_path}")

    # This branch is not expected for the v3 artifacts, but gives a clear
    # compatibility path for a complete non-streaming result.
    metrics_path = result_dir / "metrics.npy"
    if metrics_path.is_file() and np is not None:
        try:
            values = np.asarray(np.load(metrics_path, allow_pickle=False), dtype=float).reshape(-1)
        except (OSError, ValueError, TypeError, EOFError) as exc:
            raise ValueError(f"cannot read metrics.npy: {metrics_path}: {exc}") from exc
        if values.size >= 2 and math.isfinite(float(values[0])) and math.isfinite(float(values[1])):
            return float(values[1]), float(values[0])
    raise ValueError(f"missing complete stream_metrics.json/metrics.npy: {result_dir}")


def _seed(record: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    value = record.get("seed", config.get("run_seed", config.get("seed")))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid seed for {record.get('job_id')}: {value!r}") from exc


def load_rows(root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and validate one row per successful manifest job."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"screen root does not exist: {root}")

    discovered = {
        phase: root / directory
        for phase, directory in PHASE_DIRECTORIES.items()
        if (root / directory / "logs" / "job_manifest.jsonl").is_file()
    }
    missing = [phase for phase in PHASE_ORDER if phase not in discovered]
    if missing:
        raise ValueError(f"screen root is missing expected phases: {', '.join(missing)}")

    rows: List[Dict[str, Any]] = []
    phase_meta: Dict[str, Any] = {}
    seen: set[Tuple[str, str, int, int]] = set()

    for phase in PHASE_ORDER:
        phase_dir = discovered[phase]
        validation_path = phase_dir / "phase_validation.json"
        validation = _read_json(validation_path) if validation_path.is_file() else {}
        if validation and validation.get("valid") is not True:
            raise ValueError(f"phase validation is not valid: {validation_path}")
        if validation and str(validation.get("phase", "")) != phase:
            raise ValueError(
                f"phase validation mismatch for {phase_dir}: expected {phase!r}, "
                f"got {validation.get('phase')!r}"
            )
        records = _manifest_records(phase_dir)
        if validation:
            try:
                expected_jobs = int(validation["expected_jobs"])
                unique_jobs = int(validation["manifest_unique_jobs"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid job coverage in phase validation: {validation_path}") from exc
            if len(records) != expected_jobs or unique_jobs != expected_jobs:
                raise ValueError(
                    f"phase validation/job-manifest coverage mismatch for {phase}: "
                    f"manifest={len(records)}, expected={expected_jobs}, unique={unique_jobs}"
                )
            job_ids = [str(record.get("job_id", "")) for record in records]
            if len(set(job_ids)) != len(job_ids) or any(not job_id for job_id in job_ids):
                raise ValueError(f"manifest has duplicate or missing job IDs: {phase_dir}")
        manifest_phases = {str(record.get("phase", "")) for record in records}
        if manifest_phases != {phase}:
            raise ValueError(
                f"manifest phase mismatch for {phase_dir}: expected {phase!r}, "
                f"got {sorted(manifest_phases)!r}"
            )
        phase_meta[phase] = {
            "jobs": len(records),
            "validation": validation,
            "manifest": str(phase_dir / "logs" / "job_manifest.jsonl"),
        }
        for record in records:
            if str(record.get("status", "")) != "ok":
                raise ValueError(
                    f"phase {phase} contains non-successful job: {record.get('job_id')}"
                )
            result_dir, _ = _metric_artifact(record, phase_dir)
            config_path = result_dir / "config.json"
            if not config_path.is_file():
                raise ValueError(f"missing config.json for {record.get('job_id')}: {config_path}")
            config = _read_json(config_path)
            dataset = str(record.get("dataset", config.get("data", "")))
            if not dataset:
                raise ValueError(f"missing dataset for {record.get('job_id')}")
            seed = _seed(record, config)
            try:
                horizon = int(record.get("pred_len", config.get("pred_len")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid pred_len for {record.get('job_id')}") from exc
            features = str(record.get("features", config.get("features", "")))
            key = (dataset, features, seed, horizon)
            phase_key = (phase,) + key
            if phase_key in seen:
                raise ValueError(f"duplicate phase/dataset/features/seed/horizon row: {phase_key}")
            seen.add(phase_key)
            mse, mae = _read_scaled_metrics(result_dir)
            config_dataset = config.get("data", dataset)
            if str(config_dataset) != dataset:
                raise ValueError(
                    f"manifest/config dataset mismatch for {record.get('job_id')}: "
                    f"{dataset!r} vs {config_dataset!r}"
                )
            config_seed = config.get("run_seed", config.get("seed", seed))
            if int(config_seed) != seed:
                raise ValueError(
                    f"manifest/config seed mismatch for {record.get('job_id')}: "
                    f"{seed} vs {config_seed}"
                )
            rows.append(
                {
                    "phase": phase,
                    "dataset": dataset,
                    "features": features,
                    "seed": seed,
                    "horizon": horizon,
                    "mse": mse,
                    "mae": mae,
                    "model": str(record.get("model", config.get("model", ""))),
                    "autocon_lambda": record.get("autocon_lambda", config.get("AutoCon_lambda")),
                    "memory": _bool(config.get("mamba_use_acf_memory", False)),
                    "relation_loss": _bool(config.get("mamba_acf_relation_loss", False)),
                    "target_mode": str(config.get("mamba_acf_relation_target_mode", "")),
                    "relation_scope": str(config.get("mamba_acf_relation_scope", "")),
                    "relation_min_lag": config.get("mamba_acf_relation_min_lag"),
                    "result_dir": str(result_dir),
                    "job_id": str(record.get("job_id", result_dir.name)),
                }
            )

    expected = len(PHASE_ORDER) * 2 * 3
    if len(rows) != expected:
        raise ValueError(f"expected {expected} successful rows, found {len(rows)}")
    return rows, {"root": str(root), "phases": phase_meta}


def _mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    return mean, sd


def _fmt(value: Any, digits: int = 6) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _fmt_pm(mean: float, sd: float) -> str:
    if not math.isfinite(mean):
        return "n/a"
    if not math.isfinite(sd):
        return _fmt(mean)
    return f"{mean:.6f} +/- {sd:.6f}"


def _scipy_tests(deltas: Sequence[float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "t": math.nan,
        "t_p": math.nan,
        "w": math.nan,
        "w_p": math.nan,
        "backend": "scipy" if scipy_stats is not None else None,
    }
    if scipy_stats is None or len(deltas) < 2:
        return result
    if np is None:
        return result
    values = np.asarray(deltas, dtype=float)
    try:
        test = scipy_stats.ttest_1samp(values, 0.0, nan_policy="raise")
        result["t"] = float(test.statistic)
        result["t_p"] = float(test.pvalue)
    except (ValueError, FloatingPointError, RuntimeError):
        pass
    try:
        # ``wilcoxon`` raises for an all-zero vector; n/a is more honest than
        # manufacturing a p-value for a degenerate contrast.
        if not np.any(values != 0):
            return result
        test = scipy_stats.wilcoxon(
            values,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        result["w"] = float(test.statistic)
        result["w_p"] = float(test.pvalue)
    except (ValueError, FloatingPointError, RuntimeError):
        pass
    return result


def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, int, int]:
    return (
        str(row["dataset"]),
        str(row["features"]),
        int(row["seed"]),
        int(row["horizon"]),
    )


def paired_rows(
    rows: Sequence[Mapping[str, Any]], treatment: str, reference: str
) -> List[Dict[str, Any]]:
    treatment_rows = {_row_key(row): row for row in rows if row["phase"] == treatment}
    reference_rows = {_row_key(row): row for row in rows if row["phase"] == reference}
    keys = sorted(set(treatment_rows) & set(reference_rows))
    if set(treatment_rows) != set(reference_rows):
        missing_t = sorted(set(reference_rows) - set(treatment_rows))
        missing_r = sorted(set(treatment_rows) - set(reference_rows))
        raise ValueError(
            f"unmatched pairing {treatment} vs {reference}; "
            f"missing treatment={missing_t}, missing reference={missing_r}"
        )
    output: List[Dict[str, Any]] = []
    for key in keys:
        treatment_row = treatment_rows[key]
        reference_row = reference_rows[key]
        output.append(
            {
                "dataset": key[0],
                "features": key[1],
                "seed": key[2],
                "horizon": key[3],
                "treatment_mse": float(treatment_row["mse"]),
                "reference_mse": float(reference_row["mse"]),
                "delta_mse": float(treatment_row["mse"]) - float(reference_row["mse"]),
                "treatment_mae": float(treatment_row["mae"]),
                "reference_mae": float(reference_row["mae"]),
                "delta_mae": float(treatment_row["mae"]) - float(reference_row["mae"]),
            }
        )
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]], group: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["phase"]), str(row["dataset"] if group == "dataset" else "ALL"))].append(row)
    output: List[Dict[str, Any]] = []
    for (phase, dataset), values in sorted(
        groups.items(), key=lambda item: (PHASE_ORDER.index(item[0][0]), item[0][1])
    ):
        mse_mean, mse_sd = _mean_sd([float(row["mse"]) for row in values])
        mae_mean, mae_sd = _mean_sd([float(row["mae"]) for row in values])
        output.append(
            {
                "phase": phase,
                "dataset": dataset,
                "n": len(values),
                "mse_mean": mse_mean,
                "mse_sd": mse_sd,
                "mae_mean": mae_mean,
                "mae_sd": mae_sd,
            }
        )
    return output


def _contrast_summary(
    label: str, treatment: str, reference: str, pairs: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["dataset"])].append(pair)
    grouped["ALL"] = list(pairs)
    output: List[Dict[str, Any]] = []
    for dataset in sorted(grouped):
        values = grouped[dataset]
        mse = [float(pair["delta_mse"]) for pair in values]
        mae = [float(pair["delta_mae"]) for pair in values]
        mse_mean, mse_sd = _mean_sd(mse)
        mae_mean, mae_sd = _mean_sd(mae)
        mse_tests = _scipy_tests(mse)
        mae_tests = _scipy_tests(mae)
        output.append(
            {
                "label": label,
                "treatment": treatment,
                "reference": reference,
                "dataset": dataset,
                "n": len(values),
                "mse_mean": mse_mean,
                "mse_sd": mse_sd,
                "mae_mean": mae_mean,
                "mae_sd": mae_sd,
                "mse_win": sum(delta < 0 for delta in mse),
                "mae_win": sum(delta < 0 for delta in mae),
                "mse_tests": mse_tests,
                "mae_tests": mae_tests,
            }
        )
    return output


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def make_report(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> str:
    aggregates = _aggregate(rows, "dataset")
    pooled = _aggregate(rows, "all")
    all_aggregates = aggregates + pooled
    all_pairs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for label, treatment, reference in CONTRASTS:
        pairs = paired_rows(rows, treatment, reference)
        all_pairs.extend(
            [{"contrast": label, **pair} for pair in pairs]
        )
        summaries.extend(_contrast_summary(label, treatment, reference, pairs))

    validation_values = [
        value.get("validation", {})
        for value in meta.get("phases", {}).values()
        if isinstance(value, Mapping)
    ]
    plan_hashes = sorted(
        {
            str(value.get("plan_sha256"))
            for value in validation_values
            if value.get("plan_sha256")
        }
    )
    scipy_label = "available" if scipy_stats is not None else "unavailable"

    lines: List[str] = []
    lines.extend(
        [
            "# MambaACF v3 Relation Screen: Statistical Readout",
            "",
            "> Exploratory screen only. The report is generated from the five completed phases under `artifacts/paper_final/mamba_acf_v3_relation_screen`; it does not perform model or hyper-parameter selection.",
            "",
            "## Scope and interpretation",
            "",
            f"- Input root: `{meta['root']}`",
            f"- Rows: `{len(rows)}` successful runs (5 phases x 2 datasets x 3 seeds).",
            "- Metric source: `stream_metrics.json` scaled metrics (`scaled.mse`, `scaled.mae`); lower is better.",
            "- Pairing key: same dataset, feature setting, horizon, and seed. Delta is treatment minus reference; negative means lower error.",
            f"- Inference backend: SciPy {scipy_label}. Tests are two-sided paired one-sample t-tests and Wilcoxon signed-rank tests on seed-level deltas.",
            "- The inferential unit is the seed/run, not an individual forecast window.",
        ]
    )
    if plan_hashes:
        lines.append(f"- Validated plan SHA-256: `{', '.join(plan_hashes)}`")
    lines.extend(
        [
            "",
            "### n=3 limitation",
            "",
            "Each dataset-level contrast has only three seeds. The pooled row has six runs but combines two datasets and is not six independent replications of one data-generating process. Means, standard deviations, and p-values are descriptive screening evidence; they do not establish a stable gain, generalization, or publication-grade confirmation. No multiplicity correction is applied.",
            "",
            "## Per-phase metrics",
            "",
            "| Phase | Dataset | n | MSE mean +/- SD | MAE mean +/- SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in all_aggregates:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item["phase"]),
                    _md_cell(item["dataset"]),
                    str(item["n"]),
                    _fmt_pm(item["mse_mean"], item["mse_sd"]),
                    _fmt_pm(item["mae_mean"], item["mae_sd"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-seed metrics",
            "",
            "| Phase | Dataset | Seed | Horizon | MSE | MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            PHASE_ORDER.index(str(item["phase"])),
            str(item["dataset"]),
            int(item["seed"]),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row["phase"]),
                    _md_cell(row["dataset"]),
                    str(row["seed"]),
                    str(row["horizon"]),
                    _fmt(row["mse"]),
                    _fmt(row["mae"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired delta by seed",
            "",
            "| Contrast | Dataset | Seed | Delta MSE | Delta MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pair in all_pairs:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(pair["contrast"]),
                    _md_cell(pair["dataset"]),
                    str(pair["seed"]),
                    _fmt(pair["delta_mse"]),
                    _fmt(pair["delta_mae"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired summaries and tests",
            "",
            "Treatment-minus-reference deltas; `win` counts seed-level negative deltas (lower error). `t/p` is the paired one-sample t statistic/p-value; `W/p` is Wilcoxon signed-rank statistic/p-value.",
            "",
            "| Contrast | Dataset | n | Delta MSE mean +/- SD | MSE win | t/p (MSE) | W/p (MSE) | Delta MAE mean +/- SD | MAE win | t/p (MAE) | W/p (MAE) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        mse_t = item["mse_tests"]
        mae_t = item["mae_tests"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item["label"]),
                    _md_cell(item["dataset"]),
                    str(item["n"]),
                    _fmt_pm(item["mse_mean"], item["mse_sd"]),
                    f"{item['mse_win']}/{item['n']}",
                    f"{_fmt(mse_t['t'], 4)}/{_fmt(mse_t['t_p'], 4)}",
                    f"{_fmt(mse_t['w'], 4)}/{_fmt(mse_t['w_p'], 4)}",
                    _fmt_pm(item["mae_mean"], item["mae_sd"]),
                    f"{item['mae_win']}/{item['n']}",
                    f"{_fmt(mae_t['t'], 4)}/{_fmt(mae_t['t_p'], 4)}",
                    f"{_fmt(mae_t['w'], 4)}/{_fmt(mae_t['w_p'], 4)}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Registered contrast notes",
            "",
            "1. `available memory vs plain` isolates the causal ACF memory branch in MambaACF.",
            "2. `soft relation vs hard relation` compares target semantics while keeping the global, non-overlap relation control and the short training budget fixed.",
            "3. `memory + soft relation vs available memory` asks whether the relation objective adds beyond memory alone.",
            "",
            "A negative mean delta is directionally favorable, but this screen alone is insufficient to claim a reliable improvement. A future confirmation should pre-register datasets, seeds, training budget, selection rule, and an independent evaluation protocol before inspecting these deltas.",
            "",
            "## Phase validation",
            "",
            "| Phase | Manifest jobs | Validation |",
            "|---|---:|---:|",
        ]
    )
    for phase in PHASE_ORDER:
        item = meta["phases"][phase]
        validation = item.get("validation", {})
        lines.append(
            f"| `{phase}` | {item['jobs']} | {str(validation.get('valid', 'missing')).lower()} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/paper_final/mamba_acf_v3_relation_screen"),
        help="five-phase screen root (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write Markdown to this path; otherwise print to stdout",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        rows, meta = load_rows(args.root)
        report = make_report(rows, meta)
    except (OSError, ValueError, TypeError) as exc:
        print(f"analyze_mamba_acf_v3_screen: error: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        sys.stdout.write(report)
    else:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
