#!/usr/bin/env python3
"""Strict, read-only readout for the MambaACF direct-anchor screen.

The registered screen has two paired arms: the same variable-token
``MambaACFVariableAutoCon`` model with ``mamba_acf_anchor_copy`` disabled or
enabled.  This reader never trains or evaluates a model.  It validates the
frozen plan and, when a complete artifact matrix exists, checks the manifest,
sidecar, configuration, checkpoint, and anchor diagnostics before writing a
descriptive paired readout.  Missing artifacts return ``NOT_RUN`` and create
no output files.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional for read-only reporting.
    scipy_stats = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.run_training_plan import _job_specs, _validate_plan
except ImportError:  # pragma: no cover - direct script execution.
    from run_training_plan import _job_specs, _validate_plan  # type: ignore

from utils.acf_relation_contract import resolve_acf_source


EXPECTED_PHASES: Tuple[Tuple[str, bool], ...] = (
    ("anchor_copy_off_h96", False),
    ("anchor_copy_on_h96", True),
)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)
EXPECTED_HORIZON = 96
EXPECTED_ALIAS = "MambaACFVariableAutoCon"
EXPECTED_EXECUTABLE = "MambaACFAutoCon"
EXPECTED_LAMBDA = 0.03
EXPECTED_JOBS_PER_PHASE = len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS)


class NotRunError(RuntimeError):
    """Raised when the registered matrix has no complete training outputs."""


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("JSON object expected: {}".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off"}:
            return False
    raise ValueError("{} is not an explicit boolean: {!r}".format(label, value))


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} is not numeric".format(label)) from exc
    if not math.isfinite(result):
        raise ValueError("{} is non-finite".format(label))
    return result


def _phase_root(plan: Mapping[str, Any], name: str) -> Path:
    for phase in plan.get("phases", []):
        if isinstance(phase, Mapping) and phase.get("name") == name:
            return _resolve(Path(str(phase["output_root"])))
    raise KeyError("phase not found: {}".format(name))


def _plan_errors(plan: Mapping[str, Any]) -> List[str]:
    errors = [str(item) for item in _validate_plan(dict(plan))]
    if plan.get("status") != "exploratory_mamba_acf_anchor_copy_screen_draft":
        errors.append("unexpected anchor-copy plan status")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution metadata is missing")
    else:
        if execution.get("authorized") is not False:
            errors.append("draft plan must have execution.authorized=false")
        if execution.get("training_started") is not False:
            errors.append("draft plan must have execution.training_started=false")
        if execution.get("status") != plan.get("status"):
            errors.append("execution status disagrees with plan status")
    phases = plan.get("phases")
    if not isinstance(phases, list) or [p.get("name") for p in phases] != [p[0] for p in EXPECTED_PHASES]:
        errors.append("anchor-copy phases/order are not the registered two-arm matrix")
    else:
        for phase, (_, expected_anchor) in zip(phases, EXPECTED_PHASES):
            if phase.get("models") != [EXPECTED_ALIAS]:
                errors.append("{} must use {}".format(phase.get("name"), EXPECTED_ALIAS))
            if phase.get("datasets") != list(EXPECTED_DATASETS) or phase.get("horizons") != [EXPECTED_HORIZON]:
                errors.append("{} dataset/horizon contract mismatch".format(phase.get("name")))
            if phase.get("seeds") != list(EXPECTED_SEEDS):
                errors.append("{} seed contract mismatch".format(phase.get("name")))
            try:
                actual_anchor = _bool(phase.get("mamba_acf_anchor_copy"), "phase anchor flag")
                if actual_anchor != expected_anchor:
                    errors.append("{} anchor flag mismatch".format(phase.get("name")))
            except ValueError as exc:
                errors.append(str(exc))
            for key, expected in {
                "features": "M",
                "autocon_lambda": EXPECTED_LAMBDA,
                "mamba_acf_memory_gate_mode": "hidden",
                "mamba_variable_mixer": True,
                "mamba_variable_group_size": 1,
                "mamba_use_acf_memory": True,
                "mamba_dynamic_acf": False,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_scope": "global",
                "mamba_acf_relation_min_lag": 96,
                "epochs": 8,
                "patience": 3,
                "stream_metrics": True,
            }.items():
                actual = phase.get(key)
                if isinstance(expected, bool):
                    try:
                        actual = _bool(actual, key)
                    except ValueError:
                        pass
                elif isinstance(expected, float):
                    try:
                        actual = float(actual)
                    except (TypeError, ValueError):
                        pass
                if actual != expected:
                    errors.append("{} {} mismatch".format(phase.get("name"), key))
    counts = plan.get("registered_job_counts")
    if not isinstance(counts, Mapping):
        errors.append("registered_job_counts is missing")
    else:
        expected_counts = {
            "anchor_copy_off_h96": 6,
            "anchor_copy_on_h96": 6,
            "exploratory_total": 12,
            "training_total": 12,
            "total": 12,
            "training_started": False,
            "resource_runs_not_in_training_matrix": True,
        }
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                errors.append("registered_job_counts[{}] mismatch".format(key))
    resource = plan.get("resource_contract")
    if not isinstance(resource, Mapping):
        errors.append("resource_contract is missing")
    else:
        if resource.get("anchor_parameters_off") != 0 or resource.get("anchor_parameters_on") != 1:
            errors.append("anchor parameter contract mismatch")
        if resource.get("output_shape") != "B,H,C" or resource.get("dense_candidate_tensor") is not False:
            errors.append("anchor output/materialization contract mismatch")
    return errors


def validate_plan_contract(plan: Mapping[str, Any]) -> None:
    errors = _plan_errors(plan)
    if errors:
        raise ValueError("anchor-copy plan validation failed: " + "; ".join(errors))


def _missing_artifacts(plan: Mapping[str, Any]) -> List[str]:
    missing: List[str] = []
    for name, _ in EXPECTED_PHASES:
        root = _phase_root(plan, name)
        for relative in ("phase_validation.json", "logs/job_manifest.jsonl"):
            path = root / relative
            if not path.is_file():
                missing.append(str(path))
        validation = root / "phase_validation.json"
        if validation.is_file():
            try:
                if _read_json(validation).get("valid") is not True:
                    missing.append(str(validation) + " (valid=false)")
            except (OSError, ValueError, TypeError):
                missing.append(str(validation) + " (unreadable)")
    return missing


def _manifest(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid manifest line {}: {}".format(line_number, path)) from exc
            if not isinstance(value, Mapping):
                raise ValueError("manifest row is not an object: {}".format(path))
            rows.append(value)
    return rows


def _summary(logs: Mapping[str, Any], key: str, path: Path, required: bool = False) -> Dict[str, Any]:
    if key not in logs:
        if required:
            raise ValueError("loss log is missing {}: {}".format(key, path))
        return {"last": None, "mean": None, "count": 0}
    raw = logs[key]
    values = raw if isinstance(raw, list) else [raw]
    numbers = [_finite(item, "{} in {}".format(key, path)) for item in values]
    if not numbers:
        raise ValueError("loss log {} is empty: {}".format(key, path))
    return {"last": numbers[-1], "mean": statistics.fmean(numbers), "count": len(numbers)}


def _checkpoint_numel(path: Path) -> Tuple[int, bool]:
    try:
        import torch

        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover
            value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError("cannot load checkpoint {}: {}".format(path, exc)) from exc
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint is not a mapping: {}".format(path))
    state = value.get("model", value)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model state is not a mapping: {}".format(path))
    total = 0
    has_anchor = False
    for key, tensor in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[7:]
        if key == "acf_anchor_scale":
            has_anchor = True
        numel = getattr(tensor, "numel", None)
        if callable(numel):
            total += int(numel())
    return total, has_anchor


def _load_sidecar(path: Path, config: Mapping[str, Any], job_id: str) -> str:
    try:
        import numpy as np

        with np.load(str(path), allow_pickle=False) as payload:
            lags = np.asarray(payload["lags"]).reshape(-1)
            source = str(np.asarray(payload["acf_source"]).item()).lower()
            values = np.asarray(payload["acf_values"])
            full = np.asarray(payload["acf_full_values"])
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError("invalid ACF sidecar for {}: {}".format(job_id, path)) from exc
    if source != "raw" or lags.dtype.kind not in "iu" or values.ndim != 2 or full.ndim != 2:
        raise ValueError("sidecar source/shape contract failed for {}".format(job_id))
    expected = [int(item) for item in config.get("mamba_acf_prior_lags", [])]
    if [int(item) for item in lags.tolist()] != expected:
        raise ValueError("sidecar lag contract failed for {}".format(job_id))
    if not np.isfinite(values).all() or not np.isfinite(full).all():
        raise ValueError("sidecar contains non-finite values for {}".format(job_id))
    if full.shape[0] != values.shape[0] or full.shape[1] <= max(expected, default=0):
        raise ValueError("sidecar complete-table contract failed for {}".format(job_id))
    return _sha256(path)


def _artifact_path(record: Mapping[str, Any], name: str, result_dir: Path, checkpoint_dir: Path) -> Path:
    if name in {"checkpoint.pth", "loss_logs.json"}:
        return checkpoint_dir / name
    return result_dir / name


def _load_phase(plan: Mapping[str, Any], phase_name: str, anchor_enabled: bool) -> List[Dict[str, Any]]:
    root = _phase_root(plan, phase_name)
    validation_path = root / "phase_validation.json"
    manifest_path = root / "logs" / "job_manifest.jsonl"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True:
        raise ValueError("phase validation is not valid: {}".format(phase_name))
    records = _manifest(manifest_path)
    expected = {(dataset, seed) for dataset in EXPECTED_DATASETS for seed in EXPECTED_SEEDS}
    observed = set()
    rows: List[Dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            raise ValueError("manifest record is not successful: {}".format(record.get("job_id")))
        dataset = str(record.get("dataset"))
        seed = int(record.get("seed"))
        if (dataset, seed) not in expected or (dataset, seed) in observed:
            raise ValueError("unexpected or duplicate coordinate in {}".format(phase_name))
        observed.add((dataset, seed))
        job_id = str(record.get("job_id"))
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError("manifest expected_artifact missing: {}".format(job_id))
        result_dir = Path(str(expected_artifact)).resolve().parent
        config_path = result_dir / "config.json"
        stream_path = result_dir / "stream_metrics.json"
        if not config_path.is_file() or not stream_path.is_file():
            raise NotRunError("missing result files for {}".format(job_id))
        config = _read_json(config_path)
        if config.get("model") != EXPECTED_EXECUTABLE or config.get("data") != dataset:
            raise ValueError("config model/dataset mismatch: {}".format(job_id))
        if _bool(config.get("mamba_acf_anchor_copy"), "config anchor flag") != anchor_enabled:
            raise ValueError("config anchor flag mismatch: {}".format(job_id))
        if resolve_acf_source(config, default="") != "raw":
            raise ValueError("config source is not raw: {}".format(job_id))
        if _bool(config.get("mamba_dynamic_acf", False), "dynamic ACF"):
            raise ValueError("dynamic ACF is enabled: {}".format(job_id))
        if config.get("mamba_acf_memory_gate_mode", "hidden") != "hidden":
            raise ValueError("gate mode mismatch: {}".format(job_id))
        stream = _read_json(stream_path)
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError("scaled stream metrics missing: {}".format(job_id))
        checkpoint_dir = Path(str(config.get("checkpoints", root / "checkpoints"))).resolve() / result_dir.name
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        logs_path = checkpoint_dir / "loss_logs.json"
        if not checkpoint_path.is_file() or not logs_path.is_file():
            raise NotRunError("missing checkpoint/loss log for {}".format(job_id))
        logs = _read_json(logs_path)
        sidecar = result_dir / "acf_prior.npz"
        if not sidecar.is_file():
            raise NotRunError("missing ACF sidecar for {}".format(job_id))
        sidecar_sha = _load_sidecar(sidecar, config, job_id)
        declared_sha = str(config.get("mamba_acf_prior_sha256", "")).lower()
        if declared_sha != sidecar_sha:
            raise ValueError("config/sidecar hash mismatch: {}".format(job_id))
        artifact_hashes = record.get("artifact_hashes", {})
        if isinstance(artifact_hashes, Mapping):
            for name, declared in artifact_hashes.items():
                path = _artifact_path(record, str(name), result_dir, checkpoint_dir)
                if path.is_file() and str(declared).lower() != _sha256(path):
                    raise ValueError("artifact hash mismatch: {} {}".format(job_id, name))
        parameter_count, checkpoint_has_anchor = _checkpoint_numel(checkpoint_path)
        if checkpoint_has_anchor != anchor_enabled:
            raise ValueError("checkpoint anchor parameter mismatch: {}".format(job_id))
        anchor_keys = ("acf_anchor_scale", "acf_anchor_rms", "acf_anchor_valid_fraction", "acf_anchor_strength_mean", "acf_anchor_lag_mean")
        diagnostics = {key: _summary(logs, key, logs_path, required=anchor_enabled) for key in anchor_keys}
        rows.append({
            "arm": "anchor_on" if anchor_enabled else "anchor_off",
            "dataset": dataset,
            "seed": seed,
            "horizon": int(record.get("pred_len", EXPECTED_HORIZON)),
            "mse": _finite(scaled.get("mse"), "scaled mse {}".format(job_id)),
            "mae": _finite(scaled.get("mae"), "scaled mae {}".format(job_id)),
            "elapsed_sec": _finite(record.get("elapsed_sec"), "elapsed {}".format(job_id)),
            "parameter_count": parameter_count,
            "anchor_diagnostics": diagnostics,
            "sidecar_sha256": sidecar_sha,
            "job_id": job_id,
        })
    if observed != expected or len(rows) != EXPECTED_JOBS_PER_PHASE:
        raise ValueError("incomplete coordinate matrix: {}".format(phase_name))
    return rows


def _paired_summary(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(item) for item in values]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else math.nan
    p_value = None
    if scipy_stats is not None and len(values) > 1:
        p_value = float(scipy_stats.ttest_1samp(values, 0.0).pvalue)
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(item < 0 for item in values),
        "ties": sum(item == 0 for item in values),
        "two_sided_p": p_value,
        "delta_values": values,
    }


def _contrast(pairs: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    deltas = [float(row["on_" + metric] - row["off_" + metric]) for row in pairs]
    result = _paired_summary(deltas)
    reference = statistics.fmean(float(row["off_" + metric]) for row in pairs)
    result.update({
        "reference_mean": reference,
        "treatment_mean": statistics.fmean(float(row["on_" + metric]) for row in pairs),
        "relative_delta_mean": deltas and statistics.fmean(deltas) / reference if reference else math.nan,
    })
    return result


def _make_pairs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    index = {
        (row["dataset"], int(row["seed"]), row["arm"]): row
        for row in rows
    }
    pairs: List[Dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            off = index.get((dataset, seed, "anchor_off"))
            on = index.get((dataset, seed, "anchor_on"))
            if off is None or on is None:
                raise ValueError("missing pair: {} {}".format(dataset, seed))
            if off["sidecar_sha256"] != on["sidecar_sha256"]:
                raise ValueError("sidecar mismatch: {} {}".format(dataset, seed))
            item = {"dataset": dataset, "seed": seed, "sidecar_sha256": off["sidecar_sha256"]}
            for metric in ("mse", "mae", "elapsed_sec", "parameter_count"):
                item["off_" + metric] = off[metric]
                item["on_" + metric] = on[metric]
                item["delta_" + metric] = on[metric] - off[metric]
            item["anchor_diagnostics"] = {"off": off["anchor_diagnostics"], "on": on["anchor_diagnostics"]}
            pairs.append(item)
    return pairs


def build_readout(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = _make_pairs(rows)
    contrasts = {metric: _contrast(pairs, metric) for metric in ("mse", "mae", "elapsed_sec", "parameter_count")}
    by_dataset = {}
    for dataset in EXPECTED_DATASETS:
        subset = [item for item in pairs if item["dataset"] == dataset]
        by_dataset[dataset] = {metric: _contrast(subset, metric) for metric in ("mse", "mae", "elapsed_sec")}
    mse = contrasts["mse"]
    favorable_by_dataset = {
        dataset: sum(item["delta_mse"] < 0 for item in pairs if item["dataset"] == dataset)
        for dataset in EXPECTED_DATASETS
    }
    relative_by_dataset = {
        dataset: by_dataset[dataset]["mse"]["relative_delta_mean"] for dataset in EXPECTED_DATASETS
    }
    meets = (
        mse["wins_lower"] >= 4
        and all(value >= 2 for value in favorable_by_dataset.values())
        and all(value <= 0.01 for value in relative_by_dataset.values())
    )
    return {
        "schema_version": 1,
        "status": "exploratory_mamba_acf_anchor_copy_screen_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan": str(plan.get("_path", "")),
        "plan_sha256": _sha256(Path(str(plan["_path"]))) if plan.get("_path") else None,
        "rows": [dict(row) for row in rows],
        "pairs": pairs,
        "contrasts": contrasts,
        "by_dataset": by_dataset,
        "stopping_decision": {
            "criterion": "anchor-on favorable on >=4/6 paired MSE coordinates, >=2/3 per dataset, and no dataset degradation >1% (descriptive only)",
            "favorable_coordinates": mse["wins_lower"],
            "favorable_coordinates_by_dataset": favorable_by_dataset,
            "mean_relative_delta_by_dataset": relative_by_dataset,
            "meets_descriptive_screen": meets,
            "action": "continue only to a separately reviewed frozen replication" if meets else "stop exploratory screen; do not add arms, seeds, horizons, or tuning",
        },
        "resource_contract": plan.get("resource_contract"),
        "evidence_boundary": "Descriptive exploratory direct-anchor evidence only; no pooled cross-plan inference or publication claim.",
    }


def markdown(result: Mapping[str, Any]) -> str:
    mse = result["contrasts"]["mse"]
    lines = [
        "# MambaACF Direct-Anchor Screen",
        "",
        "> Exploratory, non-confirmatory readout. The two arms share the variable-token MambaACF AutoCon trunk, raw train-only ACF sidecar, relation objective, data, seeds, and budget; only the independent direct-anchor residual changes.",
        "",
        "## Paired metrics",
        "",
        "| Dataset | Seed | No anchor MSE | Anchor MSE | Delta | No anchor MAE | Anchor MAE | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["pairs"]:
        lines.append("| {dataset} | {seed} | {off_mse:.6f} | {on_mse:.6f} | {delta_mse:.6f} | {off_mae:.6f} | {on_mae:.6f} | {delta_mae:.6f} |".format(**row))
    lines += [
        "",
        "## Summary",
        "",
        "- Pooled MSE delta (anchor minus no-anchor): `{:.6f}` ({:.3f}%), wins `{}/{}`; p-value is descriptive.".format(mse["mean"], 100 * mse["relative_delta_mean"], mse["wins_lower"], mse["n"]),
        "- Screening action: **{}**.".format(result["stopping_decision"]["action"]),
        "- Anchor diagnostics are retained in the JSON; a nonzero scale and valid-anchor fraction are implementation checks, not evidence of accuracy benefit.",
        "- This screen cannot establish a general Mamba, ACF, speed, or publication claim.",
        "",
    ]
    return "\n".join(lines)


def _not_run_message(plan: Mapping[str, Any]) -> str:
    missing = _missing_artifacts(plan)
    if not missing:
        return "NOT_RUN: no complete training artifacts found for the registered anchor-copy screen"
    suffix = " ..." if len(missing) > 6 else ""
    return "NOT_RUN: registered anchor-copy screen has no complete training artifacts ({}{})".format("; ".join(missing[:6]), suffix)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/mamba_acf_anchor_copy_screen_plan_20260902.json"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", "--output-markdown", dest="output_md", type=Path, default=None)
    args = parser.parse_args(argv)
    plan_path = _resolve(args.plan)
    try:
        plan = dict(_read_json(plan_path))
        plan["_path"] = str(plan_path)
        validate_plan_contract(plan)
        missing = _missing_artifacts(plan)
        if missing:
            print(_not_run_message(plan), file=sys.stderr)
            return 3
        rows: List[Dict[str, Any]] = []
        for phase_name, anchor_enabled in EXPECTED_PHASES:
            rows.extend(_load_phase(plan, phase_name, anchor_enabled))
        result = build_readout(plan, rows)
        output_json = _resolve(args.output_json or Path("artifacts/paper_final/mamba_acf_anchor_copy_screen_20260902/readout.json"))
        output_md = _resolve(args.output_md or Path("docs/research/mamba_acf_anchor_copy_screen_readout_20260902.md"))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        output_md.write_text(markdown(result), encoding="utf-8")
        print(json.dumps({"rows": len(rows), "paired_coordinates": len(result["pairs"]), "output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=True))
        return 0
    except NotRunError as exc:
        print("NOT_RUN: {}".format(exc), file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("analyze_mamba_acf_anchor_copy_screen: error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
