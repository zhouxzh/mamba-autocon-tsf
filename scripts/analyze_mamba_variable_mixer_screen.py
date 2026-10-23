#!/usr/bin/env python3
"""Read-only paired readout for the MambaACF variable-token screen.

The script binds one completed phase to its registered plan, validation file,
manifest, provenance record, and per-job artifact hashes.  It then reports the
variable-minus-plain delta at the seed level.  Forecast windows are never
treated as independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAIN_ARM = "MambaACF"
VARIABLE_ARM = "MambaACFVariable"
ARMS = (PLAIN_ARM, VARIABLE_ARM)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = (2021, 2022, 2023)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _command_arg(command: Sequence[str], name: str) -> str | None:
    try:
        index = list(command).index(name)
    except ValueError:
        return None
    return str(command[index + 1]) if index + 1 < len(command) else None


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None) -> dict[str, str]:
    paths = [result_dir / name for name in ARTIFACT_NAMES]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _normalised_command(command: Sequence[str]) -> list[str]:
    """Remove identity fields and the one intended arm switch for parity."""
    result: list[str] = []
    index = 0
    while index < len(command):
        token = str(command[index])
        if token in {"--model_id", "--plan_job_id", "--seed"}:
            index += 2
            continue
        if token == "--mamba_variable_mixer":
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarize an empty vector")
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers) if len(numbers) > 1 else None
    output: dict[str, Any] = {
        "n": len(numbers),
        "mean": mean,
        "sd": sd,
        "wins_lower": sum(value < 0.0 for value in numbers),
        "ties": sum(value == 0.0 for value in numbers),
    }
    try:
        from scipy import stats

        test = stats.ttest_1samp(numbers, 0.0)
        output["paired_one_sample_t"] = float(test.statistic)
        output["paired_one_sample_p"] = float(test.pvalue)
    except Exception:
        output["paired_one_sample_t"] = None
        output["paired_one_sample_p"] = None
    return output


def _phase_from_plan(plan: Mapping[str, Any], phase_name: str) -> Mapping[str, Any]:
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise ValueError("plan has no phases list")
    for phase in phases:
        if isinstance(phase, Mapping) and phase.get("name") == phase_name:
            return phase
    raise ValueError(f"phase not found in plan: {phase_name}")


def _load_rows(
    root: Path,
    plan_path: Path,
    phase_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.expanduser().resolve()
    plan_path = plan_path.expanduser().resolve()
    plan = _read_json(plan_path)
    phase = _phase_from_plan(plan, phase_name)
    registered_root = (Path(__file__).resolve().parents[1] / str(phase["output_root"])).resolve()
    if registered_root != root:
        raise ValueError(f"phase output_root mismatch: plan={registered_root}, input={root}")

    validation_path = root / "phase_validation.json"
    validation = _read_json(validation_path)
    if validation.get("valid") is not True:
        raise ValueError("phase validation is not valid")
    if validation.get("phase") != phase_name:
        raise ValueError("phase validation name mismatch")
    if validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError("plan changed after phase validation")

    manifest_path = root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    if validation.get("manifest") and Path(str(validation["manifest"])).resolve() != manifest_path.resolve():
        raise ValueError("phase validation points to a different manifest")
    if validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("manifest changed after phase validation")

    records: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(record)

    expected_jobs = int(validation.get("expected_jobs", -1))
    if len(records) != expected_jobs or int(validation.get("manifest_unique_jobs", -1)) != expected_jobs:
        raise ValueError(f"manifest coverage mismatch: {len(records)} rows, expected {expected_jobs}")
    if len({str(record.get("job_id")) for record in records}) != len(records):
        raise ValueError("manifest contains duplicate job IDs")

    result_root = (root / "results").resolve()
    expected_hashes = validation.get("artifact_hashes_by_job", {})
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("phase validation has no artifact hash map")
    rows: list[dict[str, Any]] = []
    signatures: dict[tuple[str, int], list[str]] = {}
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"non-successful job: {record.get('job_id')}")
        arm = str(record.get("model", ""))
        if arm not in ARMS:
            raise ValueError(f"unexpected arm: {arm}")
        job_id = str(record.get("job_id", ""))
        expected_artifact = record.get("expected_artifact")
        if not expected_artifact:
            raise ValueError(f"job has no expected artifact: {job_id}")
        metrics_path = Path(str(expected_artifact)).expanduser().resolve()
        result_dir = metrics_path.parent
        if not _inside(result_dir, result_root) or not metrics_path.is_file():
            raise ValueError(f"artifact is missing or outside results root: {job_id}")
        checkpoint_root = _command_arg(record.get("command", []), "--checkpoints")
        checkpoint_dir = None
        if checkpoint_root:
            checkpoint_dir = Path(checkpoint_root).expanduser().resolve() / result_dir.name
        actual_hashes = _artifact_hashes(result_dir, checkpoint_dir)
        if actual_hashes != expected_hashes.get(job_id):
            raise ValueError(f"artifact hash mismatch after validation: {job_id}")

        config = _read_json(result_dir / "config.json")
        stream = _read_json(result_dir / "stream_metrics.json")
        scaled = stream.get("scaled")
        if not isinstance(scaled, Mapping):
            raise ValueError(f"missing scaled metrics: {job_id}")
        dataset = str(record.get("dataset", ""))
        seed = int(record.get("seed"))
        horizon = int(record.get("pred_len"))
        features = str(record.get("features", ""))
        if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected coordinate: {dataset}/{seed}")
        if features != "M" or horizon != 96:
            raise ValueError(f"unexpected task contract for {job_id}")
        if config.get("model") != "MambaACF":
            raise ValueError(f"alias did not resolve to MambaACF: {job_id}")
        expected_variable = arm == VARIABLE_ARM
        if _bool(config.get("mamba_variable_mixer", False)) != expected_variable:
            raise ValueError(f"variable mixer flag mismatch: {job_id}")
        if int(config.get("mamba_variable_group_size", 1)) != 1:
            raise ValueError(f"variable group size mismatch: {job_id}")
        for key in ("mamba_use_acf_memory", "mamba_acf_relation_loss"):
            if _bool(config.get(key, False)):
                raise ValueError(f"screen contract unexpectedly enables {key}: {job_id}")
        if int(config.get("train_epochs", -1)) != int(phase.get("epochs", 8)):
            raise ValueError(f"epoch contract mismatch: {job_id}")

        signature_key = (dataset, seed)
        signatures.setdefault(signature_key, _normalised_command(record.get("command", [])))
        if signatures[signature_key] != _normalised_command(record.get("command", [])):
            raise ValueError(f"plain/variable command parity mismatch: {dataset}/seed={seed}")
        rows.append({
            "dataset": dataset,
            "features": features,
            "horizon": horizon,
            "seed": seed,
            "arm": arm,
            "mse": _finite(scaled.get("mse"), f"{job_id} scaled.mse"),
            "mae": _finite(scaled.get("mae"), f"{job_id} scaled.mae"),
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id} elapsed_sec"),
            "job_id": job_id,
            "config_sha256": _sha256(result_dir / "config.json"),
            "metrics_sha256": _sha256(metrics_path),
        })

    expected_coordinates = {
        (dataset, seed, arm)
        for dataset in EXPECTED_DATASETS
        for seed in EXPECTED_SEEDS
        for arm in ARMS
    }
    observed_coordinates = {(row["dataset"], row["seed"], row["arm"]) for row in rows}
    if observed_coordinates != expected_coordinates:
        raise ValueError("manifest coordinates do not match the registered 12-job matrix")

    provenance_path = root / "provenance.json"
    provenance = _read_json(provenance_path)
    if provenance.get("provenance_id") != validation.get("provenance_id"):
        raise ValueError("provenance ID mismatch")
    before = {str(record.get("source_tree_sha256_before")) for record in records}
    after = {str(record.get("source_tree_sha256_after")) for record in records}
    if len(before) != 1 or len(after) != 1 or before != after:
        raise ValueError("job source hashes are not stable across the phase")
    if before != {str(provenance.get("source_tree_sha256"))}:
        raise ValueError("job source hash differs from provenance")

    meta = {
        "phase": phase_name,
        "root": str(root),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "validation": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "provenance": str(provenance_path),
        "provenance_id": provenance.get("provenance_id"),
        "source_tree_sha256": provenance.get("source_tree_sha256"),
        "expected_jobs": expected_jobs,
        "observed_jobs": len(rows),
        "validation_valid": True,
    }
    return rows, meta


def _paired(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["dataset"], int(row["seed"]), row["arm"]): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            plain = indexed[(dataset, seed, PLAIN_ARM)]
            variable = indexed[(dataset, seed, VARIABLE_ARM)]
            pairs.append({
                "dataset": dataset,
                "features": plain["features"],
                "horizon": int(plain["horizon"]),
                "seed": seed,
                "plain_mse": float(plain["mse"]),
                "variable_mse": float(variable["mse"]),
                "delta_mse": float(variable["mse"] - plain["mse"]),
                "plain_mae": float(plain["mae"]),
                "variable_mae": float(variable["mae"]),
                "delta_mae": float(variable["mae"] - plain["mae"]),
                "plain_elapsed_sec": float(plain["elapsed_sec"]),
                "variable_elapsed_sec": float(variable["elapsed_sec"]),
                "plain_job_id": plain["job_id"],
                "variable_job_id": variable["job_id"],
            })
    return pairs


def _contrast_summary(label: str, pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mse_delta = [float(pair["delta_mse"]) for pair in pairs]
    mae_delta = [float(pair["delta_mae"]) for pair in pairs]
    plain_mse = [float(pair["plain_mse"]) for pair in pairs]
    variable_mse = [float(pair["variable_mse"]) for pair in pairs]
    plain_mae = [float(pair["plain_mae"]) for pair in pairs]
    variable_mae = [float(pair["variable_mae"]) for pair in pairs]
    plain_time = [float(pair["plain_elapsed_sec"]) for pair in pairs]
    variable_time = [float(pair["variable_elapsed_sec"]) for pair in pairs]
    mse_stats = _summary(mse_delta)
    mae_stats = _summary(mae_delta)
    return {
        "label": label,
        "n": len(pairs),
        "plain_mse_mean": statistics.fmean(plain_mse),
        "variable_mse_mean": statistics.fmean(variable_mse),
        "delta_mse": mse_stats,
        "relative_mse_change_pct": 100.0 * mse_stats["mean"] / statistics.fmean(plain_mse),
        "plain_mae_mean": statistics.fmean(plain_mae),
        "variable_mae_mean": statistics.fmean(variable_mae),
        "delta_mae": mae_stats,
        "relative_mae_change_pct": 100.0 * mae_stats["mean"] / statistics.fmean(plain_mae),
        "plain_elapsed_mean_sec": statistics.fmean(plain_time),
        "variable_elapsed_mean_sec": statistics.fmean(variable_time),
        "runtime_overhead_pct": 100.0 * (statistics.fmean(variable_time) / statistics.fmean(plain_time) - 1.0),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_pm(item: Mapping[str, Any]) -> str:
    sd = item.get("sd")
    return f"{_fmt(item.get('mean'))} +/- {_fmt(sd) if sd is not None else 'n/a'}"


def _markdown(meta: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# MambaACF Variable-Mixer Screen Readout",
        "",
        "> Exploratory paired screen only. Each dataset has n=3 seeds; this readout is not confirmatory evidence and must not be merged into the paper's main comparison table.",
        "",
        "## Contract and provenance",
        "",
        f"- Phase: `{meta['phase']}`; validated jobs: `{meta['observed_jobs']}/{meta['expected_jobs']}`.",
        f"- Pairing: same dataset, feature setting, horizon, and seed; delta is `MambaACFVariable - MambaACF`.",
        f"- Metric: scaled test MSE/MAE from `stream_metrics.json`; forecast windows are not inferential replicates.",
        f"- Validation: `valid=true`; provenance ID `{meta['provenance_id']}`.",
        f"- Plan SHA-256: `{meta['plan_sha256']}`; manifest SHA-256: `{meta['manifest_sha256']}`.",
        f"- Frozen source-tree SHA-256: `{meta['source_tree_sha256']}`; all 12 jobs had matching before/after hashes.",
        "",
        "## Per-seed paired metrics",
        "",
        "| Dataset | Seed | Plain MSE | Variable MSE | Delta MSE | Plain MAE | Variable MAE | Delta MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            "| " + " | ".join([
                str(pair["dataset"]), str(pair["seed"]), _fmt(pair["plain_mse"]),
                _fmt(pair["variable_mse"]), _fmt(pair["delta_mse"]),
                _fmt(pair["plain_mae"]), _fmt(pair["variable_mae"]), _fmt(pair["delta_mae"]),
            ]) + " |"
        )
    lines.extend([
        "",
        "## Paired summaries",
        "",
        "Negative delta is favorable. SD is the sample standard deviation across seed-level deltas.",
        "",
        "| Contrast | n | Plain MSE mean | Variable MSE mean | Delta MSE mean +/- SD | MSE wins | MSE relative change | Plain MAE mean | Variable MAE mean | Delta MAE mean +/- SD | MAE wins | MAE relative change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for summary in summaries:
        mse = summary["delta_mse"]
        mae = summary["delta_mae"]
        lines.append(
            "| " + " | ".join([
                str(summary["label"]), str(summary["n"]), _fmt(summary["plain_mse_mean"]),
                _fmt(summary["variable_mse_mean"]), _fmt_pm(mse),
                f"{mse['wins_lower']}/{summary['n']}", _fmt(summary["relative_mse_change_pct"], 3) + "%",
                _fmt(summary["plain_mae_mean"]), _fmt(summary["variable_mae_mean"]), _fmt_pm(mae),
                f"{mae['wins_lower']}/{summary['n']}", _fmt(summary["relative_mae_change_pct"], 3) + "%",
            ]) + " |"
        )
    lines.extend([
        "",
        "## Runtime observation",
        "",
        "| Contrast | Plain mean seconds | Variable mean seconds | Overhead |",
        "|---|---:|---:|---:|",
    ])
    for summary in summaries[:2]:
        lines.append(
            f"| {summary['label']} | {_fmt(summary['plain_elapsed_mean_sec'], 3)} | "
            f"{_fmt(summary['variable_elapsed_mean_sec'], 3)} | {_fmt(summary['runtime_overhead_pct'], 2)}% |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- ETTh1 and Traffic each have only three paired seeds. Means and SDs describe this screen; they do not establish a stable gain or generalization.",
        "- The pooled six-pair row is a descriptive cross-dataset summary, not six independent replications of one data-generating process.",
        "- The variable arm adds a shared cross-variable token branch to the same executable `MambaACF`; it does not change the ACF memory or relation-loss settings in this screen.",
        "- A future confirmatory run must freeze the arm from validation evidence, pre-register the matrix, and replicate it under an independent evaluation protocol.",
        "",
    ])
    return "\n".join(lines)


def build_readout(root: Path, plan_path: Path, phase_name: str) -> dict[str, Any]:
    rows, meta = _load_rows(root, plan_path, phase_name)
    pairs = _paired(rows)
    summaries = [
        _contrast_summary("ETTh1", [pair for pair in pairs if pair["dataset"] == "ETTh1"]),
        _contrast_summary("traffic", [pair for pair in pairs if pair["dataset"] == "traffic"]),
        _contrast_summary("pooled descriptive (not independent)", pairs),
    ]
    return {
        "schema_version": 1,
        "status": "exploratory_not_confirmatory",
        "protocol": {
            "phase": phase_name,
            "arms": [PLAIN_ARM, VARIABLE_ARM],
            "executable_model": "MambaACF",
            "datasets": list(EXPECTED_DATASETS),
            "seeds": list(EXPECTED_SEEDS),
            "features": "M",
            "horizon": 96,
            "delta_definition": "variable minus plain",
            "inferential_unit": "seed/run",
        },
        "provenance": meta,
        "per_seed": pairs,
        "summaries": summaries,
        "limitations": [
            "n=3 paired seeds per dataset; descriptive screening only",
            "pooled six-pair row is not six independent replications",
            "forecast windows are not treated as independent observations",
            "no test-based architecture selection or confirmatory publication claim",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="completed phase output root")
    parser.add_argument("--plan", type=Path, required=True, help="registered plan JSON")
    parser.add_argument("--phase", required=True, help="registered phase name")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_readout(args.root, args.plan, args.phase)
        output_json = args.output_json.expanduser()
        output_md = args.output_md.expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_md.write_text(
            _markdown(payload["provenance"], payload["per_seed"], payload["summaries"]),
            encoding="utf-8",
        )
        print(f"wrote {output_json}")
        print(f"wrote {output_md}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"analyze_mamba_variable_mixer_screen: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
