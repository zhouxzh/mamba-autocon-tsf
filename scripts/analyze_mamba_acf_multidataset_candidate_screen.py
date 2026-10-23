#!/usr/bin/env python3
"""Audit and summarize the four-arm multidataset MambaACF candidate screen.

The screen is validation-only.  This reader requires complete phase
validation, exact dataset/horizon/seed coordinates, matching source hashes,
train-only ACF sidecars, and absence of test artifacts.  It reads only the
minimum pure-forecast ``val_loss`` and uses AutoCon/relation telemetry as a
mechanism audit, never as a replacement metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import _artifact_exists, _job_specs, _load_json, _validate_plan


EXPECTED_PHASES = (
    "acf_off_h96",
    "acf_memory_h96",
    "autocon_memory_relation_lambda0_h96",
    "autocon_memory_relation_lambda001_h96",
)
DATASETS = ("ETTh1", "ETTm1", "electricity", "traffic")
SEEDS = (2021, 2022, 2023)
HORIZON = 96


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def command_value(command: list, flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError, TypeError):
        return None


def finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def manifest(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest line {number}: {path}") from exc
            if not isinstance(item, dict) or not item.get("job_id"):
                raise ValueError(f"invalid manifest line {number}: {path}")
            job_id = str(item["job_id"])
            if job_id in records:
                raise ValueError(f"duplicate job id: {job_id}")
            records[job_id] = item
    return records


def trajectory(checkpoint_dir: Path) -> tuple[float, int, int, dict]:
    logs = read_json(checkpoint_dir / "loss_logs.json")
    values = logs.get("val_loss")
    if not isinstance(values, list) or not values:
        raise ValueError(f"missing val_loss: {checkpoint_dir}")
    numeric = [finite(value, f"{checkpoint_dir}:val_loss") for value in values]
    best = min(numeric)
    best_epoch = int(logs.get("best_epoch", numeric.index(best) + 1))
    epochs_ran = int(logs.get("epochs_ran", len(numeric)))
    if not 1 <= best_epoch <= epochs_ran:
        raise ValueError(f"invalid epoch metadata: {checkpoint_dir}")
    return best, best_epoch, epochs_ran, logs


def _telemetry_mean(logs: dict, key: str) -> float | None:
    values = logs.get(key)
    if values is None:
        return None
    if not isinstance(values, list):
        values = [values]
    finite_values = [finite(value, key) for value in values if value is not None]
    return float(statistics.mean(finite_values)) if finite_values else None


def _sidecar_check(config: dict, result_dir: Path) -> None:
    artifact_name = str(config.get("mamba_acf_prior_artifact", ""))
    expected_hash = str(config.get("mamba_acf_prior_sha256", ""))
    if not artifact_name or not expected_hash:
        raise ValueError(f"missing ACF sidecar identity: {result_dir}")
    sidecar = result_dir / artifact_name
    if not sidecar.is_file() or sha256(sidecar) != expected_hash:
        raise ValueError(f"ACF sidecar hash mismatch: {sidecar}")


def _test_artifact_check(result_dir: Path, marker: Path) -> None:
    marker_payload = read_json(marker)
    if marker_payload.get("status") != "trained_without_test":
        raise ValueError(f"test was not withheld: {marker}")
    forbidden = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "stream_samples.npz")
    present = [name for name in forbidden if (result_dir / name).exists()]
    if present:
        raise ValueError(f"test artifacts present at {result_dir}: {present}")


def load_rows(plan_path: Path, artifact_root: Path) -> list[dict]:
    plan = read_json(plan_path)
    errors = _validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    phases = {str(phase["name"]): phase for phase in plan.get("phases", [])}
    if tuple(phases) != EXPECTED_PHASES:
        raise ValueError(f"unexpected phase order/names: {tuple(phases)}")
    expected_by_phase = {
        name: _job_specs(plan, phase, {}, allow_unresolved=False)
        for name, phase in phases.items()
    }
    source_hashes = set()
    rows = []
    for phase_name in EXPECTED_PHASES:
        phase_root = artifact_root / phase_name
        validation = read_json(phase_root / "phase_validation.json")
        if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != 12:
            raise ValueError(f"phase validation failed: {phase_name}")
        source_hashes.add(str(validation.get("source_tree_sha256", "")))
        records = manifest(phase_root / "logs" / "job_manifest.jsonl")
        if len(records) != 12:
            raise ValueError(f"expected 12 manifest rows in {phase_name}, found {len(records)}")
        expected = {
            (job["phase"], job["dataset"], job["model"], int(job["pred_len"]), int(job["seed"])): job
            for job in expected_by_phase[phase_name]
        }
        seen = set()
        for record in records.values():
            if record.get("status") != "ok":
                raise ValueError(f"non-ok record: {record.get('job_id')}")
            source_hash = str(record.get("source_tree_sha256_before", ""))
            if source_hash != str(validation.get("source_tree_sha256")) or record.get("source_tree_sha256_after") != source_hash:
                raise ValueError(f"source hash mismatch: {record.get('job_id')}")
            coordinate = (str(record.get("phase")), str(record.get("dataset")), str(record.get("model")), int(record.get("pred_len")), int(record.get("seed")))
            if coordinate not in expected or coordinate in seen:
                raise ValueError(f"unexpected or duplicate coordinate: {coordinate}")
            seen.add(coordinate)
            if not _artifact_exists(record):
                raise ValueError(f"artifact hash mismatch: {record.get('job_id')}")
            artifact = Path(str(record["expected_artifact"]))
            result_dir = artifact.parent
            config = read_json(result_dir / "config.json")
            job = expected[coordinate]
            if config.get("plan_job_id") != record.get("job_id"):
                raise ValueError(f"plan job identity mismatch: {result_dir}")
            if str(config.get("data")) != str(job["dataset"]) or int(config.get("pred_len")) != HORIZON:
                raise ValueError(f"dataset/horizon mismatch: {result_dir}")
            actual_seed = int(config.get("run_seed", config.get("seed")))
            if actual_seed != int(job["seed"]):
                raise ValueError(f"seed mismatch: {result_dir}")
            if str(config.get("AutoCon_wnorm")) != "MeanStd":
                raise ValueError(f"normalization mismatch: {result_dir}")
            expected_lambda = job.get("autocon_lambda")
            actual_lambda = config.get("AutoCon_lambda") if config.get("AutoCon") is True else None
            actual_lambda = None if actual_lambda in (None, "", "none") else float(actual_lambda)
            if expected_lambda is None and actual_lambda is not None and config.get("AutoCon") is not True:
                raise ValueError(f"unexpected lambda in non-AutoCon config: {result_dir}")
            if expected_lambda is not None and actual_lambda != float(expected_lambda):
                raise ValueError(f"lambda mismatch: {result_dir}")
            _sidecar_check(config, result_dir)
            _test_artifact_check(result_dir, artifact)
            checkpoint_root = command_value(record.get("command", []), "--checkpoints")
            if checkpoint_root is None:
                raise ValueError(f"missing checkpoint root: {record.get('job_id')}")
            checkpoint_dir = Path(checkpoint_root) / result_dir.name
            best, best_epoch, epochs_ran, logs = trajectory(checkpoint_dir)
            relation_enabled = phase_name.startswith("autocon_")
            if relation_enabled:
                if config.get("AutoCon") is not True or config.get("mamba_acf_relation_loss") is not True:
                    raise ValueError(f"relation config mismatch: {result_dir}")
                if _telemetry_mean(logs, "autocon_loss") is None or _telemetry_mean(logs, "acf_relation_global_magnitude_ce") is None:
                    raise ValueError(f"missing AutoCon/relation telemetry: {checkpoint_dir}")
            elif config.get("AutoCon") is True:
                raise ValueError(f"unexpected AutoCon in non-relation phase: {result_dir}")
            rows.append({
                "phase": phase_name, "dataset": str(job["dataset"]), "horizon": HORIZON,
                "seed": int(job["seed"]), "model": str(job["model"]),
                "autocon_lambda": "" if actual_lambda is None else actual_lambda,
                "best_val_mse": best, "best_epoch": best_epoch, "epochs_ran": epochs_ran,
                "elapsed_sec": finite(record.get("elapsed_sec"), "elapsed_sec"),
                "autocon_loss_mean": _telemetry_mean(logs, "autocon_loss"),
                "relation_magnitude_ce_mean": _telemetry_mean(logs, "acf_relation_global_magnitude_ce"),
                "relation_signed_hinge_mean": _telemetry_mean(logs, "acf_relation_global_signed_hinge_contribution"),
                "memory_to_forecast_rms_mean": _telemetry_mean(logs, "acf_effective_memory_to_forecast_rms"),
                "config_path": str((result_dir / "config.json").resolve()),
                "checkpoint_path": str((checkpoint_dir / "loss_logs.json").resolve()),
            })
        if seen != set(expected):
            raise ValueError(f"plan/manifest coordinate mismatch in {phase_name}")
    if len(source_hashes) != 1 or not next(iter(source_hashes)):
        raise ValueError(f"phases do not share one source hash: {source_hashes}")
    expected_coordinates = {(dataset, HORIZON, seed) for dataset in DATASETS for seed in SEEDS}
    for phase_name in EXPECTED_PHASES:
        coordinates = {(row["dataset"], row["horizon"], row["seed"]) for row in rows if row["phase"] == phase_name}
        if coordinates != expected_coordinates:
            raise ValueError(f"incomplete exact coordinates: {phase_name}")
    return sorted(rows, key=lambda row: (row["phase"], row["dataset"], row["seed"]))


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    return float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)) if len(values) > 1 else 0.0


def paired(rows: list[dict], treatment: str, reference: str) -> list[dict]:
    index = {(row["phase"], row["dataset"], row["seed"]): row for row in rows}
    output = []
    for dataset in DATASETS:
        for seed in SEEDS:
            left = index[(treatment, dataset, seed)]
            right = index[(reference, dataset, seed)]
            delta = float(left["best_val_mse"]) - float(right["best_val_mse"])
            output.append({
                "treatment": treatment, "reference": reference, "dataset": dataset,
                "horizon": HORIZON, "seed": seed, "treatment_mse": float(left["best_val_mse"]),
                "reference_mse": float(right["best_val_mse"]), "delta_mse": delta,
                "relative_change_pct": 100.0 * delta / float(right["best_val_mse"]),
            })
    return output


def pair_summary(pairs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["treatment"], row["reference"], row["dataset"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        deltas = [float(row["delta_mse"]) for row in values]
        relative = [float(row["relative_change_pct"]) for row in values]
        mean_delta, sd_delta = _mean_sd(deltas)
        output.append({
            "treatment": key[0], "reference": key[1], "dataset": key[2],
            "horizon": HORIZON, "n_pairs": len(deltas),
            "mean_delta_mse": mean_delta, "sd_delta_mse": sd_delta,
            "ci95_delta_mse": _ci95(deltas), "mean_relative_change_pct": float(statistics.mean(relative)),
            "wins": int(sum(value < 0 for value in deltas)),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 6) -> str:
    return "n/a" if value is None or value == "" else f"{float(value):.{digits}f}"


def report(plan_path: Path, artifact_root: Path, output: Path, rows: list[dict], pair_rows: list[dict], summaries: list[dict], source_hash: str) -> None:
    index = {(row["phase"], row["dataset"]): [] for row in rows}
    for row in rows:
        index[(row["phase"], row["dataset"])].append(float(row["best_val_mse"]))
    lines = [
        "# Multidataset MambaACF candidate screen",
        "",
        "Status: exploratory, validation-only, exact-coordinate paired screen; not a publication result.",
        "The primary metric is the minimum pure-forecast `val_loss` in each checkpoint. Test artifacts were withheld and are not read.",
        "",
        "## Contract",
        "",
        f"- Plan: `{plan_path.resolve()}`",
        f"- Artifact root: `{artifact_root.resolve()}`",
        f"- Source tree hash: `{source_hash}`",
        "- Datasets: ETTh1, ETTm1, Electricity, Traffic; H96; M features; seeds 2021/2022/2023.",
        "- Arms: ACF-off MambaACF; train-only ACF memory; AutoCon+memory relation null (`lambda=0`); AutoCon+memory relation candidate (`lambda=0.001`).",
        "- All 48 rows passed phase validation, sidecar hash checks, and test-artifact absence checks.",
        "",
        "## Mean best validation MSE",
        "",
        "Each value is mean +/- sample SD over three seeds. Lower is better within a dataset; this is not a test ranking.",
        "",
        "| Dataset | ACF-off | ACF memory | AutoCon lambda=0 | AutoCon lambda=0.001 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        cells = []
        for phase in EXPECTED_PHASES:
            mean, sd = _mean_sd(index[(phase, dataset)])
            cells.append(f"{mean:.6f} +/- {sd:.6f}")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired deltas", "", "Negative treatment-minus-reference delta is favorable. `wins` counts lower validation MSE at the exact same seed.", "", "| Treatment | Reference | Dataset | mean delta | mean relative change | wins/3 | 95% CI delta |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for item in summaries:
        lines.append(f"| {item['treatment']} | {item['reference']} | {item['dataset']} | {_fmt(item['mean_delta_mse'])} | {_fmt(item['mean_relative_change_pct'], 2)}% | {item['wins']}/{item['n_pairs']} | [{_fmt(item['mean_delta_mse'] - item['ci95_delta_mse'])}, {_fmt(item['mean_delta_mse'] + item['ci95_delta_mse'])}] |")
    lines.extend(["", "## Mechanism telemetry", "", "AutoCon relation rows contain non-null `autocon_loss` and global relation magnitude/signed-hinge telemetry. These values confirm objective execution; they are not accuracy evidence.", "", "| Phase | Rows | mean AutoCon loss | mean relation CE | mean signed contribution | mean memory/forecast RMS |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for phase in EXPECTED_PHASES:
        subset = [row for row in rows if row["phase"] == phase]
        def mean_optional(key):
            values = [float(row[key]) for row in subset if row[key] is not None]
            return statistics.mean(values) if values else None
        lines.append(f"| {phase} | {len(subset)} | {_fmt(mean_optional('autocon_loss_mean'))} | {_fmt(mean_optional('relation_magnitude_ce_mean'))} | {_fmt(mean_optional('relation_signed_hinge_mean'))} | {_fmt(mean_optional('memory_to_forecast_rms_mean'))} |")
    lines.extend(["", "## Interpretation", "", "- First inspect ACF memory versus ACF-off. If it is not directionally stable across datasets, do not promote the memory branch to H336/H720.", "- The key AutoCon comparison is `lambda=0.001` versus the exact `lambda=0` relation null; a difference against ACF-off alone conflates the AutoCon head with the relation objective.", "- A favorable H96 screen is only a candidate gate. It cannot support a claim against PatchTST/TimesNet, an all-horizon claim, a non-inferiority claim, or a speed claim.", "- Training elapsed time in the manifest includes loading and short-budget setup. Synchronized prediction/train-step resource measurements remain required.", "", "## Provenance", "", f"- Plan SHA256: `{sha256(plan_path)}`", f"- Phase roots share source hash: `{source_hash}`", f"- Rows: `{len(rows)}`; paired rows: `{len(pair_rows)}`; test metrics produced: `false`."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_rows(plan_path, artifact_root)
    pairs = []
    pairs.extend(paired(rows, "acf_memory_h96", "acf_off_h96"))
    pairs.extend(paired(rows, "autocon_memory_relation_lambda0_h96", "acf_memory_h96"))
    pairs.extend(paired(rows, "autocon_memory_relation_lambda001_h96", "autocon_memory_relation_lambda0_h96"))
    pairs.extend(paired(rows, "autocon_memory_relation_lambda001_h96", "acf_off_h96"))
    summaries = pair_summary(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_rows.csv", rows)
    write_csv(output_dir / "paired_deltas.csv", pairs)
    write_csv(output_dir / "paired_summary.csv", summaries)
    source_hash = str(read_json(artifact_root / "acf_off_h96" / "phase_validation.json")["source_tree_sha256"])
    payload = {
        "schema_version": 1, "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "artifact_root": str(artifact_root), "source_tree_sha256": source_hash,
        "rows": len(rows), "paired_rows": len(pairs), "summary_rows": len(summaries),
        "phases": list(EXPECTED_PHASES), "datasets": list(DATASETS), "horizon": HORIZON,
        "seeds": list(SEEDS), "test_metrics_produced": False,
        "metric_source": "checkpoint/loss_logs.json:val_loss minimum",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report(plan_path, artifact_root, output_dir / "readout.md", rows, pairs, summaries, source_hash)
    print(json.dumps({"rows": len(rows), "paired_rows": len(pairs), "summary_rows": len(summaries), "readout": str(output_dir / 'readout.md')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
