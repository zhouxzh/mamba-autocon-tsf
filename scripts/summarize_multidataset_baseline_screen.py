#!/usr/bin/env python3
"""Summarize a frozen validation-only multidataset baseline screen.

The launcher withholds test metrics for this screen.  This reader therefore
uses the minimum pure-forecast ``val_loss`` in each checkpoint trajectory and
refuses incomplete, duplicated, failed, or cross-provenance rows.  It writes
seed-level rows, descriptive group summaries, exact-coordinate paired deltas,
and a short Markdown readout without selecting a final model.
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

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import (  # noqa: E402
    _artifact_exists,
    _job_specs,
    _load_json,
    _validate_plan,
)


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


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed manifest line {line_number}: {exc}") from exc
            if not isinstance(record, dict) or not record.get("job_id"):
                raise ValueError(f"manifest line {line_number} is not a job record")
            job_id = str(record["job_id"])
            if job_id in records:
                raise ValueError(f"duplicate manifest job: {job_id}")
            records[job_id] = record
    return records


def expected_jobs(plan: dict) -> list[dict]:
    errors = _validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    jobs = []
    for phase in plan.get("phases", []):
        jobs.extend(_job_specs(plan, phase, {}, allow_unresolved=False))
    return jobs


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _trajectory(checkpoint_dir: Path) -> tuple[float, int, int]:
    path = checkpoint_dir / "loss_logs.json"
    logs = read_json(path)
    values = logs.get("val_loss")
    if not isinstance(values, list) or not values:
        raise ValueError(f"empty validation trajectory: {path}")
    numeric = [_finite(value, f"{path}:val_loss") for value in values]
    best = min(numeric)
    best_index = numeric.index(best)
    best_epoch = int(logs.get("best_epoch", best_index + 1))
    epochs_ran = int(logs.get("epochs_ran", len(numeric)))
    if best_epoch < 1 or epochs_ran < best_epoch:
        raise ValueError(f"invalid epoch metadata: {path}")
    return best, best_epoch, epochs_ran


def load_rows(plan: dict, phase_root: Path) -> list[dict]:
    jobs = expected_jobs(plan)
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    records = read_manifest(manifest_path)
    validation = read_json(phase_root / "phase_validation.json")
    if validation.get("valid") is not True:
        raise ValueError("phase validation is not valid")
    expected_count = len(jobs)
    if int(validation.get("expected_jobs", -1)) != expected_count:
        raise ValueError("phase validation expected_jobs disagrees with plan")
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} manifest rows, found {len(records)}")
    source_hash = str(validation.get("source_tree_sha256", ""))
    if len(source_hash) != 64:
        raise ValueError("phase validation has no source tree hash")

    expected_index = {}
    for job in jobs:
        key = (
            str(job["phase"]), str(job["dataset"]), str(job["model"]),
            int(job["pred_len"]), int(job["seed"]), job.get("autocon_lambda"),
        )
        if key in expected_index:
            raise ValueError(f"duplicate expected coordinate: {key}")
        expected_index[key] = job

    rows = []
    seen_coordinates = set()
    seen_jobs = set()
    for record in records.values():
        job_id = str(record["job_id"])
        record_key = (
            str(record.get("phase")), str(record.get("dataset")), str(record.get("model")),
            int(record.get("pred_len")), int(record.get("seed")),
            None if record.get("autocon_lambda") in (None, "", "none") else float(record.get("autocon_lambda")),
        )
        job = expected_index.get(record_key)
        if job is None:
            raise ValueError(f"manifest job is not in plan: {job_id}")
        if record_key in seen_jobs:
            raise ValueError(f"duplicate manifest coordinate: {record_key}")
        seen_jobs.add(record_key)
        if record.get("status") != "ok":
            raise ValueError(f"non-ok manifest record: {job_id}")
        if record.get("source_tree_sha256_before") != source_hash:
            raise ValueError(f"source hash mismatch before job: {job_id}")
        if record.get("source_tree_sha256_after") != source_hash:
            raise ValueError(f"source hash mismatch after job: {job_id}")
        if not _artifact_exists(record):
            raise ValueError(f"artifact hash or anchor mismatch: {job_id}")
        artifact = Path(str(record.get("expected_artifact", "")))
        result_dir = artifact.parent
        config_path = result_dir / "config.json"
        config = read_json(config_path)
        if str(config.get("plan_job_id")) != job_id:
            raise ValueError(f"config plan_job_id mismatch: {config_path}")
        expected_coordinate = (
            str(job["dataset"]), str(job["model"]), int(job["pred_len"]), int(job["seed"])
        )
        actual_coordinate = (
            str(config.get("data")), str(config.get("model")),
            int(config.get("pred_len")), int(config.get("run_seed", config.get("seed"))),
        )
        if actual_coordinate != expected_coordinate:
            raise ValueError(
                f"config coordinate mismatch for {job_id}: {actual_coordinate!r} != {expected_coordinate!r}"
            )
        if str(config.get("features")) != str(job["config"].get("features", "M")):
            raise ValueError(f"feature contract mismatch: {config_path}")
        marker = result_dir / "training_complete.json"
        marker_payload = read_json(marker)
        if marker_payload.get("status") != "trained_without_test":
            raise ValueError(f"test was not withheld in {marker}")
        forbidden = [
            result_dir / "metrics.npy", result_dir / "original_metrics.npy",
            result_dir / "stream_metrics.json", result_dir / "stream_samples.npz",
        ]
        present = [str(path) for path in forbidden if path.exists()]
        if present:
            raise ValueError(f"test metric artifact(s) present for {job_id}: {present}")
        checkpoint_root = command_value(record.get("command", []), "--checkpoints")
        if checkpoint_root is None:
            raise ValueError(f"missing checkpoint root: {job_id}")
        checkpoint_dir = Path(checkpoint_root) / result_dir.name
        best, best_epoch, epochs_ran = _trajectory(checkpoint_dir)
        coordinate = expected_coordinate
        if coordinate in seen_coordinates:
            raise ValueError(f"duplicate dataset/horizon/seed coordinate: {coordinate}")
        seen_coordinates.add(coordinate)
        rows.append({
            "dataset": expected_coordinate[0], "model": expected_coordinate[1],
            "horizon": expected_coordinate[2], "seed": expected_coordinate[3],
            "best_val_mse": best, "best_epoch": best_epoch, "epochs_ran": epochs_ran,
            "elapsed_sec": _finite(record.get("elapsed_sec"), f"{job_id}:elapsed_sec"),
            "job_id": job_id, "config_path": str(config_path.resolve()),
            "checkpoint_path": str((checkpoint_dir / "loss_logs.json").resolve()),
        })
    if seen_jobs != set(expected_index):
        missing = sorted(set(expected_index) - seen_jobs)
        extras = sorted(seen_jobs - set(expected_index))
        raise ValueError(f"plan/manifest coordinate mismatch: missing={missing[:3]}, extras={extras[:3]}")
    return sorted(rows, key=lambda row: (row["dataset"], row["horizon"], row["model"], row["seed"]))


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(finite) - 1) * stats.sem(finite))


def group_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["horizon"], row["model"])].append(row)
    output = []
    for (dataset, horizon, model), values in sorted(grouped.items()):
        mse = [float(row["best_val_mse"]) for row in values]
        elapsed = [float(row["elapsed_sec"]) for row in values]
        mse_mean, mse_sd = mean_sd(mse)
        elapsed_mean, elapsed_sd = mean_sd(elapsed)
        output.append({
            "dataset": dataset, "horizon": horizon, "model": model,
            "n_seeds": len(values), "seeds": ",".join(str(row["seed"]) for row in sorted(values, key=lambda item: item["seed"])),
            "best_val_mse_mean": mse_mean, "best_val_mse_sd": mse_sd,
            "best_val_mse_ci95": ci95(mse), "elapsed_sec_mean": elapsed_mean,
            "elapsed_sec_sd": elapsed_sd,
        })
    return output


def paired_rows(rows: list[dict], references: tuple[str, ...] = ("DLinear", "NLinear", "PatchTST", "TimesNet")) -> list[dict]:
    indexed = {(row["dataset"], row["horizon"], row["model"], row["seed"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows})
    horizons = sorted({row["horizon"] for row in rows})
    models = sorted({row["model"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    output = []
    for dataset in datasets:
        for horizon in horizons:
            for candidate in models:
                for reference in references:
                    if candidate == reference:
                        continue
                    pairs = []
                    for seed in seeds:
                        left = indexed.get((dataset, horizon, candidate, seed))
                        right = indexed.get((dataset, horizon, reference, seed))
                        if left is None or right is None:
                            raise ValueError(f"missing exact pair: {dataset}/{horizon}/{candidate}/{reference}/{seed}")
                        pairs.append((float(left["best_val_mse"]), float(right["best_val_mse"])))
                    deltas = [left - right for left, right in pairs]
                    reference_mean = statistics.mean(right for _, right in pairs)
                    output.append({
                        "dataset": dataset, "horizon": horizon, "candidate": candidate,
                        "reference": reference, "n_pairs": len(deltas),
                        "mean_delta_mse": float(statistics.mean(deltas)),
                        "sd_delta_mse": float(statistics.stdev(deltas)) if len(deltas) > 1 else 0.0,
                        "relative_change_pct": float(100.0 * statistics.mean(deltas) / reference_mean),
                        "candidate_wins": int(sum(delta < 0.0 for delta in deltas)),
                    })
    return output


def best_cell_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["horizon"], row["seed"])].append(row)
    output = []
    for (dataset, horizon, seed), values in sorted(grouped.items()):
        winner = min(values, key=lambda row: float(row["best_val_mse"]))
        output.append({
            "dataset": dataset, "horizon": horizon, "seed": seed,
            "best_validation_model": winner["model"],
            "best_val_mse": float(winner["best_val_mse"]),
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown_report(plan_path: Path, phase_root: Path, output: Path, rows: list[dict], summary: list[dict], paired: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    horizons = sorted({row["horizon"] for row in rows})
    models = sorted({row["model"] for row in rows})
    summary_index = {(row["dataset"], row["horizon"], row["model"]): row for row in summary}
    lines = [
        "# Multidataset plain-baseline screen readout", "",
        "Status: exploratory, validation-only, and not a publication result.",
        "The primary value is the minimum pure-forecast `val_loss` in each checkpoint trajectory; test metrics were withheld and are not read.", "",
        "## Contract", "",
        f"- Plan: `{plan_path.resolve()}`",
        f"- Phase root: `{phase_root.resolve()}`",
        f"- Coordinates: {len(rows)} rows; {len(datasets)} datasets; {len(models)} models; {len(horizons)} horizons; seeds {sorted({row['seed'] for row in rows})}.",
        f"- Datasets: {', '.join(datasets)}.",
        f"- Models: {', '.join(models)}.",
        "- All rows use M features, the same frozen optimizer/normalization declaration, and `skip_test=true`.",
        "- Elapsed time is end-to-end short-budget training wall time, not a synchronized inference or train-step speed benchmark.", "",
        "## Mean best validation MSE", "",
        "Values are mean +/- sample SD across the three exact seeds. Lower is better within a dataset and horizon; values are not comparable across datasets.", "",
        "| Dataset | Horizon | " + " | ".join(models) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in models) + " |",
    ]
    for dataset in datasets:
        for horizon in horizons:
            cells = []
            for model in models:
                item = summary_index[(dataset, horizon, model)]
                cells.append(f"{_fmt(item['best_val_mse_mean'])} +/- {_fmt(item['best_val_mse_sd'])}")
            lines.append(f"| {dataset} | {horizon} | " + " | ".join(cells) + " |")

    non_mamba = tuple(model for model in ("DLinear", "NLinear", "PatchTST", "TimesNet") if model in models)
    mambas = tuple(model for model in ("Mamba1", "Mamba2") if model in models)
    if non_mamba and mambas:
        lines.extend(["", "## Mamba versus strongest non-Mamba mean", "",
                      "The reference is the lowest mean validation MSE among DLinear, NLinear, PatchTST, and TimesNet for each dataset/horizon. This is descriptive and must not be used for test selection.", "",
                      "| Dataset | Horizon | strongest non-Mamba | Mamba1 change | Mamba2 change |",
                      "| --- | ---: | --- | ---: | ---: |"])
        for dataset in datasets:
            for horizon in horizons:
                base = min(non_mamba, key=lambda model: summary_index[(dataset, horizon, model)]["best_val_mse_mean"])
                base_value = summary_index[(dataset, horizon, base)]["best_val_mse_mean"]
                changes = []
                for model in mambas:
                    value = summary_index[(dataset, horizon, model)]["best_val_mse_mean"]
                    changes.append(f"{100.0 * (value - base_value) / base_value:+.2f}%")
                while len(changes) < 2:
                    changes.append("n/a")
                lines.append(f"| {dataset} | {horizon} | {base} ({_fmt(base_value)}) | {changes[0]} | {changes[1]} |")

    lines.extend(["", "## Mean elapsed time", "",
                  "These values include data loading and short-budget training. They are not a deployment throughput claim.", "",
                  "| Dataset | Horizon | " + " | ".join(models) + " |",
                  "| --- | ---: | " + " | ".join("---:" for _ in models) + " |"])
    for dataset in datasets:
        for horizon in horizons:
            cells = [_fmt(summary_index[(dataset, horizon, model)]["elapsed_sec_mean"], 2) for model in models]
            lines.append(f"| {dataset} | {horizon} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Interpretation limits", "",
                  "- This screen establishes coverage and validation-only descriptive patterns; it does not establish a final ranking, non-inferiority, or an accuracy/speed claim.",
                  "- Mamba1 and Mamba2 remain separate model families and are not pooled.",
                  "- Overlapping windows are not independent replicates; the paired unit is dataset/horizon/seed.",
                  "- A publication matrix still needs a frozen long-budget protocol, explicit test release, multiple confirmation seeds, and synchronized train-step/inference resource measurements.",
                  "- Full paired seed deltas are in `paired_deltas.csv`; exact-cell descriptive winners are in `best_validation_by_cell.csv`.", "",
                  "## Provenance", "",
                  f"- Plan SHA256: `{sha256(plan_path)}`",
                  f"- Manifest SHA256: `{sha256(phase_root / 'logs' / 'job_manifest.jsonl')}`",
                  f"- Phase validation SHA256: `{sha256(phase_root / 'phase_validation.json')}`",
                  f"- Rows: `{len(rows)}`; paired rows: `{len(paired)}`; test metrics produced: `false`."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    phase_root = args.phase_root.resolve()
    output_dir = (args.output_dir or phase_root / "summary").resolve()
    plan = read_json(plan_path)
    rows = load_rows(plan, phase_root)
    summary = group_summary(rows)
    paired = paired_rows(rows)
    winners = best_cell_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "raw_validation_rows.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_deltas.csv", paired)
    write_csv(output_dir / "best_validation_by_cell.csv", winners)
    payload = {
        "schema_version": 1,
        "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "phase_root": str(phase_root),
        "manifest": str((phase_root / "logs" / "job_manifest.jsonl").resolve()),
        "manifest_sha256": sha256(phase_root / "logs" / "job_manifest.jsonl"),
        "phase_validation_sha256": sha256(phase_root / "phase_validation.json"),
        "rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired),
        "winner_rows": len(winners), "datasets": sorted({row["dataset"] for row in rows}),
        "models": sorted({row["model"] for row in rows}), "horizons": sorted({row["horizon"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}), "test_metrics_produced": False,
        "metric_source": "checkpoint/loss_logs.json:val_loss minimum",
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = output_dir / "readout.md"
    markdown_report(plan_path, phase_root, report_path, rows, summary, paired)
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
