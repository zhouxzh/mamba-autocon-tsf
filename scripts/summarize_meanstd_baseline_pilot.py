#!/usr/bin/env python
"""Summarize a validation-only MeanStd baseline pilot.

The runner intentionally withholds test metrics for short exploratory plans,
so this reader uses the best validation trajectory in each checkpoint rather
than expecting ``metrics.npy``.  It refuses incomplete or failed manifests and
keeps paired seed comparisons within each dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba1", "Mamba2")
MAMBA_MODELS = ("Mamba1", "Mamba2")
NON_MAMBA_MODELS = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_manifest(path: Path) -> list[dict]:
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
            records[str(record["job_id"])] = record
    return list(records.values())


def command_value(command: list, flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError, TypeError):
        return None


def load_rows(phase_root: Path, manifest_path: Path) -> list[dict]:
    records = read_manifest(manifest_path)
    rows = []
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"manifest contains non-ok record: {record.get('job_id')}")
        if record.get("model") not in MODELS:
            raise ValueError(f"unexpected model in pilot manifest: {record.get('model')}")
        artifact = Path(str(record.get("expected_artifact", "")))
        if not artifact.is_file():
            raise ValueError(f"missing expected artifact: {artifact}")
        result_dir = artifact.parent
        config_path = result_dir / "config.json"
        config = read_json(config_path)
        if config.get("AutoCon_wnorm") != "MeanStd":
            raise ValueError(f"non-MeanStd config in {config_path}")
        checkpoint_root = command_value(record.get("command", []), "--checkpoints")
        if checkpoint_root is None:
            raise ValueError(f"missing checkpoint root: {record.get('job_id')}")
        loss_path = Path(checkpoint_root) / result_dir.name / "loss_logs.json"
        logs = read_json(loss_path)
        values = logs.get("val_loss")
        if not isinstance(values, list) or not values:
            raise ValueError(f"empty validation trajectory: {loss_path}")
        numeric = [float(value) for value in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite validation trajectory: {loss_path}")
        best = min(numeric)
        rows.append({
            "dataset": str(record["dataset"]),
            "horizon": int(record["pred_len"]),
            "model": str(record["model"]),
            "seed": int(record["seed"]),
            "best_val_mse": best,
            "best_epoch": int(logs.get("best_epoch", numeric.index(best) + 1)),
            "epochs_ran": int(logs.get("epochs_ran", len(numeric))),
            "elapsed_sec": float(record.get("elapsed_sec", float("nan"))),
            "job_id": str(record["job_id"]),
            "config_path": str(config_path),
        })
    expected = 4 * len(MODELS) * 3
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    coordinates = {(row["dataset"], row["model"], row["seed"]) for row in rows}
    if len(coordinates) != len(rows):
        raise ValueError("duplicate dataset/model/seed coordinates")
    return sorted(rows, key=lambda row: (row["dataset"], row["model"], row["seed"]))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows: list[dict]) -> list[dict]:
    indexed = {(row["dataset"], row["model"], row["seed"]): row for row in rows}
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for mamba in MAMBA_MODELS:
            for baseline in NON_MAMBA_MODELS:
                deltas = []
                wins = 0
                for seed in sorted({row["seed"] for row in rows}):
                    left = indexed[(dataset, mamba, seed)]["best_val_mse"]
                    right = indexed[(dataset, baseline, seed)]["best_val_mse"]
                    delta = left - right
                    deltas.append(delta)
                    wins += int(delta < 0.0)
                average, sd = mean_sd(deltas)
                baseline_values = [indexed[(dataset, baseline, seed)]["best_val_mse"] for seed in sorted({row["seed"] for row in rows})]
                denominator = statistics.mean(baseline_values)
                output.append({
                    "dataset": dataset,
                    "mamba": mamba,
                    "baseline": baseline,
                    "n_pairs": len(deltas),
                    "mean_delta_mse": average,
                    "sd_delta_mse": sd,
                    "relative_change_pct": 100.0 * average / denominator,
                    "mamba_wins": wins,
                })
    return output


def markdown_report(
    plan_path: Path,
    validation_path: Path,
    manifest_path: Path,
    rows: list[dict],
    paired: list[dict],
    output: Path,
    resource_path: Path | None = None,
) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    walls: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row["best_val_mse"])
        walls[(row["dataset"], row["model"])].append(row["elapsed_sec"])
    datasets = sorted({row["dataset"] for row in rows})
    lines = [
        "# MeanStd baseline pilot readout",
        "",
        "Date: 2026-09-07  ",
        "Status: exploratory, validation-only, short budget; not a publication result.",
        "",
        "## Contract",
        "",
        f"- Plan: `{plan_path}`",
        f"- Phase validation: `{validation_path}`",
        f"- Manifest: `{manifest_path}`",
        "- Four datasets: ETTh1, ETTm1, Electricity, Traffic; H96; M features; seeds 2021/2022/2023.",
        "- Seven models: DLinear, NLinear, PatchTST, TimesNet, iTransformer, Mamba1, Mamba2.",
        "- Four training epochs, patience 2, FP32, batch 32, `AutoCon_wnorm=MeanStd`, `skip_test=true`.",
        "- DLinear/NLinear use the new opt-in MeanStd path; their default `LastVal` behavior is unchanged.",
        "",
        "## Best validation MSE",
        "",
        "Values are mean +/- sample SD across the three seeds. Lower is better; values are not comparable across datasets.",
        "",
        "| Dataset | " + " | ".join(MODELS) + " |",
        "| --- | " + " | ".join("---:" for _ in MODELS) + " |",
    ]
    for dataset in datasets:
        cells = []
        for model in MODELS:
            avg, sd = mean_sd(grouped[(dataset, model)])
            cells.append(f"{avg:.6f} +/- {sd:.6f}")
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Mamba versus strongest non-Mamba mean",
        "",
        "The reference is the lowest mean among DLinear, NLinear, PatchTST, TimesNet, and iTransformer. This is descriptive only; it is not a pre-registered selection gate.",
        "",
        "| Dataset | strongest non-Mamba | Mamba1 change | Mamba2 change |",
        "| --- | ---: | ---: | ---: |",
    ])
    for dataset in datasets:
        baseline_means = {
            model: mean_sd(grouped[(dataset, model)])[0] for model in NON_MAMBA_MODELS
        }
        baseline = min(baseline_means, key=baseline_means.get)
        base = baseline_means[baseline]
        changes = []
        for mamba in MAMBA_MODELS:
            mamba_mean = mean_sd(grouped[(dataset, mamba)])[0]
            changes.append(f"{100.0 * (mamba_mean - base) / base:+.2f}%")
        lines.append(f"| {dataset} | {baseline} ({base:.6f}) | {changes[0]} | {changes[1]} |")

    lines.extend([
        "",
        "## Paired seed wins",
        "",
        "A win means the Mamba validation MSE is lower for the same dataset and seed. The full pair table is in `paired_deltas.csv`.",
        "",
        "| Dataset | Mamba | PatchTST wins/3 | TimesNet wins/3 | iTransformer wins/3 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    pair_index = {(row["dataset"], row["mamba"], row["baseline"]): row for row in paired}
    for dataset in datasets:
        for mamba in MAMBA_MODELS:
            values = [pair_index[(dataset, mamba, baseline)]["mamba_wins"] for baseline in ("PatchTST", "TimesNet", "iTransformer")]
            lines.append(f"| {dataset} | {mamba} | {values[0]}/3 | {values[1]}/3 | {values[2]}/3 |")

    lines.extend([
        "",
        "## Mean training wall time",
        "",
        "Mean runner elapsed time over three seeds; this includes data loading and short-budget initialization and is not a deployment throughput measurement.",
        "",
        "| Dataset | " + " | ".join(MODELS) + " |",
        "| --- | " + " | ".join("---:" for _ in MODELS) + " |",
    ])
    for dataset in datasets:
        cells = [f"{statistics.mean(walls[(dataset, model)]):.3f}" for model in MODELS]
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")

    if resource_path is not None and resource_path.is_file():
        resource_rows = list(csv.DictReader(resource_path.open(encoding="utf-8", newline="")))
        resource_index = {(row.get("dataset"), row.get("model")): row for row in resource_rows}
        lines.extend([
            "",
            "## Prediction-only GPU resources",
            "",
            "RTX 5090 D, FP32, device-resident synthetic input, batch 8, 10 warm-up forwards, 50 timed forwards. These are inference-load measurements only; they exclude data loading, host/device transfer, backward propagation, optimizer work, and relation loss.",
            "",
            "| Dataset | " + " | ".join(MODELS) + " |",
            "| --- | " + " | ".join("---:" for _ in MODELS) + " |",
        ])
        for dataset in datasets:
            cells = [
                f"{float(resource_index[(dataset, model)]['latency_ms_median']):.6f} ms"
                for model in MODELS
            ]
            lines.append("| " + dataset + " | " + " | ".join(cells) + " |")
        lines.extend([
            "",
            "The resource CSV also records p95 latency, synthetic sequences/s, active parameters, and peak allocated memory. In this contract Mamba-1/2 are faster than TimesNet but slower than iTransformer and the linear baselines on all four datasets; this is not an end-to-end training-speed claim.",
        ])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- MeanStd parity removes one known normalization confound, but four epochs is a screening budget rather than a converged training protocol.",
        "- The strongest model is dataset-dependent. Mamba should not be pooled into a universal accuracy claim.",
        "- Mamba rows are plain Mamba encoder baselines with the repository forecasting head; they do not use ACF memory or the AutoCon relation objective.",
        "- Test metrics were not produced. Do not use this pilot to select a final model or report a publication ranking.",
        "- A full paper matrix still needs a frozen long-budget protocol, at least three to five seeds, all intended horizons, explicit test release, and train-step resource measurements.",
        "",
        "## Provenance",
        "",
        f"- Plan SHA256: `{sha256(plan_path)}`",
        f"- Validation SHA256: `{sha256(validation_path)}`",
        f"- Manifest SHA256: `{sha256(manifest_path)}`",
        *([f"- Resource CSV SHA256: `{sha256(resource_path)}`"] if resource_path is not None and resource_path.is_file() else []),
        f"- Rows: `{len(rows)}`; paired comparisons: `{len(paired)}`.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    phase_root = args.phase_root.resolve()
    plan_path = args.plan.resolve()
    output_dir = (args.output_dir or phase_root / "summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    validation_path = phase_root / "phase_validation.json"
    validation = read_json(validation_path)
    if validation.get("valid") is not True:
        raise SystemExit("phase validation is not valid")
    rows = load_rows(phase_root, manifest_path)
    paired = paired_rows(rows)
    raw_fields = ["dataset", "horizon", "model", "seed", "best_val_mse", "best_epoch", "epochs_ran", "elapsed_sec", "job_id", "config_path"]
    pair_fields = ["dataset", "mamba", "baseline", "n_pairs", "mean_delta_mse", "sd_delta_mse", "relative_change_pct", "mamba_wins"]
    write_csv(output_dir / "raw_validation_rows.csv", rows, raw_fields)
    write_csv(output_dir / "paired_deltas.csv", paired, pair_fields)
    summary = {
        "schema_version": 1,
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "validation": str(validation_path),
        "validation_sha256": sha256(validation_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "rows": len(rows),
        "paired_rows": len(paired),
        "datasets": sorted({row["dataset"] for row in rows}),
        "models": list(MODELS),
        "seeds": sorted({row["seed"] for row in rows}),
        "horizons": sorted({row["horizon"] for row in rows}),
        "test_metrics_produced": False,
        "normalization": "MeanStd",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    report_path = output_dir / "meanstd_baseline_pilot_readout.md"
    resource_path = phase_root / "resources" / "model_resources.csv"
    markdown_report(plan_path, validation_path, manifest_path, rows, paired, report_path, resource_path)
    print(json.dumps({"rows": len(rows), "paired_rows": len(paired), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
