#!/usr/bin/env python
"""Summarize the validation-only classical-baseline extension pilot.

The reader is intentionally independent of the training runner.  It checks
the manifest/result/checkpoint contract, then writes seed-level rows, paired
deltas, and a compact Markdown readout.  It never reads test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean_sd(values):
    values = [float(value) for value in values]
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt(value, digits=6):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(phase_root: Path, plan: dict):
    phases = plan.get("phases", [])
    if len(phases) != 1:
        raise ValueError("this summarizer expects exactly one pilot phase")
    phase = phases[0]
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    records = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("job_id"):
                raise ValueError(f"invalid manifest row {line_number}")
            records[str(record["job_id"])] = record
    expected = len(phase["datasets"]) * len(phase["models"]) * len(phase["horizons"]) * len(phase["seeds"])
    if len(records) != expected:
        raise ValueError(f"manifest has {len(records)} unique jobs; expected {expected}")

    rows = []
    for job_id, record in sorted(records.items()):
        if record.get("status") != "ok" or int(record.get("returncode", 1)) != 0:
            raise ValueError(f"job {job_id} is not successful")
        expected_artifact = Path(record["expected_artifact"])
        result_dir = expected_artifact.parent
        config_path = result_dir / "config.json"
        checkpoint_dir = phase_root / "checkpoints" / result_dir.name
        logs_path = checkpoint_dir / "loss_logs.json"
        if not config_path.is_file() or not logs_path.is_file():
            raise ValueError(f"job {job_id} is missing config or loss logs")
        config = _read_json(config_path)
        logs = _read_json(logs_path)
        values = [float(value) for value in logs.get("val_loss", [])]
        train_values = [float(value) for value in logs.get("train_loss", [])]
        if not values or len(values) != len(train_values):
            raise ValueError(f"job {job_id} has an invalid loss trajectory")
        if not all(math.isfinite(value) for value in values + train_values):
            raise ValueError(f"job {job_id} has non-finite losses")
        model = str(config.get("model", record.get("model")))
        dataset = str(config.get("data", record.get("dataset")))
        seed = int(config.get("run_seed", config.get("seed", record.get("seed"))))
        if model != str(record.get("model")) or dataset != str(record.get("dataset")) or seed != int(record.get("seed")):
            raise ValueError(f"job {job_id} config disagrees with manifest")
        if str(config.get("AutoCon_wnorm", "")).lower() != "meanstd":
            raise ValueError(f"job {job_id} is not a MeanStd run")
        rows.append({
            "job_id": job_id,
            "dataset": dataset,
            "model": model,
            "features": config.get("features"),
            "seq_len": int(config.get("seq_len", 0)),
            "pred_len": int(config.get("pred_len", 0)),
            "seed": seed,
            "best_val_mse": min(values),
            "best_epoch": int(logs.get("best_epoch", values.index(min(values)) + 1)),
            "epochs_ran": int(logs.get("epochs_ran", len(values))),
            "elapsed_sec": float(record.get("elapsed_sec", math.nan)),
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
        })
    return phase, rows


def _summaries(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model"]), []).append(row)
    summaries = []
    for (dataset, model), values in sorted(groups.items()):
        mse_mean, mse_sd = _mean_sd([item["best_val_mse"] for item in values])
        wall_mean, wall_sd = _mean_sd([item["elapsed_sec"] for item in values])
        summaries.append({
            "dataset": dataset,
            "model": model,
            "n_seeds": len(values),
            "seeds": ",".join(str(item["seed"]) for item in sorted(values, key=lambda item: item["seed"])),
            "best_val_mse_mean": mse_mean,
            "best_val_mse_sd": mse_sd,
            "train_elapsed_mean_sec": wall_mean,
            "train_elapsed_sd_sec": wall_sd,
            "parameter_note": "capacity is fixed to the small H96 screening configuration; not a final parameter-matched comparison",
        })
    return summaries


def _paired(rows):
    index = {(row["dataset"], row["seed"], row["model"]): row for row in rows}
    models = sorted({row["model"] for row in rows})
    out = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for treatment in ("Mamba1", "Mamba2"):
            for baseline in models:
                if baseline.startswith("Mamba") or baseline == treatment:
                    continue
                deltas = []
                wins = 0
                for seed in sorted({row["seed"] for row in rows if row["dataset"] == dataset}):
                    left = index.get((dataset, seed, treatment))
                    right = index.get((dataset, seed, baseline))
                    if left is None or right is None:
                        continue
                    delta = (left["best_val_mse"] - right["best_val_mse"]) / right["best_val_mse"] * 100.0
                    deltas.append(delta)
                    wins += int(delta < 0)
                if deltas:
                    out.append({
                        "dataset": dataset,
                        "treatment": treatment,
                        "baseline": baseline,
                        "n_pairs": len(deltas),
                        "mean_relative_delta_percent": statistics.mean(deltas),
                        "wins": wins,
                    })
    return out


def _resource_rows(phase_root: Path):
    path = phase_root / "resources" / "model_resources.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        out.append({
            "dataset": row.get("dataset"),
            "model": row.get("model"),
            "latency_ms_median": float(row["latency_ms_median"]),
            "latency_ms_p95": float(row["latency_ms_p95"]),
            "parameter_trainable": int(float(row["parameter_trainable"])),
            "steady_state_peak_increment_mb": float(row["steady_state_peak_increment_mb"]),
        })
    return sorted(out, key=lambda row: (row["dataset"], row["latency_ms_median"]))


def _write_readout(path: Path, plan_path: Path, phase, rows, summaries, paired, resources, hashes):
    models = list(phase["models"])
    datasets = list(phase["datasets"])
    mamba_models = [model for model in models if model in {"Mamba1", "Mamba2"}]
    is_mamba_extension = bool(mamba_models)
    title = "Classical baseline extension pilot readout"
    lines = [
        f"# {title}",
        "",
        "Date: 2026-09-07  ",
        "Status: exploratory, validation-only, short budget; not a publication result.",
        "",
        "## Contract",
        "",
        f"- Plan: `{plan_path.resolve()}`",
        f"- Datasets: {', '.join(datasets)}; H96; M features; seeds 2021/2022/2023.",
        f"- Models: {', '.join(models)}.",
        "- Four epochs, patience 2, FP32, batch 32, `AutoCon_wnorm=MeanStd`, `skip_test=true`.",
        f"- Successful rows: `{len(rows)}`; test metrics produced: `false`.",
        "",
        "## Validation MSE",
        "",
        "Values are mean +/- sample SD across seeds; lower is better and datasets are not comparable.",
        "",
        "| Dataset | " + " | ".join(models) + " |",
        "| --- | " + " | ".join("---:" for _ in models) + " |",
    ]
    summary_index = {(row["dataset"], row["model"]): row for row in summaries}
    for dataset in datasets:
        cells = []
        for model in models:
            item = summary_index[(dataset, model)]
            cells.append(f"{_fmt(item['best_val_mse_mean'])} +/- {_fmt(item['best_val_mse_sd'])}")
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")
    if is_mamba_extension:
        lines += ["", "## Mamba paired deltas", "", "Negative values favor Mamba; this is descriptive screening evidence.", "", "| Dataset | Mamba | Baseline | Relative MSE delta | Wins |", "| --- | --- | --- | ---: | ---: |"]
        for item in paired:
            lines.append(f"| {item['dataset']} | {item['treatment']} | {item['baseline']} | {_fmt(item['mean_relative_delta_percent'], 3)}% | {item['wins']}/{item['n_pairs']} |")
    lines += ["", "## Prediction-only resources", "", "RTX 5090 D, device-resident synthetic input, batch 8, 10 warm-up and 50 timed forwards. These exclude data loading, transfer, backward, optimizer, and relation loss.", "", "| Dataset | Model | Median ms | P95 ms | Trainable parameters | Peak increment MB |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for item in resources:
        lines.append(f"| {item['dataset']} | {item['model']} | {_fmt(item['latency_ms_median'], 6)} | {_fmt(item['latency_ms_p95'], 6)} | {item['parameter_trainable']} | {_fmt(item['steady_state_peak_increment_mb'], 3)} |")
    interpretation = [
        "The five classical models are now executable through the maintained plan runner; FEDformer and ETSformer compatibility fixes are covered by regression tests.",
        "This pilot only tests H96, three seeds, and four epochs. It cannot establish convergence, test ranking, non-inferiority, universal accuracy, AutoCon gain, or end-to-end speed.",
    ]
    if is_mamba_extension:
        interpretation.append("Mamba rows are plain Mamba encoder baselines without ACF memory or the AutoCon relation objective.")
    else:
        interpretation.append("This phase contains only classical models; comparisons with Mamba or the seven-model core pilot require a matched source snapshot and a separately reviewed merged matrix.")
    interpretation += [
        "Resource measurements are prediction-only. A paper claim about efficiency requires matched train-step and end-to-end measurements.",
        "Test release and full-budget replication remain required before any publication table.",
    ]
    lines += ["", "## Interpretation", ""] + [f"- {item}" for item in interpretation] + ["", "## Provenance"]
    for key, value in hashes.items():
        lines.append(f"- {key}: `{value}`")
    lines.append(f"- Rows: `{len(rows)}`; paired comparisons: `{len(paired)}`; resource rows: `{len(resources)}`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = _read_json(args.plan)
    phase, rows = _load_rows(args.phase_root, plan)
    summaries = _summaries(rows)
    paired = _paired(rows)
    resources = _resource_rows(args.phase_root)
    summary_dir = args.phase_root / "summary"
    _write_csv(summary_dir / "raw_validation_rows.csv", rows)
    _write_csv(summary_dir / "summary.csv", summaries)
    _write_csv(summary_dir / "paired_deltas.csv", paired)
    _write_csv(summary_dir / "resources.csv", resources)
    phase_validation = args.phase_root / "phase_validation.json"
    manifest = args.phase_root / "logs" / "job_manifest.jsonl"
    resource_csv = args.phase_root / "resources" / "model_resources.csv"
    hashes = {
        "plan_sha256": _sha256(args.plan),
        "phase_validation_sha256": _sha256(phase_validation),
        "manifest_sha256": _sha256(manifest),
        "resource_csv_sha256": _sha256(resource_csv),
    }
    payload = {
        "status": "exploratory_validation_only",
        "datasets": phase["datasets"],
        "models": phase["models"],
        "rows": len(rows),
        "summary": summaries,
        "paired_deltas": paired,
        "resources": resources,
        "provenance": hashes,
    }
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_readout(summary_dir / "classical_extension_pilot_readout.md", args.plan, phase, rows, summaries, paired, resources, hashes)
    print(f"Validated and summarized {len(rows)} rows; paired={len(paired)}; resources={len(resources)}")


if __name__ == "__main__":
    main()
