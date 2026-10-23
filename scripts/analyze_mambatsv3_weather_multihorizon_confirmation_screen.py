#!/usr/bin/env python3
"""Strict readout for the Weather MambaTS v3 multi-horizon screen."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

DATASETS = ("weather",)
HORIZONS = (96, 336, 720)
SEEDS = (2021, 2022, 2023)
PHASES = {
    "plain": "weather_mambatsv3_plain_h96_h336_h720",
    "null": "weather_mambatsv3_autocon_null_h96_h336_h720",
    "active": "weather_mambatsv3_autocon_active_h96_h336_h720",
    "baseline": "weather_strong_baselines_h96_h336_h720",
}
BASELINES = ("PatchTST", "TimesNet", "iTransformer", "Mamba2")

def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def flag(command, name):
    return Path(command[command.index(name) + 1])

def read_phase(root, expected, phase_key):
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or int(validation.get("expected_jobs", -1)) != expected:
        raise ValueError(f"phase validation failed: {root}")
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    records = {}
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["job_id"] in records:
            raise ValueError("duplicate job")
        records[record["job_id"]] = record
    if len(records) != expected:
        raise ValueError("manifest coverage mismatch")
    rows = []
    for record in records.values():
        if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("provenance mismatch")
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        if config.get("skip_test") is not True or any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test boundary violated")
        logs = load(flag(record["command"], "--checkpoints") / result.name / "loss_logs.json")
        values = [float(value) for value in logs.get("val_loss", [])]
        if not values:
            raise ValueError("missing validation trajectory")
        model = str(config["model"])
        if phase_key == "plain" and model != "MambaTSv3": raise ValueError("plain model mismatch")
        if phase_key in {"null", "active"} and model != "MambaTSv3AutoCon": raise ValueError("AutoCon model mismatch")
        if phase_key == "active" and float(config.get("AutoCon_lambda", -1)) != 0.001: raise ValueError("active lambda mismatch")
        if phase_key == "null" and float(config.get("AutoCon_lambda", -1)) != 0.0: raise ValueError("null lambda mismatch")
        if phase_key in {"null", "active"} and config.get("AutoCon") is not True: raise ValueError("AutoCon flag mismatch")
        if phase_key == "active" and not logs.get("autocon_loss"): raise ValueError("AutoCon telemetry missing")
        rows.append({"phase": phase_key, "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "model": model, "mse": min(values), "elapsed_sec": float(record["elapsed_sec"])})
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for key, phase in PHASES.items():
        rows.extend(read_phase(args.artifact_root / phase, 36 if key == "baseline" else 9, key))
    index = {(row["phase"], row["horizon"], row["seed"], row["model"]): row for row in rows}
    summaries = []
    for horizon in HORIZONS:
        for treatment, reference, label in (("active", "null", "active_vs_null"), ("null", "plain", "null_vs_plain")):
            deltas = [100.0 * (index[(treatment, horizon, seed, "MambaTSv3AutoCon")]["mse"] - index[(reference, horizon, seed, "MambaTSv3AutoCon" if reference != "plain" else "MambaTSv3")]["mse"]) / index[(reference, horizon, seed, "MambaTSv3AutoCon" if reference != "plain" else "MambaTSv3")]["mse"] for seed in SEEDS]
            summaries.append({"horizon": horizon, "contrast": label, "mean_change_pct": statistics.mean(deltas), "wins": sum(value < 0 for value in deltas), "n": len(deltas)})
        for baseline in BASELINES:
            deltas = [100.0 * (index[("active", horizon, seed, "MambaTSv3AutoCon")]["mse"] - index[("baseline", horizon, seed, baseline)]["mse"]) / index[("baseline", horizon, seed, baseline)]["mse"] for seed in SEEDS]
            summaries.append({"horizon": horizon, "contrast": "active_vs_" + baseline, "mean_change_pct": statistics.mean(deltas), "wins": sum(value < 0 for value in deltas), "n": len(deltas)})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    lines = ["# Weather MambaTS v3 Multi-Horizon Screen", "", "Validation-only; test metrics are withheld.", "", "| Horizon | Contrast | Mean relative change | Wins / 3 |", "| ---: | --- | ---: | ---: |"]
    for item in summaries:
        lines.append(f"| {item['horizon']} | {item['contrast']} | {item['mean_change_pct']:+.3f}% | {item['wins']}/{item['n']} |")
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "summary_rows": len(summaries), "test_metrics_produced": False})

if __name__ == "__main__":
    main()
