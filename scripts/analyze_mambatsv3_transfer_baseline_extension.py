#!/usr/bin/env python3
"""Strict readout for the validation-only MambaTS v3 transfer extension."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

MODELS = ("DLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2", "MambaTSv3AutoCon")
PHASES = {
    "transfer_baselines_weather_exchange_h96_h336": (72, {"weather": (96, 336), "exchange_rate": (96, 336)}),
    "transfer_baselines_ili_h24_h36_h48_h60": (72, {"national_illness": (24, 36, 48, 60)}),
}
SEEDS = (2021, 2022, 2023)


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase_rows(root: Path, expected_count: int, coordinate_spec: dict[str, tuple[int, ...]]):
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected_count:
        raise ValueError("phase validation failed: {}".format(root))
    provenance = load(root / "provenance.json")
    source = provenance.get("source_tree_sha256")
    records = {}
    for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("job_id") in records or record.get("status") != "ok":
            raise ValueError("invalid manifest row")
        records[record["job_id"]] = record
    if len(records) != expected_count:
        raise ValueError("manifest coverage mismatch")
    output = []
    for record in records.values():
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source snapshot mismatch")
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        if config.get("skip_test") is not True or any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test boundary violated")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        logs = load(checkpoint / "loss_logs.json")
        values = [float(value) for value in logs.get("val_loss", [])]
        if not values:
            raise ValueError("missing validation loss")
        model = str(config["model"])
        if model not in MODELS:
            raise ValueError("unexpected model: {}".format(model))
        if model == "MambaTSv3AutoCon":
            if config.get("AutoCon") is not True or float(config.get("AutoCon_lambda", -1)) != 0.001 or not logs.get("autocon_loss"):
                raise ValueError("AutoCon candidate contract mismatch")
        output.append({"dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "model": model, "best_val_mse": min(values), "elapsed_sec": float(record["elapsed_sec"])})
    expected = {(d, h, s, m) for d, hs in coordinate_spec.items() for h in hs for s in SEEDS for m in MODELS}
    found = {(r["dataset"], r["horizon"], r["seed"], r["model"]) for r in output}
    if found != expected:
        raise ValueError("coordinate mismatch")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for phase, (count, spec) in PHASES.items():
        rows.extend(phase_rows(args.artifact_root / phase, count, spec))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["horizon"], row["model"])].append(row)
    summaries = []
    for dataset, horizon in sorted({(r["dataset"], r["horizon"]) for r in rows}):
        means = {model: statistics.fmean(r["best_val_mse"] for r in grouped[(dataset, horizon, model)]) for model in MODELS}
        candidate_rows = {r["seed"]: r for r in grouped[(dataset, horizon, "MambaTSv3AutoCon")]}
        winner = min(means, key=means.get)
        for baseline in MODELS:
            base_rows = {r["seed"]: r for r in grouped[(dataset, horizon, baseline)]}
            changes = [100.0 * (candidate_rows[s]["best_val_mse"] - base_rows[s]["best_val_mse"]) / base_rows[s]["best_val_mse"] for s in SEEDS]
            summaries.append({"dataset": dataset, "horizon": horizon, "baseline": baseline, "winner": winner, "candidate_mse_mean": means["MambaTSv3AutoCon"], "baseline_mse_mean": means[baseline], "mean_change_pct": statistics.fmean(changes), "candidate_wins": sum(x < 0 for x in changes), "n": len(changes), "candidate_elapsed_mean_sec": statistics.fmean(r["elapsed_sec"] for r in candidate_rows.values()), "baseline_elapsed_mean_sec": statistics.fmean(r["elapsed_sec"] for r in base_rows.values())})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "paired_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    lines = ["# MambaTS v3 Transfer Baseline Extension", "", "Validation-only. No test metric files were produced.", "", "| Dataset | H | Best mean model | Candidate MSE | Baseline | Baseline MSE | Change | Wins / 3 |", "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for item in summaries:
        lines.append("| {dataset} | {horizon} | {winner} | {candidate_mse_mean:.6f} | {baseline} | {baseline_mse_mean:.6f} | {mean_change_pct:+.3f}% | {candidate_wins}/{n} |".format(**item))
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "summary_rows": len(summaries), "test_metrics_produced": False})


if __name__ == "__main__":
    main()
