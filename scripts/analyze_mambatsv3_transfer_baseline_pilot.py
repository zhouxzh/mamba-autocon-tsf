#!/usr/bin/env python3
"""Strict validation-only reader for the MambaTS v3 transfer baseline pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = ("DLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2", "MambaTSv3AutoCon")
COMPARATORS = ("PatchTST", "TimesNet", "Mamba2")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rows(phase_root: Path) -> list[dict]:
    validation = load_json(phase_root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("expected_jobs") != 54:
        raise ValueError("phase validation is incomplete")
    provenance = load_json(phase_root / "provenance.json")
    source_hash = provenance.get("source_tree_sha256")
    records = {}
    for line in (phase_root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("job_id") in records:
            raise ValueError("duplicate manifest job")
        records[record["job_id"]] = record
    if len(records) != 54:
        raise ValueError("manifest coverage is incomplete")
    output = []
    for record in records.values():
        if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source_hash or record.get("source_tree_sha256_after") != source_hash:
            raise ValueError("manifest provenance mismatch")
        result_dir = Path(record["expected_artifact"]).parent
        config = load_json(result_dir / "config.json")
        if config.get("skip_test") is not True or (result_dir / "metrics.npy").exists():
            raise ValueError("test boundary violated")
        checkpoint_dir = Path(next(value for index, value in enumerate(record["command"]) if record["command"][index - 1] == "--checkpoints")) / result_dir.name
        logs = load_json(checkpoint_dir / "loss_logs.json")
        val = [float(value) for value in logs.get("val_loss", [])]
        if not val:
            raise ValueError("validation trajectory missing")
        model = str(config.get("model"))
        if model not in MODELS:
            raise ValueError("unexpected model")
        if model == "MambaTSv3AutoCon":
            if config.get("AutoCon") is not True or float(config.get("AutoCon_lambda", -1.0)) != 0.001:
                raise ValueError("AutoCon candidate contract mismatch")
            if not logs.get("autocon_loss"):
                raise ValueError("AutoCon telemetry missing")
        output.append({"dataset": config["data"], "horizon": int(config["pred_len"]), "seed": int(config["seed"]), "model": model, "best_val_mse": min(val), "elapsed_sec": float(record["elapsed_sec"])})
    expected = {(dataset, horizon, seed, model) for dataset, horizon in (("weather", 96), ("exchange_rate", 96), ("national_illness", 24)) for seed in (2021, 2022, 2023) for model in MODELS}
    found = {(r["dataset"], r["horizon"], r["seed"], r["model"]) for r in output}
    if found != expected:
        raise ValueError("coordinate mismatch")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    args = parser.parse_args()
    data = rows(args.phase_root.resolve())
    groups = defaultdict(list)
    for row in data:
        groups[(row["dataset"], row["horizon"], row["model"])].append(row)
    for dataset, horizon in (("weather", 96), ("exchange_rate", 96), ("national_illness", 24)):
        means = {model: statistics.mean(r["best_val_mse"] for r in groups[(dataset, horizon, model)]) for model in MODELS}
        candidate = means["MambaTSv3AutoCon"]
        winner = min(means, key=means.get)
        print(f"{dataset} H{horizon}: best={winner} {means[winner]:.6f}; MambaTSv3AutoCon={candidate:.6f}")
        for baseline in COMPARATORS:
            paired = []
            for seed in (2021, 2022, 2023):
                index = {(r["seed"], r["model"]): r for r in data if r["dataset"] == dataset and r["horizon"] == horizon}
                candidate_row, baseline_row = index[(seed, "MambaTSv3AutoCon")], index[(seed, baseline)]
                paired.append(100.0 * (candidate_row["best_val_mse"] - baseline_row["best_val_mse"]) / baseline_row["best_val_mse"])
            print(f"  versus {baseline}: {statistics.mean(paired):+.3f}% ({sum(value < 0 for value in paired)}/3 wins)")
    print("rows=54 test_metrics_produced=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
