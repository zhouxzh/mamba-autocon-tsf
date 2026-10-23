#!/usr/bin/env python3
"""Audit AutoCon participation for light MambaTS v1 stride-24 training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "weather", "electricity", "traffic")
SEEDS = (3391, 3392, 3393)
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(values: object, name: str, job_id: str) -> list[float]:
    if not isinstance(values, list) or not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"invalid {name}: {job_id}")
    return [float(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validation, provenance = load(args.root / "phase_validation.json"), load(args.root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != "mambats_autocon_h96_validation" or validation.get("expected_jobs") != 12 or not source:
        raise ValueError("invalid candidate phase validation")
    rows, seen = [], set()
    for raw in (args.root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"test artifact found: {job_id}")
        config = load(result / "config.json")
        if config.get("model") != "MambaTSAutoCon" or config.get("data") not in DATASETS or int(config.get("seed", -1)) not in SEEDS or config.get("skip_test") is not True or config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v1" or config.get("mamba_ts_repr_forecast_fusion") is not False or float(config.get("AutoCon_lambda", 0.0)) != 0.001:
            raise ValueError(f"candidate contract mismatch: {job_id}")
        checkpoint = Path(record["command"][record["command"].index("--checkpoints") + 1]) / result.name
        logs = load(checkpoint / "loss_logs.json")
        auto, mse, total, effective = (finite(logs.get(key), key, job_id) for key in ("autocon_loss", "mse_loss", "train_loss", "effective_lambda"))
        if not len(auto) == len(mse) == len(total) == len(effective) or any(value != 0.001 for value in effective) or not any(value > 0 for value in auto):
            raise ValueError(f"loss contract mismatch: {job_id}")
        residual = max(abs(total_i - (mse_i + 0.001 * auto_i)) for total_i, mse_i, auto_i in zip(total, mse, auto))
        if residual > 1e-6:
            raise ValueError(f"loss identity mismatch: {job_id}")
        rows.append({"dataset": config["data"], "seed": int(config["seed"]), "epochs": len(auto), "autocon_loss_first": auto[0], "autocon_loss_last": auto[-1], "relative_change_pct": 100 * (auto[-1] / auto[0] - 1), "weighted_share_pct": 100 * statistics.fmean(0.001 * value / total_i for value, total_i in zip(auto, total)), "max_identity_residual": residual})
    if len(rows) != 12:
        raise ValueError("coverage mismatch")
    summary = []
    for dataset in DATASETS:
        current = [row for row in rows if row["dataset"] == dataset]
        summary.append({"dataset": dataset, "n": len(current), "autocon_loss_first": statistics.fmean(row["autocon_loss_first"] for row in current), "autocon_loss_last": statistics.fmean(row["autocon_loss_last"] for row in current), "relative_change_pct": statistics.fmean(row["relative_change_pct"] for row in current), "weighted_share_pct": statistics.fmean(row["weighted_share_pct"] for row in current), "max_identity_residual": max(row["max_identity_residual"] for row in current)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("training_rows.csv", rows), ("training_summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ["# Lightweight MambaTS AutoCon Training Audit", "", "Read-only audit of 12 validation jobs. It verifies the v1 MambaTS train-time AutoCon contract, finite nonzero contrastive loss, and `train_loss = mse_loss + lambda * autocon_loss`. No test artifacts were read or produced.", "", "| Dataset | Runs | First loss | Last loss | Relative change | Weighted share | Max identity residual |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend("| {dataset} | {n} | {autocon_loss_first:.6f} | {autocon_loss_last:.6f} | {relative_change_pct:+.3f}% | {weighted_share_pct:.3f}% | {max_identity_residual:.2e} |".format(**row) for row in summary)
    lines.extend(["", f"Immutable source tree SHA-256: `{source}`. Objective participation does not demonstrate a causal validation-MSE improvement."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": len(rows), "source_tree_sha256": source, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
