#!/usr/bin/env python3
"""Audit train-only AutoCon participation for Solar variable-axis Mamba jobs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle: return json.load(handle)


def finite(values: object, key: str) -> list[float]:
    if not isinstance(values, list) or not values or not all(math.isfinite(float(value)) for value in values): raise ValueError(f"invalid {key}")
    return [float(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    validation, provenance = load(args.root / "phase_validation.json"), load(args.root / "provenance.json"); source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != "variable_candidate_timemixer_validation" or validation.get("expected_jobs") != 3 or not source: raise ValueError("invalid phase")
    rows = []
    for raw in (args.root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); result = Path(record["expected_artifact"]).parent
        if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source or any((result / name).exists() for name in TEST_ARTIFACTS): raise ValueError("manifest/test boundary violation")
        config = load(result / "config.json")
        if config.get("data") != "solar" or config.get("model") != "NLinearVariableMambaAutoCon" or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True or config.get("AutoCon") is not True or config.get("mamba_variable_repr_fusion") is not True or config.get("mamba_acf_relation_loss") is not True or float(config.get("AutoCon_lambda", 0.0)) != 0.0001: raise ValueError("candidate contract mismatch")
        command = record["command"]; checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name; logs = load(checkpoint / "loss_logs.json")
        auto, mse, total, effective = (finite(logs.get(key), key) for key in ("autocon_loss", "mse_loss", "train_loss", "effective_lambda"))
        if not len(auto) == len(mse) == len(total) == len(effective) or any(value not in (0.00005, 0.0001) for value in effective) or not any(value > 0 for value in auto): raise ValueError("objective contract mismatch")
        residual = max(abs(train - (forecast + weight * contrast)) for train, forecast, contrast, weight in zip(total, mse, auto, effective))
        if residual > 1e-6: raise ValueError("loss identity mismatch")
        rows.append({"dataset": "Solar", "horizon": 96, "seed": int(config["seed"]), "autocon_loss_first": auto[0], "autocon_loss_last": auto[-1], "relative_change_pct": 100 * (auto[-1] / auto[0] - 1), "weighted_share_pct": 100 * statistics.fmean(weight * value / train for value, train, weight in zip(auto, total, effective)), "max_identity_residual": residual})
    if len(rows) != 3 or {row["seed"] for row in rows} != {3471, 3472, 3473}: raise ValueError("coverage mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_rows.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"dataset": "Solar", "horizon": 96, "n": len(rows), "autocon_loss_first": statistics.fmean(row["autocon_loss_first"] for row in rows), "autocon_loss_last": statistics.fmean(row["autocon_loss_last"] for row in rows), "relative_change_pct": statistics.fmean(row["relative_change_pct"] for row in rows), "weighted_share_pct": statistics.fmean(row["weighted_share_pct"] for row in rows), "max_identity_residual": max(row["max_identity_residual"] for row in rows)}
    with (args.output_dir / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    lines = ["# Solar Variable-Axis Mamba AutoCon Training Audit", "", "Read-only audit of three candidate jobs. It verifies finite nonzero relation loss, lambda=.0001, and `train_loss = mse_loss + lambda * autocon_loss`; no test artifacts were read or produced.", "", "| H | Runs | First loss | Last loss | Relative change | Weighted share | Max identity residual |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |", "| {horizon} | {n} | {autocon_loss_first:.6f} | {autocon_loss_last:.6f} | {relative_change_pct:+.3f}% | {weighted_share_pct:.3f}% | {max_identity_residual:.2e} |".format(**summary), "", f"Immutable source tree SHA-256: `{source}`. Objective participation does not establish a causal accuracy gain."]
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": len(rows), "source_tree_sha256": source, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "test_metrics_produced": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
