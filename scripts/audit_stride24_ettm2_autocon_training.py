#!/usr/bin/env python3
"""Audit train-only AutoCon participation in the ETTm2 stride-24 candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite(values: object, key: str, job_id: str) -> list[float]:
    if not isinstance(values, list) or not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"invalid {key}: {job_id}")
    return [float(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validation, provenance = load(args.root / "phase_validation.json"), load(args.root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != "stride24_candidate_h96_validation" or validation.get("expected_jobs") != 3 or not source:
        raise ValueError("invalid candidate phase")
    rows, seen = [], set()
    for raw in (args.root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        result = Path(record["expected_artifact"]).parent
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source or any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"source/test boundary violation: {job_id}")
        config = load(result / "config.json")
        if config.get("data") != "ETTm2" or config.get("model") != "MambaTSv3AutoCon" or int(config.get("seed", -1)) not in (3401, 3402, 3403) or config.get("skip_test") is not True or config.get("AutoCon") is not True or config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False or int(config.get("patch_len", -1)) != 24 or int(config.get("patch_stride", -1)) != 24 or float(config.get("AutoCon_lambda", 0.0)) != 0.001:
            raise ValueError(f"candidate contract mismatch: {job_id}")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        logs = load(checkpoint / "loss_logs.json")
        auto, mse, total, effective = (finite(logs.get(key), key, job_id) for key in ("autocon_loss", "mse_loss", "train_loss", "effective_lambda"))
        if not len(auto) == len(mse) == len(total) == len(effective) or any(value != 0.001 for value in effective) or not any(value > 0 for value in auto):
            raise ValueError(f"objective contract mismatch: {job_id}")
        residual = max(abs(total_i - (mse_i + 0.001 * auto_i)) for total_i, mse_i, auto_i in zip(total, mse, auto))
        if residual > 1e-6:
            raise ValueError(f"loss identity mismatch: {job_id}")
        rows.append({"dataset": "ETTm2", "seed": int(config["seed"]), "epochs": len(auto), "autocon_loss_first": auto[0], "autocon_loss_last": auto[-1], "relative_change_pct": 100 * (auto[-1] / auto[0] - 1), "weighted_share_pct": 100 * statistics.fmean(0.001 * value / total_i for value, total_i in zip(auto, total)), "max_identity_residual": residual})
    if len(rows) != 3:
        raise ValueError("coverage mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"dataset": "ETTm2", "n": len(rows), "autocon_loss_first": statistics.fmean(row["autocon_loss_first"] for row in rows), "autocon_loss_last": statistics.fmean(row["autocon_loss_last"] for row in rows), "relative_change_pct": statistics.fmean(row["relative_change_pct"] for row in rows), "weighted_share_pct": statistics.fmean(row["weighted_share_pct"] for row in rows), "max_identity_residual": max(row["max_identity_residual"] for row in rows)}
    with (args.output_dir / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    lines = ["# Stride-24 ETTm2 AutoCon Training Audit", "", "Read-only audit of three candidate jobs. It verifies finite nonzero contrastive loss, lambda=.001, and `train_loss = mse_loss + lambda * autocon_loss`; no test artifacts were read or produced.", "", "| Dataset | Runs | First loss | Last loss | Relative change | Weighted share | Max identity residual |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |", "| {dataset} | {n} | {autocon_loss_first:.6f} | {autocon_loss_last:.6f} | {relative_change_pct:+.3f}% | {weighted_share_pct:.3f}% | {max_identity_residual:.2e} |".format(**summary), "", f"Immutable source tree SHA-256: `{source}`. Objective participation does not establish a causal accuracy gain."]
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": len(rows), "source_tree_sha256": source, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
