#!/usr/bin/env python3
"""Audit AutoCon participation in the stride-24 high-dimensional candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3361, 3362, 3363)
TEST_ARTIFACTS = (
    "metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON object expected: {}".format(path))
    return payload


def _finite_list(payload: dict, key: str, job_id: str) -> list[float]:
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError("missing {}: {}".format(key, job_id))
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("non-finite {}: {}".format(key, job_id))
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    validation = _load(root / "phase_validation.json")
    provenance = _load(root / "provenance.json")
    source_tree = str(provenance.get("source_tree_sha256", ""))
    if (
        validation.get("valid") is not True
        or validation.get("phase") != "sparse_candidate_h96_validation"
        or validation.get("expected_jobs") != 6
        or not source_tree
    ):
        raise ValueError("invalid stride-24 high-dimensional candidate phase")

    manifest_path = root / "logs" / "job_manifest.jsonl"
    rows = []
    seen = set()
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row: {}".format(job_id))
        seen.add(job_id)
        if (
            record.get("source_tree_sha256_before") != source_tree
            or record.get("source_tree_sha256_after") != source_tree
        ):
            raise ValueError("source drift: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        if any((result / artifact).exists() for artifact in TEST_ARTIFACTS):
            raise ValueError("test artifact found: {}".format(job_id))
        config = _load(result / "config.json")
        if (
            config.get("model") != "MambaTSv3AutoCon"
            or config.get("data") not in DATASETS
            or int(config.get("seed", -1)) not in SEEDS
            or config.get("skip_test") is not True
            or config.get("AutoCon") is not True
            or config.get("mamba_ts_variant") != "v3"
            or int(config.get("patch_len", -1)) != 24
            or int(config.get("patch_stride", -1)) != 24
            or config.get("mamba_ts_repr_forecast_fusion") is not False
            or float(config.get("AutoCon_lambda", 0.0)) != 0.001
        ):
            raise ValueError("candidate contract mismatch: {}".format(job_id))
        command = record.get("command", [])
        if "--checkpoints" not in command:
            raise ValueError("checkpoint root missing from command: {}".format(job_id))
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        logs = _load(checkpoint / "loss_logs.json")
        auto = _finite_list(logs, "autocon_loss", job_id)
        mse = _finite_list(logs, "mse_loss", job_id)
        total = _finite_list(logs, "train_loss", job_id)
        effective = _finite_list(logs, "effective_lambda", job_id)
        if (
            not (len(auto) == len(mse) == len(total) == len(effective))
            or any(value != 0.001 for value in effective)
            or not any(value > 0.0 for value in auto)
        ):
            raise ValueError("AutoCon loss contract mismatch: {}".format(job_id))
        residual = max(
            abs(total_value - (mse_value + 0.001 * auto_value))
            for total_value, mse_value, auto_value in zip(total, mse, auto)
        )
        if residual > 1e-6:
            raise ValueError("loss identity mismatch: {}".format(job_id))
        rows.append({
            "dataset": config["data"],
            "seed": int(config["seed"]),
            "epochs": len(auto),
            "autocon_loss_first": auto[0],
            "autocon_loss_last": auto[-1],
            "relative_change_pct": 100.0 * (auto[-1] / auto[0] - 1.0),
            "weighted_share_pct": 100.0 * statistics.fmean(
                0.001 * auto_value / total_value
                for auto_value, total_value in zip(auto, total)
            ),
            "max_identity_residual": residual,
        })
    if len(rows) != 6 or {(row["dataset"], row["seed"]) for row in rows} != {
        (dataset, seed) for dataset in DATASETS for seed in SEEDS
    }:
        raise ValueError("candidate coverage mismatch")

    summary = []
    for dataset in DATASETS:
        current = [row for row in rows if row["dataset"] == dataset]
        summary.append({
            "dataset": dataset,
            "n": len(current),
            "autocon_loss_first": statistics.fmean(row["autocon_loss_first"] for row in current),
            "autocon_loss_last": statistics.fmean(row["autocon_loss_last"] for row in current),
            "relative_change_pct": statistics.fmean(row["relative_change_pct"] for row in current),
            "weighted_share_pct": statistics.fmean(row["weighted_share_pct"] for row in current),
            "max_identity_residual": max(row["max_identity_residual"] for row in current),
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("training_rows.csv", rows), ("training_summary.csv", summary)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = [
        "# Stride-24 High-Dimensional AutoCon Training Audit",
        "",
        "Read-only audit of six validation-only candidate jobs. It verifies AutoCon=True, finite nonzero contrastive loss, effective lambda=.001, and `train_loss = mse_loss + lambda * autocon_loss`. No test artifacts were read or produced.",
        "",
        "| Dataset | Runs | First loss | Last loss | Relative change | Weighted share | Max identity residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {dataset} | {n} | {autocon_loss_first:.6f} | {autocon_loss_last:.6f} | {relative_change_pct:+.3f}% | {weighted_share_pct:.3f}% | {max_identity_residual:.2e} |".format(**row)
        for row in summary
    )
    lines.extend([
        "",
        "Immutable source tree SHA-256: `{}`. Objective participation does not establish a causal accuracy gain.".format(source_tree),
    ])
    (output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "readout.json").write_text(json.dumps({
        "rows": len(rows),
        "source_tree_sha256": source_tree,
        "test_metrics_produced": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
