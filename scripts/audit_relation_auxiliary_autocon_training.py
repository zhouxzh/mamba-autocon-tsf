#!/usr/bin/env python3
"""Audit active train-only AutoCon representation learning in frozen matrices."""

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


def finite_series(payload: dict, key: str, job_id: str) -> list[float]:
    values = payload.get(key)
    if not isinstance(values, list) or not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"invalid {key}: {job_id}")
    return [float(value) for value in values]


def read_phase(root: Path, phase: str, expected_jobs: int, datasets: set[str], seeds: set[int]) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source or provenance.get("test_selection_allowed") is not False:
        raise ValueError(f"invalid provenance: {root}")
    rows, seen = [], set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest record: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        if any((result / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError(f"test boundary violation: {job_id}")
        config = load(result / "config.json")
        dataset, seed = str(config.get("data")), int(config.get("seed", -1))
        if dataset not in datasets or seed not in seeds or config.get("model") != "MambaTSv3AutoCon":
            raise ValueError(f"coordinate mismatch: {job_id}")
        if config.get("skip_test") is not True or config.get("AutoCon") is not True:
            raise ValueError(f"AutoCon/test contract mismatch: {job_id}")
        if config.get("mamba_ts_variant") != "v3" or config.get("mamba_ts_repr_forecast_fusion") is not False:
            raise ValueError(f"candidate architecture mismatch: {job_id}")
        lambda_value = float(config.get("AutoCon_lambda", config.get("autocon_lambda", 0.0)))
        if abs(lambda_value - 0.001) > 1e-12:
            raise ValueError(f"candidate lambda mismatch: {job_id}")
        command = record["command"]
        checkpoint = Path(command[command.index("--checkpoints") + 1]) / result.name
        logs = load(checkpoint / "loss_logs.json")
        auto, mse, total, effective = (finite_series(logs, key, job_id) for key in ("autocon_loss", "mse_loss", "train_loss", "effective_lambda"))
        if not (len(auto) == len(mse) == len(total) == len(effective)):
            raise ValueError(f"inconsistent epoch count: {job_id}")
        if any(abs(value - lambda_value) > 1e-12 for value in effective):
            raise ValueError(f"inactive or drifting AutoCon lambda: {job_id}")
        residual = max(abs(value - (base + weight * aux)) for value, base, weight, aux in zip(total, mse, effective, auto))
        if residual > 1e-6:
            raise ValueError(f"training loss does not include AutoCon objective: {job_id} ({residual})")
        rows.append({
            "matrix": root.parent.name,
            "dataset": dataset,
            "seed": seed,
            "epochs": len(auto),
            "autocon_lambda": lambda_value,
            "autocon_loss_first": auto[0],
            "autocon_loss_last": auto[-1],
            "autocon_loss_relative_change_pct": 100.0 * (auto[-1] / auto[0] - 1.0),
            "autocon_loss_mean": statistics.fmean(auto),
            "mean_weighted_autocon_fraction_pct": 100.0 * statistics.fmean(weight * aux / value for value, weight, aux in zip(total, effective, auto)),
            "max_loss_identity_residual": residual,
        })
    if len(rows) != expected_jobs:
        raise ValueError(f"coverage mismatch: {root}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--highdim-root", type=Path, required=True)
    parser.add_argument("--crossdomain-root", type=Path, required=True)
    parser.add_argument("--solar-root", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_a, highdim = read_phase(args.highdim_root, "relation_auxiliary_only_candidate_h96_validation", 6, {"electricity", "traffic"}, {3291, 3292, 3293})
    source_b, crossdomain = read_phase(args.crossdomain_root, "candidate_h96_validation", 6, {"ETTh1", "weather"}, {3331, 3332, 3333})
    if source_a != source_b:
        raise ValueError("candidate matrices use different source snapshots")
    rows = highdim + crossdomain
    if args.solar_root is not None:
        source_c, solar = read_phase(args.solar_root, "candidate_solar_h96_validation", 3, {"solar"}, {3341, 3342, 3343})
        if source_c != source_a:
            raise ValueError("Solar candidate uses a different source snapshot")
        rows.extend(solar)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "autocon_training_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_dataset = []
    for dataset in ("electricity", "traffic", "weather", "ETTh1", "solar"):
        current = [row for row in rows if row["dataset"] == dataset]
        if not current:
            continue
        by_dataset.append({
            "dataset": dataset,
            "n": len(current),
            "autocon_loss_first": statistics.fmean(row["autocon_loss_first"] for row in current),
            "autocon_loss_last": statistics.fmean(row["autocon_loss_last"] for row in current),
            "autocon_loss_relative_change_pct": statistics.fmean(row["autocon_loss_relative_change_pct"] for row in current),
            "mean_weighted_autocon_fraction_pct": statistics.fmean(row["mean_weighted_autocon_fraction_pct"] for row in current),
            "max_loss_identity_residual": max(row["max_loss_identity_residual"] for row in current),
        })
    with (args.output_dir / "autocon_training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_dataset[0]))
        writer.writeheader()
        writer.writerows(by_dataset)
    lines = ["# Relation-Auxiliary-Only AutoCon Training Audit", "", "This read-only audit verifies that every candidate run used `AutoCon=True`, returned a train-time representation, retained `AutoCon_lambda=.001`, produced finite nonzero contrastive losses, and recorded `train_loss = mse_loss + lambda * autocon_loss`. It does not turn the auxiliary-loss trajectory into an accuracy or causal claim.", "", "| Dataset | Seeds | First AutoCon loss | Last AutoCon loss | Relative change | Mean weighted objective share | Max identity residual |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in by_dataset:
        lines.append("| {dataset} | {n} | {autocon_loss_first:.6f} | {autocon_loss_last:.6f} | {autocon_loss_relative_change_pct:+.3f}% | {mean_weighted_autocon_fraction_pct:.3f}% | {max_loss_identity_residual:.2e} |".format(**row))
    lines.extend(["", f"All {len(rows)} runs share immutable source snapshot `{source_a}` and produced no test artifact. The representation is train-only in this architecture because prediction fusion is disabled by the paired mechanism result."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "readout.json").write_text(json.dumps({"rows": len(rows), "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "source_tree_sha256": source_a, "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
