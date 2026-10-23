#!/usr/bin/env python3
"""Strict reader for the latent-temporal Electricity/Traffic H96 validation plan.

The plan is validation-only.  This reader deliberately rejects any held-out
artifact before producing a table, and it requires every registered phase to
have passed ``validate_training_phase.py`` with a matching manifest/config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = ("electricity", "traffic")
SEEDS = (3401, 3402, 3403)
BASELINES = (
    "DLinear", "NLinear", "RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2",
)
PHASES = {
    "null": ("latent_temporal_null_h96_validation", 6, {"LatentTemporalMambaAutoCon"}),
    "active": ("latent_temporal_active_h96_validation", 6, {"LatentTemporalMambaAutoCon"}),
    "reference": ("mambatsv3_reference_h96_validation", 6, {"MambaTSv3AutoCon"}),
    "baselines": ("strong_baselines_h96_validation", 42, set(BASELINES)),
}
TEST_ARTIFACTS = (
    "metrics.npy", "original_metrics.npy", "stream_metrics.json", "stream_samples.npz",
    "pred.npy", "true.npy", "channel_metrics.npy",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON object required: {}".format(path))
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_dir(record: dict, result_dir: Path) -> Path:
    command = list(record.get("command", []))
    try:
        return Path(command[command.index("--checkpoints") + 1]) / result_dir.name
    except (ValueError, IndexError) as exc:
        raise ValueError("manifest checkpoint command is invalid") from exc


def _phase_rows(root: Path, label: str) -> tuple[str, list[dict]]:
    expected_phase, expected_count, allowed_models = PHASES[label]
    validation_path = root / "phase_validation.json"
    validation = _load(validation_path)
    if (
        validation.get("valid") is not True
        or validation.get("errors")
        or validation.get("phase") != expected_phase
        or validation.get("expected_jobs") != expected_count
    ):
        raise ValueError("invalid strict phase validation: {}".format(label))
    manifest_path = root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file() or validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("validated manifest changed or is missing: {}".format(label))
    provenance = _load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if not source:
        raise ValueError("invalid validation-only provenance: {}".format(label))
    config_hashes = validation.get("artifact_hashes_by_job")
    if not isinstance(config_hashes, dict):
        raise ValueError("phase validation lacks artifact hashes: {}".format(label))

    rows, seen = [], set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest row: {}".format(job_id))
        seen.add(job_id)
        if (
            record.get("source_tree_sha256_before") != source
            or record.get("source_tree_sha256_after") != source
        ):
            raise ValueError("source drift: {}".format(job_id))
        artifact = Path(str(record.get("expected_artifact", "")))
        result_dir = artifact.parent
        config_path = result_dir / "config.json"
        expected_hash = config_hashes.get(job_id, {}).get("config.json")
        if not config_path.is_file() or not expected_hash or _sha256(config_path) != expected_hash:
            raise ValueError("config is not validator-bound: {}".format(job_id))
        if not (result_dir / "training_complete.json").is_file():
            raise ValueError("training completion artifact missing: {}".format(job_id))
        if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
            raise ValueError("test artifact boundary violation: {}".format(job_id))
        config = _load(config_path)
        dataset = str(config.get("data", ""))
        model = str(config.get("model", ""))
        seed = int(config.get("seed", -1))
        if (
            dataset not in DATASETS or model not in allowed_models or seed not in SEEDS
            or int(config.get("pred_len", -1)) != 96 or config.get("skip_test") is not True
        ):
            raise ValueError("coordinate/model boundary violation: {}".format(job_id))
        if label in {"null", "active"}:
            expected_lambda = 0.0 if label == "null" else 0.001
            if config.get("AutoCon") is not True or not math.isclose(
                float(config.get("AutoCon_lambda", float("nan"))), expected_lambda,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("latent AutoCon contract mismatch: {}".format(job_id))
        elif label == "reference":
            if (
                config.get("AutoCon") is not True
                or config.get("mamba_ts_variant") != "v3"
                or config.get("mamba_ts_repr_forecast_fusion") is not False
                or not math.isclose(float(config.get("AutoCon_lambda", float("nan"))), 0.001, abs_tol=1e-12)
            ):
                raise ValueError("MambaTSv3 reference contract mismatch: {}".format(job_id))
        elif config.get("AutoCon") is not False:
            raise ValueError("external baseline isolation mismatch: {}".format(job_id))
        losses = _load(_checkpoint_dir(record, result_dir) / "loss_logs.json").get("val_loss", [])
        losses = [float(value) for value in losses]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("invalid validation trajectory: {}".format(job_id))
        rows.append({
            "phase": expected_phase, "dataset": dataset, "horizon": 96,
            "model": model, "seed": seed, "best_val_mse": min(losses),
            "elapsed_sec": float(record.get("elapsed_sec", float("nan"))),
        })
    if len(rows) != expected_count:
        raise ValueError("phase coverage mismatch: {}".format(label))
    return source, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired(active: list[dict], comparator: list[dict], label: str) -> dict:
    by_seed = {row["seed"]: row for row in comparator}
    if set(by_seed) != set(SEEDS):
        raise ValueError("incomplete paired comparator: {}".format(label))
    ratios = [row["best_val_mse"] / by_seed[row["seed"]]["best_val_mse"] for row in active]
    return {
        "comparator": label,
        "candidate_mse_mean": statistics.fmean(row["best_val_mse"] for row in active),
        "comparator_mse_mean": statistics.fmean(row["best_val_mse"] for row in comparator),
        "mean_ratio": statistics.fmean(ratios),
        "candidate_change_pct": 100.0 * (statistics.fmean(ratios) - 1.0),
        "candidate_wins": sum(value < 1.0 for value in ratios),
        "n": len(ratios),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.plan)
    if plan.get("status") != "exploratory_latent_temporal_mamba_highdim_autocon_validation":
        raise ValueError("unexpected plan status")
    roots = {
        "null": args.artifact_root / "null",
        "active": args.artifact_root / "active",
        "reference": args.artifact_root / "mambatsv3_reference",
        "baselines": args.artifact_root / "baselines",
    }
    sources, rows_by_phase = set(), {}
    for label, root in roots.items():
        source, rows = _phase_rows(root, label)
        sources.add(source)
        rows_by_phase[label] = rows
    if len(sources) != 1:
        raise ValueError("phases do not share one immutable source snapshot")
    all_rows = [row for rows in rows_by_phase.values() for row in rows]
    index = {(row["phase"], row["dataset"], row["model"], row["seed"]): row for row in all_rows}
    if len(index) != len(all_rows):
        raise ValueError("duplicate validation coordinate")

    summaries, gates = [], {}
    for dataset in DATASETS:
        active = [row for row in rows_by_phase["active"] if row["dataset"] == dataset]
        null = [row for row in rows_by_phase["null"] if row["dataset"] == dataset]
        reference = [row for row in rows_by_phase["reference"] if row["dataset"] == dataset]
        if len({len(active), len(null), len(reference)}) != 1 or len(active) != len(SEEDS):
            raise ValueError("candidate/control coverage mismatch: {}".format(dataset))
        null_summary = _paired(active, null, "LatentTemporalMambaAutoCon lambda=0")
        reference_summary = _paired(active, reference, "MambaTSv3AutoCon stride-24")
        summaries.extend([{**null_summary, "dataset": dataset}, {**reference_summary, "dataset": dataset}])
        baseline_summaries = []
        for model in BASELINES:
            baseline = [row for row in rows_by_phase["baselines"] if row["dataset"] == dataset and row["model"] == model]
            summary = _paired(active, baseline, model)
            summaries.append({**summary, "dataset": dataset})
            baseline_summaries.append(summary)
        strongest = min(baseline_summaries, key=lambda row: row["comparator_mse_mean"])
        gates[dataset] = {
            "autocon_vs_null_pass": bool(null_summary["candidate_change_pct"] < 0.0 and null_summary["candidate_wins"] >= 2),
            "external_2pct_gate_pass": bool(strongest["mean_ratio"] <= 1.02),
            "strongest_external_baseline": strongest["comparator"],
            "strongest_external_ratio": strongest["mean_ratio"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "validation_rows.csv", all_rows)
    _write_csv(args.output_dir / "paired_summary.csv", summaries)
    payload = {
        "plan_sha256": _sha256(args.plan), "source_tree_sha256": next(iter(sources)),
        "rows": len(all_rows), "paired_cells": len(summaries),
        "test_metrics_produced": False, "gates": gates,
    }
    (args.output_dir / "readout.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Latent Temporal Mamba Electricity / Traffic H96 Validation", "",
        "Validation-only, three fresh paired seeds, one immutable source snapshot. The reader rejects test artifacts and requires every phase validator/config binding before reporting this table.", "",
        "| Dataset | Comparator | Candidate MSE | Comparator MSE | Change | Wins |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append("| {dataset} | {comparator} | {candidate_mse_mean:.6f} | {comparator_mse_mean:.6f} | {candidate_change_pct:+.3f}% | {candidate_wins}/{n} |".format(**row))
    lines.extend(["", "## Gates", ""])
    for dataset, gate in gates.items():
        lines.append("- `{}`: AutoCon vs null `{}`; strongest external `{}` ratio `{:.4f}`; 2% gate `{}`.".format(dataset, gate["autocon_vs_null_pass"], gate["strongest_external_baseline"], gate["strongest_external_ratio"], gate["external_2pct_gate_pass"]))
    lines.extend(["", "This is validation evidence only. It cannot establish held-out accuracy, universal speed, or an AutoCon effect beyond the registered exact null control."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "paired_cells": len(summaries), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
