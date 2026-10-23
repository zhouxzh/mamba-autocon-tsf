#!/usr/bin/env python3
"""Strict validation-only reader for the graph MambaTSv3 traffic screen.

The reader never loads test metrics. It requires complete manifests, immutable
phase validation, matching source snapshots, and the exact graph/AutoCon flags
before producing paired validation summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = ("metr_la", "pems_bay")
SEEDS = (2511, 2512, 2513)
CHANNELS = {"metr_la": 207, "pems_bay": 325}
GRAPH_PATHS = {
    "metr_la": "dataset/METR-LA/adjacency.npz",
    "pems_bay": "dataset/PEMS-BAY/adjacency.npz",
}
GRAPH_HASHES = {
    "metr_la": "97cfc5c63ab90bc6517108881f1aa3c26c62fbc548584b4ec76b28b0e16eac89",
    "pems_bay": "3567957c59f994a678526725b810cf29031affe03dca99c72d0c4eba9e8761c7",
}
EXTERNAL_MODELS = ("DLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json")
PHASES = {
    "graph_off": "mambatsv3_autocon_graph_off_h12",
    "graph_on": "mambatsv3_autocon_graph_on_h12",
    "graph_dlinear": "graphdlinear_h12",
    "external": "external_context_h12",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path, phase: str) -> list[dict]:
    path = root / phase / "logs" / "job_manifest.jsonl"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _best_validation_loss(record: dict) -> tuple[Path, dict, float]:
    if record.get("status") != "ok" or record.get("returncode") != 0:
        raise ValueError(f"failed job: {record.get('job_id')}")
    before = record.get("source_tree_sha256_before")
    if not before or before != record.get("source_tree_sha256_after"):
        raise ValueError(f"source snapshot drift: {record.get('job_id')}")
    artifact = Path(str(record.get("expected_artifact", ""))).resolve()
    result_dir = artifact.parent
    marker = result_dir / "training_complete.json"
    if not marker.is_file():
        raise ValueError(f"missing validation completion marker: {result_dir}")
    if any((result_dir / name).exists() for name in TEST_ARTIFACTS):
        raise ValueError(f"test artifact present in validation-only phase: {result_dir}")
    command = record.get("command", [])
    if "--checkpoints" not in command:
        raise ValueError(f"checkpoint path missing: {record.get('job_id')}")
    loss_path = Path(command[command.index("--checkpoints") + 1]) / result_dir.name / "loss_logs.json"
    losses = [float(value) for value in _json(loss_path).get("val_loss", [])]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError(f"invalid validation losses: {loss_path}")
    return result_dir, _json(result_dir / "config.json"), min(losses)


def _validate_phase_validation(root: Path, phase: str, expected_jobs: int, records: list[dict]) -> str:
    path = root / phase / "phase_validation.json"
    if not path.is_file():
        raise ValueError(f"missing phase validation: {path}")
    payload = _json(path)
    frozen_launcher_format = (
        payload.get("schema_version") == 1
        and payload.get("kind") == "exploratory_frozen_phase_validation"
    )
    strict_validator_format = (
        "schema_version" not in payload
        and isinstance(payload.get("provenance"), str)
        and payload.get("legacy_unknown_provenance") is False
    )
    if not (frozen_launcher_format or strict_validator_format):
        raise ValueError(f"phase validation schema mismatch: {path}")
    if payload.get("valid") is not True or payload.get("phase") != phase:
        raise ValueError(f"phase validation is not valid: {path}")
    if payload.get("expected_jobs") != expected_jobs or payload.get("manifest_unique_jobs") != expected_jobs:
        raise ValueError(f"phase validation coverage mismatch: {path}")
    if payload.get("malformed_manifest_lines") != 0 or payload.get("status_counts") != {"ok": expected_jobs}:
        raise ValueError(f"phase validation status mismatch: {path}")
    manifest = root / phase / "logs" / "job_manifest.jsonl"
    if payload.get("manifest_sha256") != _sha256(manifest):
        raise ValueError(f"phase validation manifest hash mismatch: {path}")
    source_hashes = {str(record.get("source_tree_sha256_before")) for record in records}
    if frozen_launcher_format:
        source_hash = payload.get("source_tree_sha256")
    else:
        provenance_path = Path(payload["provenance"])
        if not provenance_path.is_file():
            raise ValueError(f"phase validation provenance missing: {provenance_path}")
        source_hash = _json(provenance_path).get("source_tree_sha256")
    if len(source_hashes) != 1 or source_hash not in source_hashes:
        raise ValueError(f"phase validation source hash mismatch: {path}")
    return str(source_hash)


def _validate_config(config: dict, phase_key: str, dataset: str, seed: int, model: str) -> None:
    expected = {
        "data": dataset, "model": model, "features": "M", "seq_len": 96,
        "pred_len": 12, "enc_in": CHANNELS[dataset], "dec_in": CHANNELS[dataset],
        "c_out": CHANNELS[dataset], "run_seed": seed, "skip_test": True,
    }
    for key, value in expected.items():
        observed = config.get(key, config.get("seed") if key == "run_seed" else None)
        if observed != value:
            raise ValueError(f"{phase_key} config mismatch {key}: expected {value!r}, got {observed!r}")
    graph_enabled = bool(config.get("mamba_ts_graph_spatial", False))
    if phase_key in {"graph_on", "graph_dlinear"} and not graph_enabled:
        raise ValueError(f"graph flag missing in {phase_key}: {dataset}/{seed}")
    if phase_key in {"graph_off", "external"} and graph_enabled:
        raise ValueError(f"graph flag leaked into {phase_key}: {dataset}/{seed}")
    if phase_key in {"graph_off", "graph_on"}:
        if config.get("AutoCon") is not True:
            raise ValueError(f"AutoCon missing in {phase_key}: {dataset}/{seed}")
        if config.get("mamba_ts_variant") != "v3":
            raise ValueError(f"MambaTS variant mismatch in {phase_key}: {dataset}/{seed}")
    if phase_key == "graph_dlinear" and config.get("AutoCon") is not False:
        raise ValueError(f"AutoCon leaked into GraphDLinear: {dataset}/{seed}")
    if phase_key in {"graph_on", "graph_dlinear"}:
        expected_path = GRAPH_PATHS[dataset]
        expected_hash = GRAPH_HASHES[dataset]
        observed_path = Path(str(config.get("mamba_ts_graph_path", "")))
        if not observed_path.is_absolute():
            observed_path = REPO_ROOT / observed_path
        if observed_path.resolve() != (REPO_ROOT / expected_path).resolve():
            raise ValueError(f"graph path mismatch: expected {expected_path}, got {config.get('mamba_ts_graph_path')}")
        if str(config.get("mamba_ts_graph_expected_sha256", "")) != expected_hash:
            raise ValueError(f"graph hash mismatch for {dataset}")


def _read_phase(root: Path, phase_key: str, models: tuple[str, ...], expected_graph_jobs: int) -> tuple[list[dict], set[str]]:
    phase = PHASES[phase_key]
    expected = {(dataset, model, seed) for dataset in DATASETS for model in models for seed in SEEDS}
    rows, seen, hashes = [], set(), set()
    records = _manifest(root, phase)
    validation_hash = _validate_phase_validation(root, phase, expected_graph_jobs, records)
    for record in records:
        coordinate = (str(record.get("dataset")), str(record.get("model")), int(record.get("seed")))
        if coordinate not in expected or coordinate in seen:
            raise ValueError(f"invalid or duplicate coordinate in {phase}: {coordinate}")
        result_dir, config, loss = _best_validation_loss(record)
        dataset, model, seed = coordinate
        _validate_config(config, phase_key, dataset, seed, model)
        if phase_key in {"graph_on", "graph_dlinear"}:
            if not config.get("mamba_ts_graph_path") or not config.get("mamba_ts_graph_expected_sha256"):
                raise ValueError(f"graph contract missing from config: {result_dir}")
        seen.add(coordinate)
        hashes.add(str(record["source_tree_sha256_before"]))
        rows.append({"phase": phase_key, "dataset": dataset, "model": model, "seed": seed,
                     "best_val_mse": loss, "result_dir": str(result_dir),
                     "elapsed_sec": record.get("elapsed_sec")})
    if seen != expected:
        raise ValueError(f"incomplete coverage for {phase}: expected {len(expected)}, got {len(seen)}")
    if len(rows) != expected_graph_jobs:
        raise ValueError(f"unexpected job count for {phase}: {len(rows)}")
    if hashes != {validation_hash}:
        raise ValueError(f"phase validation source hash differs from manifest: {phase}")
    return rows, hashes


def _paired(left: list[dict], right: list[dict]) -> list[dict]:
    lhs = {(row["dataset"], row["seed"]): row for row in left}
    rhs = {(row["dataset"], row["seed"]): row for row in right}
    if set(lhs) != set(rhs):
        raise ValueError("paired coordinates do not match")
    return [
        {"dataset": key[0], "seed": key[1], "treatment_mse": lhs[key]["best_val_mse"],
         "reference_mse": rhs[key]["best_val_mse"],
         "relative_change_pct": 100.0 * (lhs[key]["best_val_mse"] / rhs[key]["best_val_mse"] - 1.0)}
        for key in sorted(lhs)
    ]


def _summary(rows: list[dict]) -> list[dict]:
    by_phase = {phase: [row for row in rows if row["phase"] == phase] for phase in PHASES}
    summaries = []
    for dataset in DATASETS:
        def values(phase: str, model: str | None = None) -> list[float]:
            return [row["best_val_mse"] for row in by_phase[phase]
                    if row["dataset"] == dataset and (model is None or row["model"] == model)]
        off = values("graph_off")
        on = values("graph_on")
        graph_linear = values("graph_dlinear")
        baseline_means = {model: statistics.fmean(values("external", model)) for model in EXTERNAL_MODELS}
        strongest = min(baseline_means, key=baseline_means.get)
        paired_off_on = _paired(
            [row for row in by_phase["graph_on"] if row["dataset"] == dataset],
            [row for row in by_phase["graph_off"] if row["dataset"] == dataset],
        )
        max_degradation = max(item["relative_change_pct"] for item in paired_off_on)
        on_mean, off_mean = statistics.fmean(on), statistics.fmean(off)
        summary = {
            "dataset": dataset, "horizon": 12, "graph_on_mean_val_mse": on_mean,
            "graph_off_mean_val_mse": off_mean, "graph_dlinear_mean_val_mse": statistics.fmean(graph_linear),
            "graph_on_vs_off_percent": 100.0 * (on_mean / off_mean - 1.0),
            "graph_on_wins_vs_off": sum(item["relative_change_pct"] < 0 for item in paired_off_on),
            "max_paired_degradation_percent": max_degradation,
            "strongest_external_model": strongest,
            "strongest_external_mean_val_mse": baseline_means[strongest],
            "graph_on_vs_strongest_external_percent": 100.0 * (on_mean / baseline_means[strongest] - 1.0),
            "passes_graph_gate": on_mean <= off_mean * 0.99 and sum(item["relative_change_pct"] < 0 for item in paired_off_on) >= 2 and max_degradation <= 2.0,
            "passes_topology_attribution": on_mean <= statistics.fmean(graph_linear) and on_mean <= off_mean,
            "passes_external_gate": on_mean <= baseline_means[strongest] * 1.02,
        }
        summary["passes_dataset_gate"] = all(summary[key] for key in ("passes_graph_gate", "passes_topology_attribution", "passes_external_gate"))
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    rows, hashes = [], set()
    for phase_key, models, count in (
        ("graph_off", ("MambaTSv3AutoCon",), 6),
        ("graph_on", ("MambaTSv3AutoCon",), 6),
        ("graph_dlinear", ("GraphDLinear",), 6),
        ("external", EXTERNAL_MODELS, 30),
    ):
        phase_rows, phase_hashes = _read_phase(root, phase_key, models, count)
        rows.extend(phase_rows)
        hashes |= phase_hashes
    if len(hashes) != 1:
        raise ValueError(f"source snapshot differs across phases: {sorted(hashes)}")
    summary = _summary(rows)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "summary.csv", summary)
    payload = {"schema_version": 1, "test_metrics_produced": False,
               "source_snapshot": next(iter(hashes)), "summary": summary}
    (output / "readout.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Graph MambaTSv3 AutoCon H12 readout", "", "Validation-only; test metrics were not read.", "", "| Dataset | Graph on | Graph off | On vs off | Wins | GraphDLinear | Strongest external | Gate |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        lines.append("| {dataset} | {graph_on_mean_val_mse:.6f} | {graph_off_mean_val_mse:.6f} | {graph_on_vs_off_percent:+.3f}% | {graph_on_wins_vs_off}/3 | {graph_dlinear_mean_val_mse:.6f} | {strongest_external_model} ({strongest_external_mean_val_mse:.6f}) | {passes_dataset_gate} |".format(**row))
    lines.extend(["", f"Source snapshot: `{next(iter(hashes))}`."])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": len(summary), "rows": len(rows), "passes_all": all(row["passes_dataset_gate"] for row in summary), "test_metrics_produced": False}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
