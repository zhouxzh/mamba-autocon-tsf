#!/usr/bin/env python3
"""Bind cross-domain resource records to frozen exploratory manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("ETTh1", "weather")
MODELS = ("MambaTSv3AutoCon", "RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def phase_configs(root: Path, phase: str, expected: int) -> tuple[Path, set[Path]]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load(root / "provenance.json")
    snapshot = Path(str(provenance.get("source_snapshot", ""))).resolve()
    if not snapshot.is_dir() or not provenance.get("source_tree_sha256"):
        raise ValueError(f"invalid provenance: {root}")
    configs, seen = set(), set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest record: {job_id}")
        seen.add(job_id)
        result = Path(record["expected_artifact"]).parent
        config = result / "config.json"
        if not config.is_file():
            raise ValueError(f"missing config: {job_id}")
        configs.add(config.resolve())
    if len(configs) != expected:
        raise ValueError(f"manifest coverage mismatch: {root}")
    return snapshot, configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate_snapshot, candidate_configs = phase_configs(args.candidate_root, "candidate_h96_validation", 6)
    baseline_snapshot, baseline_configs = phase_configs(args.baseline_root, "baselines_h96_validation", 30)
    if candidate_snapshot != baseline_snapshot:
        raise ValueError("candidate/baseline snapshots differ")
    resource = load(args.resource_json)
    protocol = resource.get("benchmark_protocol", {})
    expected_protocol = {"run_benchmark": True, "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3331, "include_representation": False, "relation_loss_included": False}
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("resource protocol mismatch")
    all_configs = candidate_configs | baseline_configs
    records = resource.get("resource_records", [])
    if not isinstance(records, list) or len(records) != len(DATASETS) * len(MODELS):
        raise ValueError("resource record coverage mismatch")
    rows, seen = [], set()
    for record in records:
        dataset, model = str(record.get("dataset")), str(record.get("model"))
        key = (dataset, model)
        if key in seen or dataset not in DATASETS or model not in MODELS:
            raise ValueError(f"unexpected resource coordinate: {key}")
        seen.add(key)
        source = record.get("source", {})
        config = Path(str(source.get("config_path", ""))).resolve()
        if config not in all_configs or digest(config) != source.get("config_sha256"):
            raise ValueError(f"resource config is not manifest-bound: {key}")
        if Path(str(record.get("source_root", ""))).resolve() != candidate_snapshot:
            raise ValueError(f"resource source root mismatch: {key}")
        source_file = candidate_snapshot / str(source.get("model_source_file", ""))
        if not source_file.is_file() or digest(source_file) != source.get("model_source_sha256"):
            raise ValueError(f"resource model source mismatch: {key}")
        gpu = record.get("gpu_inference", {})
        parameters = record.get("parameters", {})
        if gpu.get("status") != "ok":
            raise ValueError(f"resource benchmark failed: {key}")
        rows.append({"dataset": dataset, "model": model, "latency_ms_median": float(gpu["latency_ms_median"]), "parameter_trainable": int(parameters["trainable"]), "peak_allocated_mb": float(gpu["steady_state_peak_allocated_mb"]), "forward_path": str(gpu["forward_path"])})
    if seen != {(dataset, model) for dataset in DATASETS for model in MODELS}:
        raise ValueError("incomplete resource coordinates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lookup = {(row["dataset"], row["model"]): row for row in rows}
    lines = ["# Cross-Domain Relation-Auxiliary-Only Resource Attestation", "", "Read-only attestation of the prediction-only resource bundle. Every record is bound to an `ok` manifest result config, its phase provenance snapshot, and the matching immutable model-source hash. The benchmark uses RTX 5090 D, FP32, device-resident synthetic inputs, `B=32,L=96,H=96`, 25 warm-up and 100 timed forwards; it excludes relation-loss computation and is not training throughput.", "", "| Dataset | Candidate ms | RLinear | PatchTST | TimesNet | iTransformer | Mamba2 | Candidate speed boundary |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for dataset in DATASETS:
        values = {model: lookup[(dataset, model)]["latency_ms_median"] for model in MODELS}
        faster = [model for model in MODELS[1:] if values["MambaTSv3AutoCon"] < values[model]]
        slower = [model for model in MODELS[1:] if values["MambaTSv3AutoCon"] > values[model]]
        lines.append("| {dataset} | {candidate:.4f} | {rlinear:.4f} | {patch:.4f} | {times:.4f} | {itrans:.4f} | {mamba2:.4f} | faster than {faster}; slower than {slower} |".format(dataset=dataset, candidate=values["MambaTSv3AutoCon"], rlinear=values["RLinear"], patch=values["PatchTST"], times=values["TimesNet"], itrans=values["iTransformer"], mamba2=values["Mamba2"], faster=", ".join(faster), slower=", ".join(slower)))
    lines.extend(["", f"Immutable source snapshot: `{candidate_snapshot}`. This attestation upgrades provenance linkage for an exploratory resource record; it does not convert validation accuracy or synthetic latency into held-out or universal claims."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_snapshot": str(candidate_snapshot), "test_metrics_produced": False, "protocol": expected_protocol}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
