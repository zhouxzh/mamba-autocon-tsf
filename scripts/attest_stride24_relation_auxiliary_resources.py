#!/usr/bin/env python3
"""Attest stride-24 relation-auxiliary resource records to frozen phases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("electricity", "traffic")
MODELS = ("MambaTSv3AutoCon", "DLinear", "PatchTST", "iTransformer", "Mamba1")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def phase_configs(root: Path, phase: str, expected_jobs: int, models: set[str]) -> tuple[Path, set[Path], str]:
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load(root / "provenance.json")
    snapshot = Path(str(provenance.get("source_snapshot", ""))).resolve()
    source_hash = str(provenance.get("source_tree_sha256", ""))
    if not snapshot.is_dir() or not source_hash:
        raise ValueError(f"invalid provenance: {root}")
    configs, seen = set(), set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        result = Path(record["expected_artifact"]).parent
        config = (result / "config.json").resolve()
        payload = load(config)
        if payload.get("data") not in DATASETS or payload.get("model") not in models or int(payload.get("pred_len", -1)) != 96:
            raise ValueError(f"invalid phase coordinate: {job_id}")
        configs.add(config)
    if len(configs) != expected_jobs:
        raise ValueError(f"phase coverage mismatch: {root}")
    return snapshot, configs, source_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot, candidate_configs, candidate_hash = phase_configs(args.candidate_root, "sparse_candidate_h96_validation", 6, {"MambaTSv3AutoCon"})
    baseline_snapshot, baseline_configs, baseline_hash = phase_configs(args.baseline_root, "sparse_baselines_h96_validation", 24, set(MODELS[1:]))
    if candidate_hash != baseline_hash or snapshot != baseline_snapshot:
        raise ValueError("candidate/baseline immutable sources differ")
    resource = load(args.resource_json)
    protocol = resource.get("benchmark_protocol", {})
    required = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3361, "include_representation": False, "relation_loss_included": False}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("resource protocol mismatch")
    all_configs = candidate_configs | baseline_configs
    rows, seen = [], set()
    for record in resource.get("resource_records", []):
        dataset, model = str(record.get("dataset")), str(record.get("model"))
        key = (dataset, model)
        if dataset not in DATASETS or model not in MODELS or key in seen:
            raise ValueError(f"unexpected resource coordinate: {key}")
        seen.add(key)
        source = record.get("source", {})
        config = Path(str(source.get("config_path", ""))).resolve()
        if config not in all_configs or digest(config) != source.get("config_sha256"):
            raise ValueError(f"resource config is not manifest-bound: {key}")
        if Path(str(record.get("source_root", ""))).resolve() != snapshot:
            raise ValueError(f"resource source root mismatch: {key}")
        source_file = snapshot / str(source.get("model_source_file", ""))
        if not source_file.is_file() or digest(source_file) != source.get("model_source_sha256"):
            raise ValueError(f"model source mismatch: {key}")
        gpu, parameters = record.get("gpu_inference", {}), record.get("parameters", {})
        if gpu.get("status") != "ok":
            raise ValueError(f"resource benchmark failed: {key}")
        rows.append({"dataset": dataset, "model": model, "latency_ms_median": float(gpu["latency_ms_median"]), "parameter_trainable": int(parameters["trainable"]), "peak_allocated_mb": float(gpu["steady_state_peak_allocated_mb"])})
    expected = {(dataset, model) for dataset in DATASETS for model in MODELS}
    if seen != expected:
        raise ValueError("incomplete resource matrix")
    lookup = {(row["dataset"], row["model"]): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Stride-24 Relation-Auxiliary Resource Attestation", "",
        "Every record is bound to a successful frozen validation-manifest config, the shared immutable source snapshot, and its matching model-source hash. Protocol: RTX 5090 D, FP32, device-resident synthetic inputs, `B=32,L=96,H=96`, 25 warm-up and 100 timed prediction-only forwards. AutoCon relation loss and returned representations are excluded.", "",
        "| Dataset | Candidate ms | DLinear | PatchTST | iTransformer | Mamba-1 | Candidate speed boundary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset in DATASETS:
        values = {model: lookup[(dataset, model)]["latency_ms_median"] for model in MODELS}
        faster = [model for model in MODELS[1:] if values["MambaTSv3AutoCon"] < values[model]]
        slower = [model for model in MODELS[1:] if values["MambaTSv3AutoCon"] > values[model]]
        lines.append("| {dataset} | {candidate:.4f} | {dlinear:.4f} | {patch:.4f} | {itrans:.4f} | {mamba1:.4f} | faster than {faster}; slower than {slower} |".format(dataset=dataset, candidate=values["MambaTSv3AutoCon"], dlinear=values["DLinear"], patch=values["PatchTST"], itrans=values["iTransformer"], mamba1=values["Mamba1"], faster=", ".join(faster), slower=", ".join(slower)))
    lines.extend(["", f"Immutable source tree SHA-256: `{candidate_hash}`. This is a fixed-shape resource measurement, not training throughput, end-to-end latency, or a universal speed claim."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": candidate_hash, "test_metrics_produced": False, "protocol": required}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
