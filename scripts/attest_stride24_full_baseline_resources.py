#!/usr/bin/env python3
"""Attest the complete stride-24 H96 resource matrix across frozen phases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("electricity", "traffic")
MODELS = ("MambaTSv3AutoCon", "DLinear", "PatchTST", "iTransformer", "Mamba1", "TimesNet", "Mamba2")
PHASES = {
    "candidate": ("sparse_candidate_h96_validation", 6, {"MambaTSv3AutoCon"}),
    "initial": ("sparse_baselines_h96_validation", 24, {"DLinear", "PatchTST", "iTransformer", "Mamba1"}),
    "strong": ("strong_baselines_stride24_h96_validation", 12, {"TimesNet", "Mamba2"}),
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def phase_configs(root: Path, label: str) -> tuple[Path, set[Path], str]:
    phase, expected_jobs, models = PHASES[label]
    validation = load(root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected_jobs:
        raise ValueError(f"invalid phase validation: {label}")
    provenance = load(root / "provenance.json")
    snapshot, source_hash = Path(str(provenance.get("source_snapshot", ""))).resolve(), str(provenance.get("source_tree_sha256", ""))
    if not snapshot.is_dir() or not source_hash:
        raise ValueError(f"invalid provenance: {label}")
    configs, seen = set(), set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        config = (Path(record["expected_artifact"]).parent / "config.json").resolve()
        payload = load(config)
        if payload.get("data") not in DATASETS or payload.get("model") not in models or int(payload.get("pred_len", -1)) != 96:
            raise ValueError(f"invalid phase coordinate: {job_id}")
        configs.add(config)
    if len(configs) != expected_jobs:
        raise ValueError(f"phase coverage mismatch: {label}")
    return snapshot, configs, source_hash


def records_from(path: Path) -> list[dict]:
    resource = load(path)
    protocol = resource.get("benchmark_protocol", {})
    expected = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3361, "include_representation": False, "relation_loss_included": False}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError(f"resource protocol mismatch: {path}")
    return resource.get("resource_records", [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--initial-baseline-root", type=Path, required=True)
    parser.add_argument("--strong-baseline-root", type=Path, required=True)
    parser.add_argument("--initial-resource-json", type=Path, required=True)
    parser.add_argument("--strong-resource-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshots, hashes, all_configs = [], [], set()
    for label, root in (("candidate", args.candidate_root), ("initial", args.initial_baseline_root), ("strong", args.strong_baseline_root)):
        snapshot, configs, source_hash = phase_configs(root, label)
        snapshots.append(snapshot); hashes.append(source_hash); all_configs |= configs
    if len(set(hashes)) != 1:
        raise ValueError("phases do not share one immutable source")
    snapshot, source_hash = snapshots[0], hashes[0]
    rows, seen = [], set()
    for record in records_from(args.initial_resource_json) + records_from(args.strong_resource_json):
        dataset, model = str(record.get("dataset")), str(record.get("model")); key = (dataset, model)
        if dataset not in DATASETS or model not in MODELS or key in seen:
            raise ValueError(f"unexpected/duplicate resource coordinate: {key}")
        seen.add(key)
        source = record.get("source", {}); config = Path(str(source.get("config_path", ""))).resolve()
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
        raise ValueError("incomplete full resource matrix")
    lookup = {(row["dataset"], row["model"]): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Stride-24 Relation-Auxiliary Full Resource Attestation", "", "Every record is bound to a successful frozen validation-manifest config, shared immutable source snapshot, and matching model-source hash. Protocol: RTX 5090 D, FP32, device-resident synthetic inputs, `B=32,L=96,H=96`, 25 warm-up and 100 timed prediction-only forwards. AutoCon relation loss and returned representations are excluded.", "", "| Dataset | Candidate ms | DLinear | PatchTST | iTransformer | Mamba-1 | TimesNet | Mamba-2 | Candidate speed boundary |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for dataset in DATASETS:
        values = {model: lookup[(dataset, model)]["latency_ms_median"] for model in MODELS}
        faster = [model for model in MODELS[1:] if values["MambaTSv3AutoCon"] < values[model]]
        slower = [model for model in MODELS[1:] if values["MambaTSv3AutoCon"] > values[model]]
        lines.append("| {dataset} | {candidate:.4f} | {dlinear:.4f} | {patch:.4f} | {itrans:.4f} | {mamba1:.4f} | {times:.4f} | {mamba2:.4f} | faster than {faster}; slower than {slower} |".format(dataset=dataset, candidate=values["MambaTSv3AutoCon"], dlinear=values["DLinear"], patch=values["PatchTST"], itrans=values["iTransformer"], mamba1=values["Mamba1"], times=values["TimesNet"], mamba2=values["Mamba2"], faster=", ".join(faster), slower=", ".join(slower)))
    lines.extend(["", f"Immutable source tree SHA-256: `{source_hash}`. This is a fixed-shape resource measurement, not training throughput, end-to-end latency, or a universal speed claim."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": source_hash, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
