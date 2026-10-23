#!/usr/bin/env python3
"""Attest fixed-shape resource records for relation-auxiliary Mamba vs Mamba-1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("electricity", "traffic")
EXPECTED = {
    "candidate": ("relation_auxiliary_only_candidate_h96_validation", 6, "MambaTSv3AutoCon"),
    "baseline": ("mamba1_h96_validation", 6, "Mamba1"),
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


def phase_configs(root: Path, phase_key: str) -> tuple[Path, set[Path], str]:
    phase, expected_jobs, model = EXPECTED[phase_key]
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
            raise ValueError(f"invalid manifest record: {job_id}")
        seen.add(job_id)
        result = Path(record["expected_artifact"]).parent
        config = (result / "config.json").resolve()
        if not config.is_file():
            raise ValueError(f"missing config: {job_id}")
        payload = load(config)
        if payload.get("model") != model or payload.get("data") not in DATASETS or int(payload.get("pred_len", -1)) != 96:
            raise ValueError(f"phase coordinate mismatch: {job_id}")
        configs.add(config)
    if len(configs) != expected_jobs:
        raise ValueError(f"manifest coverage mismatch: {root}")
    return snapshot, configs, source_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_snapshot, candidate_configs, candidate_hash = phase_configs(args.candidate_root, "candidate")
    baseline_snapshot, baseline_configs, baseline_hash = phase_configs(args.baseline_root, "baseline")
    if candidate_hash != baseline_hash:
        raise ValueError("candidate and Mamba-1 source snapshots differ")

    resource = load(args.resource_json)
    protocol = resource.get("benchmark_protocol", {})
    required = {
        "run_benchmark": True,
        "device": "cuda:0",
        "benchmark_batch_size": 32,
        "warmup_forwards": 25,
        "timed_forwards": 100,
        "include_representation": False,
        "relation_loss_included": False,
    }
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("resource protocol mismatch")

    records = resource.get("resource_records", [])
    if len(records) != 4:
        raise ValueError("resource record coverage mismatch")
    all_configs = candidate_configs | baseline_configs
    rows, seen = [], set()
    for record in records:
        dataset, model = str(record.get("dataset")), str(record.get("model"))
        key = (dataset, model)
        if dataset not in DATASETS or model not in ("MambaTSv3AutoCon", "Mamba1") or key in seen:
            raise ValueError(f"unexpected resource coordinate: {key}")
        seen.add(key)
        source = record.get("source", {})
        config = Path(str(source.get("config_path", ""))).resolve()
        if config not in all_configs or digest(config) != source.get("config_sha256"):
            raise ValueError(f"resource config is not phase-bound: {key}")
        if Path(str(record.get("source_root", ""))).resolve() != candidate_snapshot:
            raise ValueError(f"resource source root mismatch: {key}")
        source_file = candidate_snapshot / str(source.get("model_source_file", ""))
        if not source_file.is_file() or digest(source_file) != source.get("model_source_sha256"):
            raise ValueError(f"resource model source mismatch: {key}")
        gpu = record.get("gpu_inference", {})
        parameters = record.get("parameters", {})
        if gpu.get("status") != "ok":
            raise ValueError(f"resource benchmark failed: {key}")
        rows.append({
            "dataset": dataset,
            "model": model,
            "latency_ms_median": float(gpu["latency_ms_median"]),
            "latency_ms_p95": float(gpu["latency_ms_p95"]),
            "parameter_trainable": int(parameters["trainable"]),
            "peak_allocated_mb": float(gpu["steady_state_peak_allocated_mb"]),
            "forward_path": str(gpu["forward_path"]),
        })
    expected = {(dataset, model) for dataset in DATASETS for model in ("MambaTSv3AutoCon", "Mamba1")}
    if seen != expected:
        raise ValueError("incomplete resource coordinates")

    lookup = {(row["dataset"], row["model"]): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Relation-Auxiliary Mamba vs Plain Mamba-1 Resource Attestation",
        "",
        "All records are bound to successful validation manifests, the shared immutable source snapshot, and matching model-source hashes. Protocol: RTX 5090 D, FP32, device-resident synthetic inputs, `B=32,L=96,H=96`, 25 warm-up and 100 timed prediction-only forwards. AutoCon relation loss and returned representations are excluded.",
        "",
        "| Dataset | Candidate ms | Mamba-1 ms | Candidate latency delta | Candidate trainable params | Mamba-1 trainable params | Candidate peak MB | Mamba-1 peak MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        candidate = lookup[(dataset, "MambaTSv3AutoCon")]
        baseline = lookup[(dataset, "Mamba1")]
        delta = 100.0 * (candidate["latency_ms_median"] / baseline["latency_ms_median"] - 1.0)
        lines.append(
            f"| {dataset} | {candidate['latency_ms_median']:.4f} | {baseline['latency_ms_median']:.4f} | {delta:+.1f}% | {candidate['parameter_trainable']} | {baseline['parameter_trainable']} | {candidate['peak_allocated_mb']:.1f} | {baseline['peak_allocated_mb']:.1f} |"
        )
    lines.extend([
        "",
        f"Shared source tree SHA-256: `{candidate_hash}`. The candidate is slower and uses more memory than plain Mamba-1 on both measured shapes; this is a scoped resource boundary, not an accuracy claim or an end-to-end throughput result.",
    ])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": candidate_hash, "test_metrics_produced": False, "protocol": required}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
