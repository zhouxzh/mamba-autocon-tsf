#!/usr/bin/env python3
"""Attest stride-24 candidate/plain MambaTS fixed-shape resources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("ETTh1", "weather", "electricity", "traffic")
MODELS = ("MambaTSv3AutoCon", "MambaTS")
PHASES = {
    "candidate": ("stride24_candidate_h96_validation", 12, {"MambaTSv3AutoCon"}),
    "baseline": ("stride24_plain_mambats_h96_validation", 12, {"MambaTS"}),
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


def manifest_configs(root: Path, label: str) -> tuple[Path, set[Path], str]:
    phase, expected, models = PHASES[label]
    validation = load(root / "phase_validation.json")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    if validation.get("valid") is not True or validation.get("phase") != phase or validation.get("expected_jobs") != expected or not source:
        raise ValueError(f"invalid phase: {label}")
    configs, seen = set(), set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        config = (Path(record["expected_artifact"]).parent / "config.json").resolve()
        payload = load(config)
        if payload.get("data") not in DATASETS or payload.get("model") not in models or int(payload.get("pred_len", -1)) != 96:
            raise ValueError(f"invalid coordinate: {job_id}")
        configs.add(config)
    if len(configs) != expected:
        raise ValueError(f"coverage mismatch: {label}")
    return Path(str(provenance["source_snapshot"])).resolve(), configs, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshots, configs, hashes = [], set(), []
    for label, root in (("candidate", args.candidate_root), ("baseline", args.baseline_root)):
        snapshot, phase_configs, source = manifest_configs(root, label)
        snapshots.append(snapshot); configs |= phase_configs; hashes.append(source)
    if len(set(hashes)) != 1:
        raise ValueError("phases do not share one immutable source hash")
    payload = load(args.resource_json)
    protocol = payload.get("benchmark_protocol", {})
    expected_protocol = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3381, "include_representation": False, "relation_loss_included": False}
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("resource protocol mismatch")
    rows, seen = [], set()
    for record in payload.get("resource_records", []):
        key = (str(record.get("dataset")), str(record.get("model")))
        if key in seen or key not in {(dataset, model) for dataset in DATASETS for model in MODELS}:
            raise ValueError(f"unexpected resource coordinate: {key}")
        seen.add(key)
        source = record.get("source", {}); config = Path(str(source.get("config_path", ""))).resolve()
        if config not in configs or digest(config) != source.get("config_sha256"):
            raise ValueError(f"resource config is not manifest-bound: {key}")
        if Path(str(record.get("source_root", ""))).resolve() != snapshots[0]:
            raise ValueError(f"resource source-root mismatch: {key}")
        source_file = snapshots[0] / str(source.get("model_source_file", ""))
        if not source_file.is_file() or digest(source_file) != source.get("model_source_sha256"):
            raise ValueError(f"model source mismatch: {key}")
        gpu = record.get("gpu_inference", {})
        if gpu.get("status") != "ok":
            raise ValueError(f"resource benchmark failed: {key}")
        rows.append({"dataset": key[0], "model": key[1], "latency_ms_median": float(gpu["latency_ms_median"]), "parameter_trainable": int(record["parameters"]["trainable"]), "peak_allocated_mb": float(gpu["steady_state_peak_allocated_mb"])})
    expected_keys = {(dataset, model) for dataset in DATASETS for model in MODELS}
    if seen != expected_keys:
        raise ValueError("incomplete resource matrix")
    lookup = {(row["dataset"], row["model"]): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Stride-24 AutoCon MambaTSv3 versus Plain MambaTS Resources", "", "Fixed-shape GPU prediction-only measurement on RTX 5090 D, FP32, device-resident synthetic inputs, `B=32,L=96,H=96`, 25 warm-up and 100 timed forwards. AutoCon relation loss and returned representations are excluded.", "", "| Dataset | AutoCon MambaTSv3 ms | Plain MambaTS ms | Candidate delta |", "| --- | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        candidate = lookup[(dataset, "MambaTSv3AutoCon")]["latency_ms_median"]; baseline = lookup[(dataset, "MambaTS")]["latency_ms_median"]
        lines.append(f"| {dataset} | {candidate:.4f} | {baseline:.4f} | {100 * (candidate / baseline - 1):+.1f}% |")
    lines.extend(["", f"Immutable source tree SHA-256: `{hashes[0]}`. This is a fixed-shape forward measurement, not training throughput, end-to-end latency, or an official external-repository reproduction."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": hashes[0], "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
