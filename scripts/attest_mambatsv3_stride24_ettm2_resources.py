#!/usr/bin/env python3
"""Attest the ETTm2 stride-24 validation-bound resource matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASET = "ETTm2"
MODELS = ("MambaTSv3AutoCon", "DLinear", "RLinear", "PatchTST", "iTransformer", "Mamba1", "TimesNet", "Mamba2")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def phase_configs(root: Path, label: str) -> tuple[Path, set[Path], str]:
    validation, provenance = load(root / "phase_validation.json"), load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    snapshot = Path(str(provenance.get("source_snapshot", ""))).resolve()
    expected = 3 if label == "candidate" else 21
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected or not source or not snapshot.is_dir():
        raise ValueError(f"invalid phase: {label}")
    configs, seen = set(), set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw); job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        config = (Path(record["expected_artifact"]).parent / "config.json").resolve()
        payload = load(config)
        if payload.get("data") != DATASET or int(payload.get("pred_len", -1)) != 96:
            raise ValueError(f"invalid coordinate: {job_id}")
        configs.add(config)
    if len(configs) != expected:
        raise ValueError(f"coverage mismatch: {label}")
    return snapshot, configs, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--resource-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot_a, candidate, source_a = phase_configs(args.candidate_root, "candidate")
    snapshot_b, baselines, source_b = phase_configs(args.baseline_root, "baseline")
    if snapshot_a != snapshot_b or source_a != source_b:
        raise ValueError("phases do not share immutable source")
    payload = load(args.resource_json)
    protocol = payload.get("benchmark_protocol", {})
    required = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3401, "include_representation": False, "relation_loss_included": False}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("resource protocol mismatch")
    configs = candidate | baselines
    rows, seen = [], set()
    for record in payload.get("resource_records", []):
        key = (str(record.get("dataset")), str(record.get("model")))
        if key in seen or key[0] != DATASET or key[1] not in MODELS:
            raise ValueError(f"unexpected resource coordinate: {key}")
        seen.add(key)
        source = record.get("source", {}); config = Path(str(source.get("config_path", ""))).resolve()
        if config not in configs or digest(config) != source.get("config_sha256"):
            raise ValueError(f"resource config is not manifest-bound: {key}")
        if Path(str(record.get("source_root", ""))).resolve() != snapshot_a:
            raise ValueError(f"source root mismatch: {key}")
        source_file = snapshot_a / str(source.get("model_source_file", ""))
        if not source_file.is_file() or digest(source_file) != source.get("model_source_sha256"):
            raise ValueError(f"model source mismatch: {key}")
        gpu = record.get("gpu_inference", {})
        if gpu.get("status") != "ok":
            raise ValueError(f"resource benchmark failed: {key}")
        rows.append({"dataset": key[0], "model": key[1], "latency_ms_median": float(gpu["latency_ms_median"]), "parameter_trainable": int(record["parameters"]["trainable"]), "peak_allocated_mb": float(gpu["steady_state_peak_allocated_mb"])})
    if seen != {(DATASET, model) for model in MODELS}:
        raise ValueError("incomplete resource matrix")
    lookup = {(row["dataset"], row["model"]): row for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    values = {model: lookup[(DATASET, model)] for model in MODELS}
    lines = ["# Stride-24 ETTm2 Resource Attestation", "", "Manifest-bound RTX 5090 D FP32 prediction-only measurement with device-resident synthetic `B=32,L=96,H=96` inputs, 25 warm-up and 100 timed forwards. AutoCon relation loss and returned representations are excluded.", "", "| Dataset | Candidate ms | DLinear | RLinear | PatchTST | iTransformer | Mamba-1 | TimesNet | Mamba-2 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", "| ETTm2 | {candidate:.4f} | {dlinear:.4f} | {rlinear:.4f} | {patch:.4f} | {itrans:.4f} | {mamba1:.4f} | {times:.4f} | {mamba2:.4f} |".format(candidate=values["MambaTSv3AutoCon"]["latency_ms_median"], dlinear=values["DLinear"]["latency_ms_median"], rlinear=values["RLinear"]["latency_ms_median"], patch=values["PatchTST"]["latency_ms_median"], itrans=values["iTransformer"]["latency_ms_median"], mamba1=values["Mamba1"]["latency_ms_median"], times=values["TimesNet"]["latency_ms_median"], mamba2=values["Mamba2"]["latency_ms_median"]), "", f"Immutable source tree SHA-256: `{source_a}`. This is a fixed-shape resource measurement, not training throughput, end-to-end latency, or a universal speed claim."]
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
