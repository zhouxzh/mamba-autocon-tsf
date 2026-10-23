#!/usr/bin/env python3
"""Attest ETTh2 H96 candidate versus TimeMixer fixed-shape resources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASET, HORIZON = "ETTh2", 96
MODELS = {"MambaTSv3AutoCon", "TimeMixer"}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def configs(root: Path, expected: int) -> tuple[Path, set[Path], str]:
    validation, provenance = load(root / "phase_validation.json"), load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", "")); snapshot = Path(str(provenance.get("source_snapshot", ""))).resolve()
    if validation.get("valid") is not True or validation.get("expected_jobs") != expected or not source or not snapshot.is_dir(): raise ValueError("invalid phase")
    output = set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        if record.get("status") != "ok" or record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source: raise ValueError("invalid manifest")
        output.add((Path(record["expected_artifact"]).parent / "config.json").resolve())
    if len(output) != expected: raise ValueError("incomplete manifest")
    return snapshot, output, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--candidate-root", type=Path, required=True); parser.add_argument("--timemixer-root", type=Path, required=True); parser.add_argument("--resource-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    snapshot_a, candidate_configs, source_a = configs(args.candidate_root, 3); snapshot_b, timemixer_configs, source_b = configs(args.timemixer_root, 3)
    if snapshot_a != snapshot_b or source_a != source_b: raise ValueError("phases do not share source")
    all_configs, rows = candidate_configs | timemixer_configs, []
    for label, model in (("candidate", "MambaTSv3AutoCon"), ("timemixer", "TimeMixer")):
        payload = load(args.resource_root / label / "model_resources.json"); protocol = payload.get("benchmark_protocol", {})
        required = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3441, "include_representation": False, "relation_loss_included": False}
        if any(protocol.get(key) != value for key, value in required.items()): raise ValueError(f"protocol mismatch {model}")
        records = payload.get("resource_records", [])
        if len(records) != 1: raise ValueError(f"resource count mismatch {model}")
        record = records[0]; config = Path(str(record.get("source", {}).get("config_path", ""))).resolve(); config_payload = load(config)
        if record.get("model") != model or record.get("dataset") != DATASET or config not in all_configs or digest(config) != record.get("source", {}).get("config_sha256") or int(config_payload.get("pred_len", -1)) != HORIZON: raise ValueError(f"unbound resource {model}")
        if Path(str(record.get("source_root", ""))).resolve() != snapshot_a or record.get("gpu_inference", {}).get("status") != "ok": raise ValueError(f"invalid resource {model}")
        source_file = snapshot_a / str(record.get("source", {}).get("model_source_file", "")); expected_source_hash = str(record.get("source", {}).get("model_source_sha256", ""))
        if not source_file.is_file() or not expected_source_hash or digest(source_file) != expected_source_hash: raise ValueError(f"model source mismatch {model}")
        rows.append({"dataset": DATASET, "horizon": HORIZON, "model": model, "latency_ms_median": float(record["gpu_inference"]["latency_ms_median"]), "parameter_trainable": int(record["parameters"]["trainable"]), "peak_allocated_mb": float(record["gpu_inference"]["steady_state_peak_allocated_mb"])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lookup = {row["model"]: row for row in rows}; candidate, baseline = lookup["MambaTSv3AutoCon"], lookup["TimeMixer"]
    gain = 100 * (1 - candidate["latency_ms_median"] / baseline["latency_ms_median"])
    lines = ["# Stride-24 MambaTSv3AutoCon versus TimeMixer ETTh2 H96 Resource Attestation", "", "Manifest-bound serial RTX 5090 D FP32 prediction-only measurement with device-resident synthetic `B=32,L=96` inputs, 25 warm-up and 100 timed forwards. AutoCon relation loss and returned representations are excluded.", "", "| H | Candidate ms | TimeMixer ms | Candidate faster by | Candidate params | TimeMixer params |", "| ---: | ---: | ---: | ---: | ---: | ---: |", f"| 96 | {candidate['latency_ms_median']:.4f} | {baseline['latency_ms_median']:.4f} | {gain:.2f}% | {candidate['parameter_trainable']} | {baseline['parameter_trainable']} |", "", f"Immutable source tree SHA-256: `{source_a}`. This is fixed-shape prediction latency, not training throughput, end-to-end latency, or a universal speed claim."]
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
