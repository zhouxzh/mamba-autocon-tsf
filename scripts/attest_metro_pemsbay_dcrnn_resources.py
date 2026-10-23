#!/usr/bin/env python3
"""Bind DCRNN resource records to the complete frozen validation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASET, HORIZON = "metr_la", 12
MODELS = ("MambaTSv3AutoCon", "DCRNN")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def phase_configs(root: Path, expected: int) -> tuple[Path, str, set[Path]]:
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    snapshot = Path(str(provenance.get("source_snapshot", ""))).resolve()
    if not source or not snapshot.is_dir() or provenance.get("test_selection_allowed") is not False:
        raise ValueError("invalid phase provenance")
    rows = [json.loads(line) for line in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or any(row.get("status") != "ok" or row.get("source_tree_sha256_before") != source or row.get("source_tree_sha256_after") != source for row in rows):
        raise ValueError("incomplete manifest")
    configs = {(Path(row["expected_artifact"]).parent / "config.json").resolve() for row in rows}
    if len(configs) != expected:
        raise ValueError("config coverage mismatch")
    return snapshot, source, configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--dcrnn-root", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot_a, source_a, candidate_configs = phase_configs(args.candidate_root, 6)
    snapshot_b, source_b, dcrnn_configs = phase_configs(args.dcrnn_root, 6)
    if snapshot_a != snapshot_b or source_a != source_b:
        raise ValueError("phases do not share one source snapshot")
    allowed = candidate_configs | dcrnn_configs
    payload = load(args.resource_root / "model_resources.json")
    protocol = payload.get("benchmark_protocol", {})
    required = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3601, "include_representation": False, "relation_loss_included": False}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("benchmark protocol mismatch")
    by_model = {record.get("model"): record for record in payload.get("resource_records", [])}
    if set(by_model) != set(MODELS):
        raise ValueError("resource model coverage mismatch")
    rows = []
    for model in MODELS:
        record = by_model[model]
        config = Path(str(record.get("source", {}).get("config_path", ""))).resolve()
        if record.get("dataset") != DATASET or record.get("gpu_inference", {}).get("status") != "ok" or config not in allowed:
            raise ValueError(f"unbound resource: {model}")
        if digest(config) != record.get("source", {}).get("config_sha256") or Path(str(record.get("source_root", ""))).resolve() != snapshot_a:
            raise ValueError(f"source binding mismatch: {model}")
        source_file = snapshot_a / str(record.get("source", {}).get("model_source_file", ""))
        if not source_file.is_file() or digest(source_file) != record.get("source", {}).get("model_source_sha256"):
            raise ValueError(f"model source mismatch: {model}")
        rows.append({"dataset": "METR-LA", "horizon": HORIZON, "model": model, "latency_ms_median": float(record["gpu_inference"]["latency_ms_median"]), "parameter_trainable": int(record["parameters"]["trainable"]), "peak_allocated_mb": float(record["gpu_inference"]["steady_state_peak_allocated_mb"])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lookup = {row["model"]: row for row in rows}; candidate, dcrnn = lookup["MambaTSv3AutoCon"], lookup["DCRNN"]
    speedup = 100 * (dcrnn["latency_ms_median"] / candidate["latency_ms_median"] - 1)
    lines = ["# METR-LA MambaTSv3AutoCon versus Optimized DCRNN Resource Attestation", "", "Manifest-bound serial RTX 5090 D FP32 prediction-only measurement with device-resident synthetic `B=32,L=96` inputs, 25 warm-up and 100 timed forwards. AutoCon relation loss is excluded; the candidate request disables auxiliary representation output where supported.", "", "| H | Candidate ms | DCRNN ms | Candidate lower by | Candidate params | DCRNN params |", "| ---: | ---: | ---: | ---: | ---: | ---: |", f"| 12 | {candidate['latency_ms_median']:.4f} | {dcrnn['latency_ms_median']:.4f} | {speedup:.2f}% | {candidate['parameter_trainable']} | {dcrnn['parameter_trainable']} |", "", f"Immutable source tree SHA-256: `{source_a}`. This is fixed-shape prediction latency, not training throughput, end-to-end latency, or a universal speed claim."]
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
