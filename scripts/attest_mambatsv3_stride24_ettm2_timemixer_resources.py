#!/usr/bin/env python3
"""Attest ETTm2 candidate versus TimeMixer fixed-shape resources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASET, HORIZONS = "ETTm2", (96, 336, 720)
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
    source, snapshot = str(provenance.get("source_tree_sha256", "")), Path(str(provenance.get("source_snapshot", ""))).resolve()
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
    snapshot_a, candidate_configs, source_a = configs(args.candidate_root, 9); snapshot_b, timemixer_configs, source_b = configs(args.timemixer_root, 9)
    if snapshot_a != snapshot_b or source_a != source_b: raise ValueError("phases do not share source")
    all_configs, rows = candidate_configs | timemixer_configs, []
    for horizon in HORIZONS:
        payload = load(args.resource_root / f"h{horizon}" / "model_resources.json")
        protocol = payload.get("benchmark_protocol", {})
        required = {"run_benchmark": True, "device": "cuda:0", "benchmark_batch_size": 32, "warmup_forwards": 25, "timed_forwards": 100, "input_seed": 3431, "include_representation": False, "relation_loss_included": False}
        if any(protocol.get(key) != value for key, value in required.items()): raise ValueError(f"protocol mismatch H{horizon}")
        seen = set()
        for record in payload.get("resource_records", []):
            model = str(record.get("model")); seen.add(model); config = Path(str(record.get("source", {}).get("config_path", ""))).resolve()
            config_payload = load(config)
            if model not in MODELS or config not in all_configs or digest(config) != record.get("source", {}).get("config_sha256") or int(config_payload.get("pred_len", -1)) != horizon: raise ValueError(f"unbound resource H{horizon} {model}")
            if Path(str(record.get("source_root", ""))).resolve() != snapshot_a or record.get("gpu_inference", {}).get("status") != "ok": raise ValueError(f"invalid resource H{horizon} {model}")
            rows.append({"dataset": DATASET, "horizon": horizon, "model": model, "latency_ms_median": float(record["gpu_inference"]["latency_ms_median"]), "parameter_trainable": int(record["parameters"]["trainable"]), "peak_allocated_mb": float(record["gpu_inference"]["steady_state_peak_allocated_mb"])})
        if seen != MODELS: raise ValueError(f"incomplete resource H{horizon}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "resource_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lookup = {(row["horizon"], row["model"]): row for row in rows}
    lines = ["# Stride-24 MambaTSv3AutoCon versus TimeMixer Resource Attestation", "", "Manifest-bound serial RTX 5090 D FP32 prediction-only measurement with device-resident synthetic `B=32,L=96` inputs, 25 warm-up and 100 timed forwards per horizon. AutoCon relation loss and returned representations are excluded.", "", "| H | Candidate ms | TimeMixer ms | Candidate faster by | Candidate params | TimeMixer params |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for horizon in HORIZONS:
        candidate, baseline = lookup[(horizon, "MambaTSv3AutoCon")], lookup[(horizon, "TimeMixer")]
        lines.append("| {horizon} | {candidate:.4f} | {baseline:.4f} | {gain:.2f}% | {cp} | {bp} |".format(horizon=horizon, candidate=candidate["latency_ms_median"], baseline=baseline["latency_ms_median"], gain=100 * (1 - candidate["latency_ms_median"] / baseline["latency_ms_median"]), cp=candidate["parameter_trainable"], bp=baseline["parameter_trainable"]))
    lines.extend(["", f"Immutable source tree SHA-256: `{source_a}`. This is fixed-shape prediction latency, not training throughput, end-to-end latency, or a universal speed claim."])
    (args.output_dir / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "attestation.json").write_text(json.dumps({"resource_records": len(rows), "source_tree_sha256": source_a, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"resource_records": len(rows), "test_metrics_produced": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
