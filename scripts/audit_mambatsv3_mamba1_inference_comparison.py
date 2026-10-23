#!/usr/bin/env python3
"""Audit same-shape prediction-only inference comparison for frozen release."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/inference_comparison_pems_bay"


def _load(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _guard(name: str) -> None:
    payload = _load(name)
    if payload.get("status") != "completed" or payload.get("trigger") is not None or payload.get("returncode") != 0:
        raise ValueError(f"guard failure: {name}")


def main() -> int:
    mamba = _load("mamba1.json")
    patch = _load("patchtst.json")
    itransformer = _load("itransformer.json")
    for event in ("mamba1_event.json", "patchtst_event.json", "itransformer_event.json"):
        _guard(event)
    shared = ("batch_size", "warmup", "iterations")
    if any(mamba[key] != patch[key] or mamba[key] != itransformer[key] for key in shared):
        raise ValueError("benchmark geometry mismatch")
    if not (mamba["mamba_scan_count"] == 1 and not mamba["relation_loss_included"]
            and not mamba["autocon_representation_returned"]):
        raise ValueError("Mamba inference path contract failed")
    result = {
        "geometry": {key: mamba[key] for key in shared},
        "mamba1": {key: mamba[key] for key in ("latency_ms_median", "latency_ms_p95", "parameters", "peak_allocated_mb", "mamba_scan_count", "relation_loss_included", "autocon_representation_returned")},
        "official_patchtst": {key: patch[key] for key in ("latency_ms_median", "latency_ms_p95", "parameters", "peak_allocated_mb", "official_commit")},
        "official_itransformer": {key: itransformer[key] for key in ("latency_ms_median", "latency_ms_p95", "parameters", "peak_allocated_mb", "official_commit")},
        "mamba1_vs_patchtst_latency_percent": float((mamba["latency_ms_median"] / patch["latency_ms_median"] - 1) * 100),
        "mamba1_vs_itransformer_latency_percent": float((mamba["latency_ms_median"] / itransformer["latency_ms_median"] - 1) * 100),
        "scope": "same-device PEMS-BAY-shaped prediction-only microbenchmark; not end-to-end throughput",
        "claim_boundary": "proves clean one-scan AutoCon-free inference, not external-baseline latency superiority",
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
