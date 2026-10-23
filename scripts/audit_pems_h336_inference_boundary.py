#!/usr/bin/env python3
"""Compare same-geometry deployed active and official PatchTST H336 benchmarks."""

from __future__ import annotations

import json
from pathlib import Path


ACTIVE = Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_h336_confirmation_20261003/inference.json")
OFFICIAL = Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_h336_extension_20261003/official_patchtst_inference.json")
OUTPUT = Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_h336_extension_20261003/inference_boundary.json")


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    active, official = load(ACTIVE), load(OFFICIAL)
    for name, payload in (("active", active), ("official", official)):
        if payload["batch_size"] != 4 or payload["warmup"] != 25 or payload["iterations"] != 100:
            raise ValueError("{} benchmark geometry mismatch".format(name))
        if payload.get("relation_loss_included") is not False:
            raise ValueError("{} benchmark included relation loss".format(name))
    if active.get("mamba_scan_count") != 1 or active.get("autocon_representation_returned") is not False:
        raise ValueError("active benchmark is not deployed single-scan prediction path")
    latency_percent = (active["latency_ms_median"] / official["latency_ms_median"] - 1.0) * 100.0
    memory_percent = (active["peak_allocated_mb"] / official["peak_allocated_mb"] - 1.0) * 100.0
    result = {
        "coordinate": "PEMS-BAY H336, batch=4, prediction-only device-resident",
        "active": active,
        "official_patchtst": official,
        "active_vs_official_latency_percent": latency_percent,
        "active_vs_official_peak_memory_percent": memory_percent,
        "speed_advantage_supported": bool(latency_percent < 0.0),
        "single_scan_inference_supported": True,
        "audit_valid": True,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
