#!/usr/bin/env python3
"""Read-only PEMS03 prediction-only deployment benchmark audit."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    plan, candidate, patch = (load(OUTPUT / name) for name in ("plan.json", "channel_patch.json", "official_patchtst.json"))
    if plan.get("test_metrics_produced") or candidate.get("relation_loss_included") or candidate.get("autocon_representation_returned"):
        raise ValueError("prediction-only contract failure")
    if candidate["mamba_scan_count"] != 1 or candidate["mamba_encoder_block_calls"] != [candidate["warmup"] + candidate["iterations"]]:
        raise ValueError("single Mamba scan contract failure")
    result = {"prediction_only": True, "test_metrics_produced": False, "candidate_one_scan_no_relation_loss": True, "channel_patch_latency_ms_median": candidate["latency_ms_median"], "official_patchtst_latency_ms_median": patch["latency_ms_median"], "candidate_vs_patchtst_percent": (candidate["latency_ms_median"] / patch["latency_ms_median"] - 1) * 100, "candidate_parameters": candidate["parameters"], "patchtst_parameters": patch["parameters"], "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
