#!/usr/bin/env python3
"""Read-only audit of fixed-shape PEMS03 CUDA Graph batch grid."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/cuda_graph_batch_grid"
BATCHES = (1, 4, 8)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    rows = []
    for batch in BATCHES:
        candidate, patch = load(OUTPUT / f"channel_patch_b{batch}.json"), load(OUTPUT / f"official_patchtst_b{batch}.json")
        event_c, event_p = load(OUTPUT / f"channel_patch_b{batch}_event.json"), load(OUTPUT / f"official_patchtst_b{batch}_event.json")
        if any(event.get("status") != "completed" or event.get("trigger") is not None for event in (event_c, event_p)): raise ValueError("GPU guard failure")
        if candidate["execution_mode"] != "cuda_graph_fixed_shape_with_device_input_copy" or candidate["cuda_graph_max_abs_error"] != 0.0: raise ValueError("candidate graph contract failure")
        if candidate["mamba_scan_count"] != 1 or candidate["mamba_encoder_block_calls_during_capture"] != [5] or candidate["relation_loss_included"]: raise ValueError("candidate inference contract failure")
        rows.append({"batch_size": batch, "channel_patch_cuda_graph_ms": candidate["latency_ms_median"], "official_patchtst_ms": patch["latency_ms_median"], "candidate_vs_patchtst_percent": (candidate["latency_ms_median"] / patch["latency_ms_median"] - 1) * 100})
    result = {"test_metrics_produced": False, "fixed_shape_device_only": True, "rows": rows, "candidate_faster_batches": [row["batch_size"] for row in rows if row["candidate_vs_patchtst_percent"] < 0], "all_cuda_graph_equivalent": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
