#!/usr/bin/env python3
"""Read-only audit for fixed-shape CUDA Graph PEMS03 deployment benchmark."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    graph = load(OUTPUT / "channel_patch_cuda_graph.json"); eager = load(OUTPUT / "channel_patch.json"); patch = load(OUTPUT / "official_patchtst.json"); event = load(OUTPUT / "channel_patch_cuda_graph_event.json")
    if event.get("status") != "completed" or event.get("trigger") is not None: raise ValueError("GPU guard failure")
    if graph["execution_mode"] != "cuda_graph_fixed_shape_with_device_input_copy" or graph["cuda_graph_max_abs_error"] != 0.0: raise ValueError("CUDA Graph equivalence contract failure")
    if graph["mamba_scan_count"] != 1 or graph["mamba_encoder_block_calls_during_capture"] != [5]: raise ValueError("single-scan capture contract failure")
    if graph.get("relation_loss_included") or graph.get("autocon_representation_returned"): raise ValueError("prediction-only contract failure")
    result = {"test_metrics_produced": False, "fixed_shape_device_only": True, "cuda_graph_equivalent": True, "one_scan_no_relation_loss": True, "cuda_graph_latency_ms_median": graph["latency_ms_median"], "eager_latency_ms_median": eager["latency_ms_median"], "official_patchtst_latency_ms_median": patch["latency_ms_median"], "cuda_graph_vs_eager_percent": (graph["latency_ms_median"] / eager["latency_ms_median"] - 1) * 100, "cuda_graph_vs_patchtst_percent": (graph["latency_ms_median"] / patch["latency_ms_median"] - 1) * 100, "parameters": graph["parameters"], "submission_ready": False}
    (OUTPUT / "cuda_graph_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
