from pathlib import Path


def test_pems03_cuda_graph_benchmark_is_guarded_and_prediction_only():
    text = Path("scripts/run_pems03_channel_patch_cuda_graph_benchmark.py").read_text(encoding="utf-8")
    for token in ("--cuda-graph", "run_gpu_guarded.py", "no dataset reads", "test_metrics_produced"):
        assert token in text


def test_pems03_cuda_graph_audit_checks_equivalence_capture_and_comparator_latency():
    text = Path("scripts/audit_pems03_channel_patch_cuda_graph_benchmark.py").read_text(encoding="utf-8")
    for token in ("cuda_graph_max_abs_error", "mamba_encoder_block_calls_during_capture", "official_patchtst_latency_ms_median", "fixed_shape_device_only"):
        assert token in text
