from pathlib import Path


def test_cuda_graph_batch_grid_is_registered_over_all_fixed_shapes():
    text = Path("scripts/run_pems03_cuda_graph_batch_grid.py").read_text(encoding="utf-8")
    for token in ("BATCHES = (1, 4, 8)", "--cuda-graph", "run_gpu_guarded.py", "fixed-shape latency"):
        assert token in text


def test_cuda_graph_batch_grid_audit_checks_capture_and_all_batches():
    text = Path("scripts/audit_pems03_cuda_graph_batch_grid.py").read_text(encoding="utf-8")
    for token in ("BATCHES = (1, 4, 8)", "mamba_encoder_block_calls_during_capture", "candidate_vs_patchtst_percent", "candidate_faster_batches"):
        assert token in text
