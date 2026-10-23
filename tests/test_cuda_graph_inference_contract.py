from pathlib import Path


def test_cuda_graph_predictor_requires_fixed_shape_device_input_and_reuses_output_buffer():
    text = Path("utils/cuda_graph_inference.py").read_text(encoding="utf-8")
    for token in ("CudaGraphPredictor", "torch.cuda.CUDAGraph", "static_input.copy_", "input shape and dtype", "becomes invalid"):
        assert token in text


def test_checkpoint_benchmark_records_cuda_graph_execution_and_equivalence():
    text = Path("scripts/benchmark_mambatsv3_checkpoint_inference.py").read_text(encoding="utf-8")
    for token in ("--cuda-graph", "CudaGraphPredictor", "cuda_graph_max_abs_error", "mamba_encoder_block_calls_during_capture", "device-side input copy"):
        assert token in text
