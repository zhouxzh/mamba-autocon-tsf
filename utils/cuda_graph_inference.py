"""Fixed-shape CUDA Graph execution for already-deployed PyTorch models."""

from __future__ import annotations

import torch


class CudaGraphPredictor:
    """Replay one fixed-shape CUDA prediction graph with device-side input copy.

    The returned tensor is graph-owned and becomes invalid after the next
    :meth:`predict` call.  Callers requiring output retention must clone it
    outside the measured deployment path.
    """

    def __init__(self, model, example_input: torch.Tensor, warmup: int = 3):
        if not example_input.is_cuda:
            raise ValueError("CudaGraphPredictor requires a CUDA example input")
        if not example_input.is_contiguous():
            raise ValueError("CudaGraphPredictor requires a contiguous example input")
        if int(warmup) < 0:
            raise ValueError("warmup must be non-negative")
        self.model = model.eval()
        self.shape = tuple(example_input.shape)
        self.dtype = example_input.dtype
        self.device = example_input.device
        self.static_input = torch.empty_like(example_input)
        self.static_input.copy_(example_input)
        with torch.inference_mode():
            for _ in range(int(warmup)):
                self.model(self.static_input)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.model(self.static_input)
        if isinstance(self.static_output, tuple):
            raise ValueError("CudaGraphPredictor requires a prediction tensor output")

    @torch.inference_mode()
    def predict(self, values: torch.Tensor) -> torch.Tensor:
        if not values.is_cuda or values.device != self.device:
            raise ValueError("CUDA input must use the captured device")
        if tuple(values.shape) != self.shape or values.dtype != self.dtype:
            raise ValueError("input shape and dtype must match the captured graph")
        if not values.is_contiguous():
            raise ValueError("CudaGraphPredictor requires contiguous inference input")
        self.static_input.copy_(values)
        self.graph.replay()
        return self.static_output
