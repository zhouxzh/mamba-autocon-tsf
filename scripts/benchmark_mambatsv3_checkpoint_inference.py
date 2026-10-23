#!/usr/bin/env python3
"""Benchmark deployed prediction-only MambaTSv3 checkpoints on CUDA.

This loads a completed target checkpoint, creates device-resident synthetic
inputs, and explicitly calls ``return_repr=False``. It never constructs an
AutoCon/ACF loss, reads no dataset rows, and measures only forward latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DLinear, MambaTSv3, NLinear
from models.MambaDLinearHybrid import Model as Hybrid
from scripts.finetune_mambatsv3_core_transfer import config
from scripts.evaluate_mambatsv3_lowdata_robustness import _args
from utils.cuda_graph_inference import CudaGraphPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="measure fixed-shape CUDA Graph replay including device-side input copy")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("invalid benchmark geometry")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    channels = int(saved["target_channels"])
    if args.hybrid:
        mamba = MambaTSv3.Model(config(_args(cfg, channels), channels)).cuda()
        linear_config = config(_args(cfg, channels), channels); linear_config.AutoCon_wnorm = "MeanStd"; linear_config.moving_avg = 25; linear_config.individual = False
        linear_class = {"DLinear": DLinear.Model, "NLinear": NLinear.Model}[cfg["linear_model"]]
        model = Hybrid(mamba, linear_class(linear_config).cuda(), linear_only=bool(cfg["linear_only"]),
                       mamba_correction_output=bool(cfg.get("mamba_correction_output", False)),
                       channelwise_residual_scale=bool(cfg.get("mamba_channelwise_residual_scale", False)),
                       centered_channelwise_residual_scale=bool(cfg.get("mamba_centered_channelwise_residual_scale", False)),
                       channelwise_horizon_residual_scale=bool(cfg.get("mamba_channelwise_horizon_residual_scale", False)),
                       channelwise_residual_bias=bool(cfg.get("mamba_channelwise_residual_bias", False)),
                       shared_channel_statistic_residual_gate=bool(cfg.get("mamba_shared_channel_statistic_residual_gate", False)),
                       shared_channel_statistic_gate_rank=int(cfg.get("mamba_shared_channel_statistic_gate_rank", 8))).cuda().eval()
    else:
        model = MambaTSv3.Model(config(_args(cfg, channels), channels)).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    x = torch.randn(args.batch_size, int(cfg["seq_len"]), channels, device="cuda")
    mamba = model.mamba if isinstance(model, Hybrid) else model
    encoder_blocks = list(getattr(mamba, "encoder", []) or [])
    calls = [0 for _ in encoder_blocks]
    hooks = []
    for index, block in enumerate(encoder_blocks):
        def _count(_, __, index=index):
            calls[index] += 1
        hooks.append(block.register_forward_pre_hook(_count))
    graph_predictor = None
    graph_max_abs_error = None
    capture_calls = None
    try:
        if args.cuda_graph:
            with torch.inference_mode():
                eager = model(x) if isinstance(model, Hybrid) else model(x, return_repr=False)
            graph_predictor = CudaGraphPredictor(model, x)
            with torch.inference_mode():
                graph_output = graph_predictor.predict(x)
                torch.cuda.synchronize()
            graph_max_abs_error = float((graph_output - eager).abs().max())
            if graph_max_abs_error > 1e-5:
                raise RuntimeError("CUDA Graph output differs from eager prediction")
            capture_calls = list(calls)
            if encoder_blocks and any(value != 5 for value in capture_calls):
                raise RuntimeError("CUDA Graph capture did not execute exactly one Mamba scan per capture forward")
        with torch.inference_mode():
            for _ in range(args.warmup):
                output = graph_predictor.predict(x) if graph_predictor is not None else (
                    model(x) if isinstance(model, Hybrid) else model(x, return_repr=False)
                )
                if isinstance(output, tuple):
                    prediction, representation = output
                    if representation is not None:
                        raise RuntimeError("prediction-only benchmark unexpectedly returned AutoCon representation")
                else:
                    prediction = output
                del prediction
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            latencies = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = graph_predictor.predict(x) if graph_predictor is not None else (
                    model(x) if isinstance(model, Hybrid) else model(x, return_repr=False)
                )
                end.record(); torch.cuda.synchronize()
                prediction = output[0] if isinstance(output, tuple) else output
                latencies.append(float(start.elapsed_time(end)))
                del prediction
    finally:
        for hook in hooks:
            hook.remove()
    expected_calls = args.warmup + args.iterations
    if not args.cuda_graph and encoder_blocks and any(value != expected_calls for value in calls):
        raise RuntimeError("Mamba encoder did not run exactly once per prediction call")
    result = {
        "checkpoint": str(args.checkpoint), "batch_size": args.batch_size,
        "hybrid": bool(args.hybrid),
        "warmup": args.warmup, "iterations": args.iterations,
        "forward_path": "prediction_only_return_repr_false",
        "execution_mode": "cuda_graph_fixed_shape_with_device_input_copy" if args.cuda_graph else "eager",
        "cuda_graph_max_abs_error": graph_max_abs_error,
        "autocon_representation_returned": False,
        "relation_loss_included": False,
        "mamba_scan_count": len(encoder_blocks),
        "mamba_encoder_block_calls": calls,
        "mamba_encoder_block_calls_during_capture": capture_calls,
        "latency_ms_median": float(statistics.median(latencies)),
        "latency_ms_mean": float(statistics.fmean(latencies)),
        "latency_ms_p95": float(sorted(latencies)[int(0.95 * (len(latencies) - 1))]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
