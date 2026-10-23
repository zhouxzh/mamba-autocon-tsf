#!/usr/bin/env python3
"""Bounded CUDA resource preflight for one Mamba-1 or Mamba-3 block."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixer", choices=("mamba1", "mamba3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-channels", type=int, default=1300)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-state", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6124)
    args = parser.parse_args()
    if min(args.batch_channels, args.tokens, args.d_model, args.d_state, args.warmup, args.iterations) <= 0:
        raise ValueError("all benchmark dimensions and counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
    if args.mixer == "mamba1":
        from mamba_ssm import Mamba
        model = Mamba(
            d_model=args.d_model, d_state=args.d_state, d_conv=2, expand=1
        )
        mixer_metadata = {"d_conv": 2, "expand": 1}
    else:
        from mamba_ssm import Mamba3
        model = Mamba3(
            d_model=args.d_model, d_state=args.d_state, headdim=args.d_model,
            is_mimo=False, chunk_size=32, dtype=torch.float32,
        )
        mixer_metadata = {"headdim": args.d_model, "chunk_size": 32, "is_mimo": False}
    torch.manual_seed(args.seed)
    model = model.cuda().eval()
    values = torch.randn(
        args.batch_channels, args.tokens, args.d_model, device="cuda"
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(values)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(values)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
        finite = bool(torch.isfinite(output).all())
    result = {
        "mixer": args.mixer,
        "shape": [args.batch_channels, args.tokens, args.d_model],
        "d_state": args.d_state,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "median_ms": float(torch.tensor(samples).median()),
        "mean_ms": float(sum(samples) / len(samples)),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "finite": finite,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mixer_metadata": mixer_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
