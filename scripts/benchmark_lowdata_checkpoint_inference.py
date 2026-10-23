#!/usr/bin/env python3
"""Benchmark a frozen low-data checkpoint on a prediction-only CUDA path."""

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

from scripts.evaluate_lowdata_checkpoint_test import _model, _prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=["hybrid", "baseline"], required=True)
    parser.add_argument("--baseline", choices=["DLinear", "NLinear", "RLinear", "PatchTST", "TimeMixer", "TimesNet", "iTransformer"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.kind == "baseline" and args.baseline is None:
        raise ValueError("--baseline is required for baseline checkpoints")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("invalid benchmark geometry")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = int(saved["target_channels"])
    model = _model(saved, args.kind, args.baseline, channels)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    x = torch.randn(args.batch_size, int(saved["config"]["seq_len"]), channels, device="cuda")
    with torch.inference_mode():
        for _ in range(args.warmup):
            _prediction(model, args.kind, x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); _prediction(model, args.kind, x); end.record()
            torch.cuda.synchronize(); samples.append(float(start.elapsed_time(end)))
    result = {
        "checkpoint": str(args.checkpoint), "kind": args.kind, "baseline": args.baseline,
        "batch_size": args.batch_size, "warmup": args.warmup, "iterations": args.iterations,
        "forward_path": "prediction_only", "relation_loss_included": False,
        "mamba_scan_count": 1 if args.kind == "hybrid" else 0,
        "latency_ms_median": float(statistics.median(samples)),
        "latency_ms_mean": float(statistics.fmean(samples)),
        "latency_ms_p95": float(sorted(samples)[int(.95 * (len(samples) - 1))]),
        "parameters": int(sum(item.numel() for item in model.parameters())),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
        "dataset_rows_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
