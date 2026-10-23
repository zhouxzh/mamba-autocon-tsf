#!/usr/bin/env python3
"""Profile a frozen model config on a device-resident synthetic forward path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_model_resources import (
    _activate_resource_acf_channel_order,
    _forward_outputs,
    _make_inputs,
    _model_class,
    _set_seeds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--active", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires CUDA")
    torch.cuda.set_device(device)
    _set_seeds(torch, int(config.get("seed", 3601)))
    source_root = args.source_root.resolve()
    model_cls = _model_class(source_root, config)
    model = model_cls(SimpleNamespace(**config)).float().to(device).eval()
    _activate_resource_acf_channel_order(model, config, torch)
    inputs = _make_inputs(torch, config, int(args.batch_size), device)
    with torch.inference_mode():
        for _ in range(int(args.warmup)):
            _forward_outputs(model, inputs, torch, config, request_representation=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with torch.inference_mode():
            for _ in range(int(args.active)):
                _forward_outputs(model, inputs, torch, config, request_representation=False)
    torch.cuda.synchronize(device)
    events = []
    for item in profiler.key_averages():
        cuda_us = float(getattr(item, "self_device_time_total", 0.0))
        if cuda_us <= 0:
            cuda_us = float(getattr(item, "self_cuda_time_total", 0.0))
        events.append({
            "name": item.key,
            "calls": int(item.count),
            "self_cuda_us": cuda_us,
            "cuda_us": float(getattr(item, "device_time_total", 0.0)),
            "self_cpu_us": float(getattr(item, "self_cpu_time_total", 0.0)),
            "cpu_us": float(getattr(item, "cpu_time_total", 0.0)),
            "self_cuda_memory_bytes": int(getattr(item, "self_device_memory_usage", 0)),
        })
    events.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    payload = {
        "config": str(args.config.resolve()),
        "source_root": str(source_root),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch_size": int(args.batch_size),
        "warmup_forwards": int(args.warmup),
        "profile_forwards": int(args.active),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "events_sorted_by_self_cuda": events,
        "trace_path": str((args.output.parent / "trace.json").resolve()),
        "scope": "synthetic device-resident prediction-only forward; no data or test metrics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    profiler.export_chrome_trace(str(args.output.parent / "trace.json"))
    print(json.dumps({"events": len(events), "peak_allocated_mb": payload["peak_allocated_mb"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
