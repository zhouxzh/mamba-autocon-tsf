#!/usr/bin/env python3
"""Benchmark prediction-only latency of a frozen official PatchTST checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.official_patchtst_adapter import config, digest, official_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("invalid benchmark geometry")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    official_root = Path(saved["official_root"])
    for relative, key in (("models/PatchTST.py", "official_model_sha256"), ("layers/PatchTST_backbone.py", "official_backbone_sha256")):
        if key not in saved or digest(official_root / relative) != saved[key]:
            raise ValueError("official PatchTST source hash does not match checkpoint")
    model = official_model(official_root)(
        config(argparse.Namespace(**cfg), int(saved["target_channels"]))
    ).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    x = torch.randn(args.batch_size, int(cfg["seq_len"]), int(saved["target_channels"]), device="cuda")
    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(x)
            del output
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); output = model(x); end.record(); torch.cuda.synchronize()
            latencies.append(float(start.elapsed_time(end))); del output
    result = {
        "checkpoint": str(args.checkpoint), "official_repository": saved["official_model"],
        "official_commit": saved["official_commit"],
        "official_model_sha256": saved["official_model_sha256"],
        "official_backbone_sha256": saved["official_backbone_sha256"],
        "batch_size": args.batch_size, "warmup": args.warmup, "iterations": args.iterations,
        "forward_path": "prediction_only_official_patchtst", "relation_loss_included": False,
        "mamba_scan_count": 0, "autocon_representation_returned": False,
        "latency_ms_median": float(statistics.median(latencies)),
        "latency_ms_mean": float(statistics.fmean(latencies)),
        "latency_ms_p95": float(sorted(latencies)[int(.95 * (len(latencies) - 1))]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
