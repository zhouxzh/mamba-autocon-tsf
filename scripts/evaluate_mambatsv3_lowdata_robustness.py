#!/usr/bin/env python3
"""Evaluate a saved low-data MambaTSv3 checkpoint on closed validation stresses.

The script opens only the chronological validation block, never the test
partition. Stresses mutate inference inputs only: later validation origins,
fixed channel masking, and deterministic signal-relative Gaussian noise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune_mambatsv3_core_transfer import config, model_output
from data_provider.data_loader import load_numeric_array_file
from models import DLinear, MambaTSv3, NLinear
from models.MambaDLinearHybrid import Model as Hybrid


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("no validation windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def _args(checkpoint_config, channels):
    return argparse.Namespace(
        seq_len=int(checkpoint_config["seq_len"]), pred_len=int(checkpoint_config["pred_len"]),
        patch_len=int(checkpoint_config["patch_len"]), patch_stride=int(checkpoint_config["patch_stride"]),
        d_model=int(checkpoint_config["d_model"]), d_ff=int(checkpoint_config["d_ff"]),
        mamba_d_state=int(checkpoint_config["mamba_d_state"]), mamba_d_conv=int(checkpoint_config["mamba_d_conv"]),
        mamba_expand=int(checkpoint_config["mamba_expand"]),
        mamba_headdim=int(checkpoint_config.get("mamba_headdim", checkpoint_config["d_model"])),
        mamba_ts_mixer_type=str(checkpoint_config.get("mamba_ts_mixer_type", "mamba1")),
        context_rank=int(checkpoint_config["context_rank"]),
        channel_agnostic_scan=bool(checkpoint_config.get("channel_agnostic_scan", False)),
        mamba_ts_channel_adapter=bool(checkpoint_config.get("mamba_ts_channel_adapter", False)),
        mamba_ts_channel_adapter_rank=int(checkpoint_config.get("mamba_ts_channel_adapter_rank", 8)),
        mamba_ts_target_mlp_adapter=bool(checkpoint_config.get("mamba_ts_target_mlp_adapter", False)),
        mamba_ts_target_mlp_adapter_rank=int(checkpoint_config.get("mamba_ts_target_mlp_adapter_rank", 32)),
        mamba_ts_repr_forecast_fusion=bool(checkpoint_config.get("mamba_ts_repr_forecast_fusion", False)),
        mamba_ts_repr_forecast_channelwise=bool(checkpoint_config.get("mamba_ts_repr_forecast_channelwise", False)),
        mamba_ts_shared_channel_summary=bool(checkpoint_config.get("mamba_ts_shared_channel_summary", False)),
        mamba_channelwise_residual_scale=bool(checkpoint_config.get("mamba_channelwise_residual_scale", False)),
        mamba_centered_channelwise_residual_scale=bool(checkpoint_config.get("mamba_centered_channelwise_residual_scale", False)),
        mamba_channelwise_horizon_residual_scale=bool(checkpoint_config.get("mamba_channelwise_horizon_residual_scale", False)),
        mamba_channelwise_residual_bias=bool(checkpoint_config.get("mamba_channelwise_residual_bias", False)),
        mamba_shared_channel_statistic_residual_gate=bool(checkpoint_config.get("mamba_shared_channel_statistic_residual_gate", False)),
        mamba_shared_channel_statistic_gate_rank=int(checkpoint_config.get("mamba_shared_channel_statistic_gate_rank", 8)),
    )


def _prediction(model, x):
    if isinstance(model, Hybrid):
        return model(x)
    return model_output(model, x)


def _evaluate(model, loader, mask_fraction, noise_scale, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    errors = []
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        if mask_fraction:
            channels = x.shape[-1]
            count = max(1, int(round(channels * mask_fraction)))
            chosen = torch.randperm(channels, generator=generator, device=x.device)[:count]
            x = x.clone(); x[:, :, chosen] = 0.0
        if noise_scale:
            scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            x = x + noise_scale * scale * torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        errors.append(F.mse_loss(_prediction(model, x), y, reduction="none").mean(dim=(1, 2)).cpu())
    return float(torch.cat(errors).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    # These local experiment checkpoints include argparse Path values in
    # metadata. They are produced in this workspace and are read-only here.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    raw = load_numeric_array_file(args.dataset)
    train_end = int(len(raw) * float(cfg["train_fraction"]))
    val_end = int(len(raw) * (float(cfg["train_fraction"]) + float(cfg["val_fraction"])))
    few_end = int(train_end * float(cfg["few_shot_fraction"]))
    mean = raw[:few_end].mean(axis=0, keepdims=True)
    std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    starts = np.arange(max(0, train_end - int(cfg["seq_len"])), val_end - int(cfg["seq_len"]) - int(cfg["pred_len"]) + 1)
    starts = starts[starts + int(cfg["seq_len"]) >= train_end]
    midpoint = starts[len(starts) // 2:]
    if args.hybrid:
        mamba = MambaTSv3.Model(config(_args(cfg, raw.shape[1]), raw.shape[1])).cuda()
        linear_config = config(_args(cfg, raw.shape[1]), raw.shape[1]); linear_config.AutoCon_wnorm = "MeanStd"; linear_config.moving_avg = 25; linear_config.individual = False
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
        model = MambaTSv3.Model(config(_args(cfg, raw.shape[1]), raw.shape[1])).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    conditions = {
        "in_distribution": (starts, 0.0, 0.0),
        "late_validation_shift": (midpoint, 0.0, 0.0),
        "missing_channels_10": (starts, 0.10, 0.0),
        "missing_channels_30": (starts, 0.30, 0.0),
        "noise_0p05": (starts, 0.0, 0.05),
        "noise_0p10": (starts, 0.0, 0.10),
    }
    result = {"checkpoint": str(args.checkpoint), "dataset": str(args.dataset), "hybrid": bool(args.hybrid), "test_metrics_produced": False, "conditions": {}}
    with torch.no_grad():
        for index, (name, (condition_starts, mask, noise)) in enumerate(conditions.items()):
            loader = DataLoader(Windows(values, cfg["seq_len"], cfg["pred_len"], condition_starts), batch_size=args.batch_size)
            result["conditions"][name] = {"mse": _evaluate(model, loader, mask, noise, args.seed + index), "windows": int(len(condition_starts)), "mask_fraction": mask, "noise_scale": noise}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
