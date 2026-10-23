#!/usr/bin/env python3
"""Evaluate a frozen low-data checkpoint on its chronological test partition.

This reader never trains, selects, or mutates a checkpoint. It is intended for
an already frozen validation release only: model selection occurs in the saved
checkpoint's training run, and this script then reads the final chronological
test block exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DLinear, MambaTSv3, NLinear, PatchTST, RLinear, TimeMixer, TimesNet, iTransformer
from models.MambaDLinearHybrid import Model as Hybrid
from scripts.finetune_mambatsv3_core_transfer import config, model_output
from scripts.evaluate_mambatsv3_lowdata_robustness import _args
from data_provider.data_loader import load_numeric_array_file


BASELINES = {
    "DLinear": DLinear.Model, "NLinear": NLinear.Model, "RLinear": RLinear.Model,
    "PatchTST": PatchTST.Model, "TimeMixer": TimeMixer.Model, "TimesNet": TimesNet.Model,
    "iTransformer": iTransformer.Model,
}


def _load_target_values(path: Path) -> np.ndarray:
    """Match the trainer's CSV/NPZ/TXT target loading contract."""
    return load_numeric_array_file(path, data_format="auto")


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("no held-out test windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def _baseline_config(saved_config, channels):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=int(saved_config["seq_len"]),
        pred_len=int(saved_config["pred_len"]), enc_in=channels, dec_in=channels,
        c_out=channels, moving_avg=25, dropout=0.1, individual=False,
        AutoCon_wnorm="MeanStd", d_model=int(saved_config["d_model"]),
        n_heads=int(saved_config.get("n_heads", 4)), e_layers=1,
        d_ff=int(saved_config["d_ff"]), factor=1, output_attention=False,
        activation="gelu", embed="timeF", freq="h",
        # The baseline trainer intentionally relies on PatchTST's published
        # local defaults. Persist them here so a frozen checkpoint can be
        # reconstructed by this read-only test evaluator.
        patch_len=int(saved_config.get("patch_len", 16)),
        patch_stride=int(saved_config.get("patch_stride", 8)),
        label_len=int(saved_config.get("label_len", 48)),
        top_k=int(saved_config.get("top_k", 3)),
        num_kernels=int(saved_config.get("num_kernels", 3)),
        wo_freq=str(saved_config.get("wo_freq", " ")),
        timemixer_down_sampling_window=int(saved_config.get("timemixer_down_sampling_window", 2)),
        timemixer_down_sampling_layers=int(saved_config.get("timemixer_down_sampling_layers", 1)),
        timemixer_channel_independence=bool(saved_config.get("timemixer_channel_independence", False)),
        timemixer_decomp_method=str(saved_config.get("timemixer_decomp_method", "moving_avg")),
    )


def _model(saved, kind, baseline, channels):
    cfg = saved["config"]
    if kind == "hybrid":
        mamba = MambaTSv3.Model(config(_args(cfg, channels), channels)).cuda()
        linear_cfg = config(_args(cfg, channels), channels)
        linear_cfg.AutoCon_wnorm = "MeanStd"; linear_cfg.moving_avg = 25; linear_cfg.individual = False
        linear = {"DLinear": DLinear.Model, "NLinear": NLinear.Model}[cfg["linear_model"]](linear_cfg).cuda()
        model = Hybrid(
            mamba, linear, linear_only=bool(cfg["linear_only"]),
            mamba_correction_output=bool(cfg.get("mamba_correction_output", False)),
            channelwise_residual_scale=bool(cfg.get("mamba_channelwise_residual_scale", False)),
            centered_channelwise_residual_scale=bool(cfg.get("mamba_centered_channelwise_residual_scale", False)),
            channelwise_horizon_residual_scale=bool(cfg.get("mamba_channelwise_horizon_residual_scale", False)),
            channelwise_residual_bias=bool(cfg.get("mamba_channelwise_residual_bias", False)),
            shared_channel_statistic_residual_gate=bool(cfg.get("mamba_shared_channel_statistic_residual_gate", False)),
            shared_channel_statistic_gate_rank=int(cfg.get("mamba_shared_channel_statistic_gate_rank", 8)),
        ).cuda()
    else:
        expected = saved.get("config", {}).get("model")
        if baseline not in BASELINES:
            raise ValueError("unsupported baseline: {}".format(baseline))
        if expected and expected != baseline:
            raise ValueError("checkpoint model does not match --baseline")
        model = BASELINES[baseline](_baseline_config(cfg, channels)).cuda()
    model.load_state_dict(saved["model"], strict=True)
    return model.eval()


def _prediction(model, kind, x):
    """Respect the separate Mamba and baseline forecasting call contracts."""
    if kind == "hybrid":
        return model(x)
    return model(x, None, None, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["hybrid", "baseline"], required=True)
    parser.add_argument("--baseline", choices=sorted(BASELINES))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.kind == "baseline" and args.baseline is None:
        raise ValueError("--baseline is required for baseline checkpoints")
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
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    raw = _load_target_values(args.dataset)
    train_end = int(len(raw) * float(cfg["train_fraction"]))
    val_end = int(len(raw) * (float(cfg["train_fraction"]) + float(cfg["val_fraction"])))
    few_end = int(train_end * float(cfg["few_shot_fraction"]))
    mean = raw[:few_end].mean(axis=0, keepdims=True)
    std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    seq_len, pred_len = int(cfg["seq_len"]), int(cfg["pred_len"])
    starts = np.arange(max(0, val_end - seq_len), len(raw) - seq_len - pred_len + 1)
    starts = starts[starts + seq_len >= val_end]
    model = _model(saved, args.kind, args.baseline, raw.shape[1])
    errors = []
    with torch.no_grad():
        for x, y in DataLoader(Windows(values, seq_len, pred_len, starts), batch_size=args.batch_size):
            prediction = _prediction(model, args.kind, x.cuda())
            errors.append(F.mse_loss(prediction, y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
    result = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "kind": args.kind, "baseline": args.baseline,
        "test_mse": float(torch.cat(errors).mean()), "test_windows": int(len(starts)),
        "test_metrics_produced": True,
        "selection_done_before_test": True,
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
