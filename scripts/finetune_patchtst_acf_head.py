#!/usr/bin/env python3
"""Adapt only the shared PatchTST ACF prediction head on target few-shot data."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import load_numeric_array_file
from models.PatchTSTACFPretrain import Model


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len, self.starts = int(seq_len), int(pred_len), starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--few-shot-fraction", type=float, choices=[.1, .25], required=True)
    parser.add_argument("--train-fraction", type=float, default=.6)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate_test:
        raise ValueError("this adaptation script is validation-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = (pd.read_csv(args.dataset).iloc[:, 1:].to_numpy(dtype=np.float32)
           if args.dataset.suffix.lower() == ".csv"
           else load_numeric_array_file(args.dataset, data_format="auto"))
    train_end = int(len(raw) * args.train_fraction)
    val_end = int(len(raw) * (args.train_fraction + args.val_fraction))
    few_end = int(train_end * args.few_shot_fraction)
    mean, std = raw[:few_end].mean(axis=0, keepdims=True), raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    starts = np.arange(few_end - args.seq_len - args.pred_len + 1, dtype=np.int64)
    if not len(starts):
        raise ValueError("few-shot target prefix is too short")
    if len(starts) > args.max_train_windows:
        starts = np.linspace(0, len(starts) - 1, args.max_train_windows, dtype=np.int64)
    val_starts = np.arange(max(0, train_end - args.seq_len), val_end - args.seq_len - args.pred_len + 1, dtype=np.int64)
    val_starts = val_starts[val_starts + args.seq_len >= train_end]
    train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    model = Model(args.seq_len, args.pred_len, args.patch_len, args.patch_stride, args.d_model, args.d_ff, args.n_heads, args.e_layers).cuda()
    model.load_state_dict(source["model"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("prediction_head")
    optimizer = torch.optim.AdamW(model.prediction_head.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, y in train_loader:
            prediction = model(x.cuda())
            loss = F.mse_loss(prediction, y.cuda())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader:
                errors.append(F.mse_loss(model(x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        value = float(torch.cat(errors).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_mse": value, "encoder_frozen": True, "prediction_head_trainable": True})
        if value < best:
            best = value
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result = {"model": best_state, "config": config, "best_validation_mse": best, "history": history, "test_metrics_produced": False, "architecture": "shared_channel_independent_patchtst_encoder"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({**result, "model": None}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": False}, sort_keys=True))


if __name__ == "__main__":
    main()
