#!/usr/bin/env python3
"""Train one external low-data baseline on the closed chronological validation split."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DLinear, NLinear, PatchTST, RLinear, TimeMixer, TimesNet, iTransformer


MODELS = {
    "DLinear": DLinear.Model, "NLinear": NLinear.Model, "RLinear": RLinear.Model,
    "PatchTST": PatchTST.Model, "TimeMixer": TimeMixer.Model, "TimesNet": TimesNet.Model,
    "iTransformer": iTransformer.Model,
}


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("no forecast windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def _config(args, channels):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=args.seq_len, pred_len=args.pred_len,
        enc_in=channels, dec_in=channels, c_out=channels, moving_avg=25,
        dropout=0.1, individual=False, AutoCon_wnorm="MeanStd",
        d_model=args.d_model, n_heads=args.n_heads, e_layers=1, d_ff=args.d_ff,
        factor=1, output_attention=False, activation="gelu", embed="timeF", freq="h",
        label_len=48, top_k=3, num_kernels=3, wo_freq=" ",
        timemixer_down_sampling_window=2, timemixer_down_sampling_layers=1,
        timemixer_channel_independence=False, timemixer_decomp_method="moving_avg",
    )


def _forward(model, x):
    return model(x, None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25, 1.0], required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.d_model % args.n_heads:
        raise ValueError("d-model must be divisible by n-heads")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = pd.read_csv(args.dataset).iloc[:, 1:].to_numpy(dtype=np.float32)
    train_end = int(len(raw) * args.train_fraction)
    val_end = int(len(raw) * (args.train_fraction + args.val_fraction))
    few_end = int(train_end * args.few_shot_fraction)
    mean = raw[:few_end].mean(axis=0, keepdims=True)
    std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    starts = np.arange(0, few_end - args.seq_len - args.pred_len + 1)
    if len(starts) > args.max_train_windows:
        starts = np.linspace(0, len(starts) - 1, args.max_train_windows, dtype=np.int64)
    val_starts = np.arange(max(0, train_end - args.seq_len), val_end - args.seq_len - args.pred_len + 1)
    val_starts = val_starts[val_starts + args.seq_len >= train_end]
    test_starts = np.arange(max(0, val_end - args.seq_len), len(raw) - args.seq_len - args.pred_len + 1)
    test_starts = test_starts[test_starts + args.seq_len >= val_end]
    train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    model = MODELS[args.model](_config(args, raw.shape[1])).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, y in train_loader:
            loss = F.mse_loss(_forward(model, x.cuda()), y.cuda())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader:
                errors.append(F.mse_loss(_forward(model, x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        val = float(torch.cat(errors).mean())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val})
        if val < best:
            best, best_state = val, {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    test_mse = None
    if args.evaluate_test:
        model.load_state_dict(best_state, strict=True); model.eval(); errors = []
        test_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, test_starts), batch_size=args.batch_size)
        with torch.no_grad():
            for x, y in test_loader:
                errors.append(F.mse_loss(_forward(model, x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        test_mse = float(torch.cat(errors).mean())
    payload = {"model": best_state, "config": vars(args), "best_validation_mse": best, "test_mse": test_mse, "history": history, "target_channels": int(raw.shape[1]), "few_train_rows": few_end, "original_train_rows": train_end, "validation_rows": val_end-train_end, "test_rows": len(raw)-val_end, "test_metrics_produced": bool(args.evaluate_test)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({**payload, "model": None}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": bool(args.evaluate_test)}, sort_keys=True))


if __name__ == "__main__":
    main()
