#!/usr/bin/env python3
"""Run a pinned official TSLib TimesNet under the audited split contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts): raise ValueError("no forecast windows")
    def __len__(self): return len(self.starts)
    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def official_model(root: Path):
    root = root.resolve()
    if not (root / "models" / "TimesNet.py").is_file() or not (root / "layers" / "Conv_Blocks.py").is_file():
        raise ValueError("--official-root must be a pinned Time-Series-Library checkout")
    sys.path.insert(0, str(root))
    try: return importlib.import_module("models.TimesNet").Model
    finally: sys.path.pop(0)


def config(args, channels):
    return SimpleNamespace(task_name="long_term_forecast", seq_len=args.seq_len, label_len=48, pred_len=args.pred_len, top_k=args.top_k, num_kernels=args.num_kernels, e_layers=args.e_layers, enc_in=channels, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout, embed="timeF", freq="h", c_out=channels)


def git_commit(root: Path):
    try: return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError): return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--official-commit", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25, 1.0], required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-kernels", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if len(args.official_commit) != 40 or any(char not in "0123456789abcdef" for char in args.official_commit.lower()): raise ValueError("--official-commit must be a 40-character hexadecimal git revision")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1 or args.train_fraction + args.val_fraction >= 1: raise ValueError("invalid chronological split")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = pd.read_csv(args.dataset).iloc[:, 1:].to_numpy(dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all(): raise ValueError("dataset must be a finite numeric CSV after timestamp")
    train_end = int(len(raw) * args.train_fraction); val_end = int(len(raw) * (args.train_fraction + args.val_fraction)); few_end = int(train_end * args.few_shot_fraction)
    mean = raw[:few_end].mean(axis=0, keepdims=True); std = raw[:few_end].std(axis=0, keepdims=True); values = (raw - mean) / np.sqrt(std * std + 1e-5)
    starts = np.arange(0, few_end - args.seq_len - args.pred_len + 1)
    if len(starts) > args.max_train_windows: starts = np.linspace(0, len(starts) - 1, args.max_train_windows, dtype=np.int64)
    val_starts = np.arange(max(0, train_end - args.seq_len), val_end - args.seq_len - args.pred_len + 1); val_starts = val_starts[val_starts + args.seq_len >= train_end]
    test_starts = np.arange(max(0, val_end - args.seq_len), len(raw) - args.seq_len - args.pred_len + 1); test_starts = test_starts[test_starts + args.seq_len >= val_end]
    model = official_model(args.official_root)(config(args, raw.shape[1])).cuda(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, starts), batch_size=args.batch_size, shuffle=True); val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, y in train_loader:
            loss = F.mse_loss(model(x.cuda(), None, None, None), y.cuda()); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader: errors.append(F.mse_loss(model(x.cuda(), None, None, None), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        val = float(torch.cat(errors).mean()); history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val})
        if val < best: best, best_state = val, {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    test_mse = None
    if args.evaluate_test:
        model.load_state_dict(best_state, strict=True); model.eval(); errors = []
        with torch.no_grad():
            for x, y in DataLoader(Windows(values, args.seq_len, args.pred_len, test_starts), batch_size=args.batch_size): errors.append(F.mse_loss(model(x.cuda(), None, None, None), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        test_mse = float(torch.cat(errors).mean())
    payload = {"model": best_state, "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "official_model": "thuml/Time-Series-Library:models/TimesNet.py", "official_commit": git_commit(args.official_root) or args.official_commit, "official_root": str(args.official_root.resolve()), "dataset_sha256": digest(args.dataset), "best_validation_mse": best, "test_mse": test_mse, "history": history, "target_channels": int(raw.shape[1]), "few_train_rows": few_end, "original_train_rows": train_end, "validation_rows": val_end - train_end, "test_rows": len(raw) - val_end, "test_metrics_produced": bool(args.evaluate_test), "selection_done_before_test": bool(args.evaluate_test), "training_contract": "official model with local chronological adapter"}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.output); args.output.with_suffix(".json").write_text(json.dumps({**payload, "model": None}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": bool(args.evaluate_test), "official_commit": payload["official_commit"]}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
