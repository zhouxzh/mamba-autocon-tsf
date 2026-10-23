#!/usr/bin/env python3
"""Run a pinned official TimeMixer model with the audited chronology."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import load_numeric_array_file


SOURCES = (
    ("models/TimeMixer.py", "official_model_sha256"),
    ("layers/Autoformer_EncDec.py", "official_decomposition_sha256"),
    ("layers/Embed.py", "official_embed_sha256"),
    ("layers/StandardNorm.py", "official_normalization_sha256"),
)


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("no forecast windows")

    def __len__(self): return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_model(root):
    root = Path(root).resolve()
    if not (root / "models" / "TimeMixer.py").is_file():
        raise ValueError("--official-root must contain models/TimeMixer.py")
    for name in tuple(sys.modules):
        if name in {"models", "layers"} or name.startswith(("models.", "layers.")):
            del sys.modules[name]
    local_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    sys.path[:] = [str(root), *[entry for entry in original_path if Path(entry or ".").resolve() != local_root]]
    try:
        module = importlib.import_module("models.TimeMixer")
    finally:
        sys.path[:] = original_path
    if Path(module.__file__).resolve() != (root / "models" / "TimeMixer.py").resolve():
        raise RuntimeError("official TimeMixer import resolved outside --official-root")
    return module.Model


def cfg(args, channels):
    return SimpleNamespace(
        task_name="long_term_forecast", seq_len=args.seq_len, label_len=48,
        pred_len=args.pred_len, down_sampling_window=args.down_sampling_window,
        channel_independence=args.channel_independence, e_layers=args.e_layers,
        moving_avg=25, enc_in=channels, use_future_temporal_feature=0,
        down_sampling_method="avg", down_sampling_layers=args.down_sampling_layers,
        use_norm=1, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout,
        decomp_method="moving_avg", c_out=channels, embed="timeF", freq="h",
        top_k=3, num_kernels=3,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--official-commit", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-format", choices=["auto", "npz", "txt", "csv_numeric"], default="auto")
    parser.add_argument("--data-key", type=str, default="data")
    parser.add_argument("--data-feature-index", type=int, default=0)
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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--down-sampling-window", type=int, default=2)
    parser.add_argument("--down-sampling-layers", type=int, default=1)
    parser.add_argument("--channel-independence", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if len(args.official_commit) != 40:
        raise ValueError("--official-commit must be a 40-character revision")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("invalid chronological split")
    if args.down_sampling_window < 2 or args.down_sampling_layers < 0:
        raise ValueError("invalid TimeMixer down-sampling")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = load_numeric_array_file(
        args.dataset, data_format=args.data_format, data_key=args.data_key,
        data_feature_index=args.data_feature_index,
    )
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
    model = official_model(args.official_root)(cfg(args, raw.shape[1])).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, y in train_loader:
            loss = F.mse_loss(model(x.cuda(), None, None, None), y.cuda())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader:
                errors.append(F.mse_loss(model(x.cuda(), None, None, None), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        val = float(torch.cat(errors).mean())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val})
        if val < best:
            best, best_state = val, {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    test_mse = None
    if args.evaluate_test:
        model.load_state_dict(best_state, strict=True); model.eval(); errors = []
        with torch.no_grad():
            for x, y in DataLoader(Windows(values, args.seq_len, args.pred_len, test_starts), batch_size=args.batch_size):
                errors.append(F.mse_loss(model(x.cuda(), None, None, None), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        test_mse = float(torch.cat(errors).mean())
    payload = {
        "model": best_state, "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "official_model": "kwuking/TimeMixer:models/TimeMixer.py", "official_commit": args.official_commit,
        "official_root": str(args.official_root.resolve()), "dataset_sha256": sha256(args.dataset),
        **{key: sha256(args.official_root / relative) for relative, key in SOURCES},
        "best_validation_mse": best, "test_mse": test_mse, "history": history,
        "target_channels": int(raw.shape[1]), "few_train_rows": few_end,
        "original_train_rows": train_end, "validation_rows": val_end - train_end,
        "test_rows": len(raw) - val_end, "test_metrics_produced": bool(args.evaluate_test),
        "selection_done_before_test": bool(args.evaluate_test),
        "training_contract": "official model with local chronological adapter",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({**payload, "model": None}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": bool(args.evaluate_test), "official_commit": args.official_commit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
