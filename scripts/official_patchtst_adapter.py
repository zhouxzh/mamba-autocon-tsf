#!/usr/bin/env python3
"""Run a fixed official PatchTST checkout under the audited local split contract.

The upstream PatchTST supervised runner hard-codes a 70/10/20 split and tests
immediately after training.  This adapter intentionally imports the official
``models/PatchTST.py`` implementation but owns only the data boundaries and
train/validation/test sequencing needed for a fair controlled comparison.
"""

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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Guarded launchers invoke this file by its absolute path.  Make the local
# numeric-array contract importable before the later official-model isolation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import load_numeric_array_file


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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def official_model(official_root: Path):
    root = official_root.resolve()
    module_path = root / "models" / "PatchTST.py"
    if not module_path.is_file() or not (root / "layers" / "PatchTST_backbone.py").is_file():
        raise ValueError("--official-root must be PatchTST_supervised from the pinned official checkout")
    # No repository model modules are imported before this insertion.  The
    # official module's absolute ``layers.*`` imports therefore resolve inside
    # the fixed external checkout rather than the local implementation.
    # An evaluator can already have imported this repository's local
    # ``models`` package.  Remove that name before importing so the fixed
    # checkout cannot silently resolve to the local PatchTST implementation.
    for name in tuple(sys.modules):
        if name in {"models", "layers"} or name.startswith(("models.", "layers.")):
            del sys.modules[name]
    local_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    sys.path[:] = [
        str(root),
        *[
            entry for entry in original_path
            if Path(entry or ".").resolve() != local_root
        ],
    ]
    try:
        module = importlib.import_module("models.PatchTST")
    finally:
        sys.path[:] = original_path
    if Path(module.__file__).resolve() != module_path.resolve():
        raise RuntimeError("official PatchTST import resolved outside --official-root")
    return module.Model


def config(args, channels):
    return SimpleNamespace(
        enc_in=channels, seq_len=args.seq_len, pred_len=args.pred_len,
        e_layers=args.e_layers, n_heads=args.n_heads, d_model=args.d_model,
        d_ff=args.d_ff, dropout=args.dropout, fc_dropout=args.fc_dropout,
        head_dropout=args.head_dropout, individual=int(args.individual),
        patch_len=args.patch_len, stride=args.patch_stride,
        padding_patch="end", revin=int(args.revin), affine=int(args.affine),
        subtract_last=int(args.subtract_last), decomposition=0, kernel_size=25,
    )


def git_commit(official_root: Path):
    for candidate in (official_root.resolve(), *official_root.resolve().parents):
        try:
            return subprocess.check_output(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--official-commit", type=str, required=True,
                        help="immutable upstream commit recorded for this checkout")
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc-dropout", type=float, default=0.05)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--revin", type=int, choices=[0, 1], default=1)
    parser.add_argument("--affine", type=int, choices=[0, 1], default=0)
    parser.add_argument("--subtract-last", type=int, choices=[0, 1], default=0)
    parser.add_argument("--individual", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if len(args.official_commit) != 40 or any(char not in "0123456789abcdef" for char in args.official_commit.lower()):
        raise ValueError("--official-commit must be a 40-character hexadecimal git revision")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("invalid chronological split")
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
    raw = load_numeric_array_file(
        args.dataset,
        data_format=args.data_format,
        data_key=args.data_key,
        data_feature_index=args.data_feature_index,
    )
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("dataset must be a finite numeric array")
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
    model_class = official_model(args.official_root)
    model = model_class(config(args, raw.shape[1])).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, y in train_loader:
            prediction = model(x.cuda())
            loss = F.mse_loss(prediction, y.cuda())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader:
                errors.append(F.mse_loss(model(x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        validation_mse = float(torch.cat(errors).mean())
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": validation_mse})
        if validation_mse < best:
            best, best_state = validation_mse, {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    test_mse = None
    if args.evaluate_test:
        model.load_state_dict(best_state, strict=True); model.eval(); errors = []
        with torch.no_grad():
            for x, y in DataLoader(Windows(values, args.seq_len, args.pred_len, test_starts), batch_size=args.batch_size):
                errors.append(F.mse_loss(model(x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        test_mse = float(torch.cat(errors).mean())
    payload = {
        "model": best_state, "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "official_model": "yuqinie98/PatchTST:PatchTST_supervised/models/PatchTST.py",
        "official_commit": git_commit(args.official_root) or args.official_commit,
        "official_root": str(args.official_root.resolve()),
        "official_model_sha256": digest(args.official_root / "models" / "PatchTST.py"),
        "official_backbone_sha256": digest(args.official_root / "layers" / "PatchTST_backbone.py"),
        "dataset_sha256": digest(args.dataset), "best_validation_mse": best,
        "test_mse": test_mse, "history": history, "target_channels": int(raw.shape[1]),
        "few_train_rows": few_end, "original_train_rows": train_end,
        "validation_rows": val_end - train_end, "test_rows": len(raw) - val_end,
        "test_metrics_produced": bool(args.evaluate_test),
        "selection_done_before_test": bool(args.evaluate_test),
        "training_contract": "official model with local chronological adapter",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({**payload, "model": None}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": bool(args.evaluate_test), "official_commit": payload["official_commit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
