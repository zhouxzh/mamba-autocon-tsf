#!/usr/bin/env python3
"""Read the chronological test block of a validation-selected official PatchTST checkpoint."""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.official_patchtst_adapter import Windows, config, digest, git_commit, official_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
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
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    official_root = Path(saved["official_root"])
    if digest(args.dataset) != saved["dataset_sha256"]:
        raise ValueError("dataset hash does not match the validation-selected checkpoint")
    observed_commit = git_commit(official_root)
    if observed_commit is not None and observed_commit != saved["official_commit"]:
        raise ValueError("official PatchTST checkout commit does not match checkpoint")
    for relative, key in (("models/PatchTST.py", "official_model_sha256"), ("layers/PatchTST_backbone.py", "official_backbone_sha256")):
        if key not in saved or digest(official_root / relative) != saved[key]:
            raise ValueError("official PatchTST source hash does not match checkpoint")
    raw = pd.read_csv(args.dataset).iloc[:, 1:].to_numpy(dtype=np.float32)
    train_end = int(len(raw) * float(cfg["train_fraction"]))
    val_end = int(len(raw) * (float(cfg["train_fraction"]) + float(cfg["val_fraction"])))
    few_end = int(train_end * float(cfg["few_shot_fraction"]))
    mean = raw[:few_end].mean(axis=0, keepdims=True)
    std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    starts = np.arange(max(0, val_end - int(cfg["seq_len"])), len(raw) - int(cfg["seq_len"]) - int(cfg["pred_len"]) + 1)
    starts = starts[starts + int(cfg["seq_len"]) >= val_end]
    model = official_model(official_root)(config(argparse.Namespace(**cfg), raw.shape[1])).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    errors = []
    with torch.no_grad():
        for x, y in DataLoader(Windows(values, cfg["seq_len"], cfg["pred_len"], starts), batch_size=args.batch_size):
            errors.append(F.mse_loss(model(x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
    result = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "official_repository": saved["official_model"], "official_commit": saved["official_commit"],
        "official_model_sha256": saved["official_model_sha256"],
        "official_backbone_sha256": saved["official_backbone_sha256"],
        "test_mse": float(torch.cat(errors).mean()), "test_windows": int(len(starts)),
        "test_metrics_produced": True, "selection_done_before_test": True,
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
