#!/usr/bin/env python3
"""Closed low-data adaptive-linear / pretrained one-scan Mamba hybrid."""

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

from models import DLinear, MambaTSv3, NLinear
from models.MambaAdaptiveLinearHybrid import Model as Hybrid
from scripts.finetune_mambatsv3_core_transfer import Windows, config, transfer_core, json_compatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96); parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24); parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=24); parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--mamba-d-state", type=int, default=8); parser.add_argument("--mamba-d-conv", type=int, default=2); parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--context-rank", type=int, default=16); parser.add_argument("--channel-agnostic-scan", action="store_true")
    parser.add_argument("--transfer-prediction-head", action="store_true"); parser.add_argument("--prediction-head-transfer-mode", choices=["tile", "exact"], default="exact")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25], required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6); parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--max-train-windows", type=int, default=64); parser.add_argument("--freeze-mamba-epochs", type=int, default=1); parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.linear_only and args.source_checkpoint: raise ValueError("linear-only control must not receive a source checkpoint")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    cpu=os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw=pd.read_csv(args.dataset).iloc[:,1:].to_numpy(dtype=np.float32); train_end=int(len(raw)*args.train_fraction); val_end=int(len(raw)*(args.train_fraction+args.val_fraction)); few_end=int(train_end*args.few_shot_fraction)
    mean=raw[:few_end].mean(axis=0,keepdims=True); std=raw[:few_end].std(axis=0,keepdims=True); values=(raw-mean)/np.sqrt(std*std+1e-5)
    starts=np.arange(0,few_end-args.seq_len-args.pred_len+1)
    if len(starts)>args.max_train_windows: starts=np.linspace(0,len(starts)-1,args.max_train_windows,dtype=np.int64)
    val_starts=np.arange(max(0,train_end-args.seq_len),val_end-args.seq_len-args.pred_len+1); val_starts=val_starts[val_starts+args.seq_len>=train_end]
    train_loader=DataLoader(Windows(values,args.seq_len,args.pred_len,starts),batch_size=args.batch_size,shuffle=True); val_loader=DataLoader(Windows(values,args.seq_len,args.pred_len,val_starts),batch_size=args.batch_size)
    mamba=MambaTSv3.Model(config(args,raw.shape[1])).cuda(); transfer=None if args.source_checkpoint is None else transfer_core(mamba,args.source_checkpoint,args.transfer_prediction_head,args.prediction_head_transfer_mode)
    linear_config=config(args,raw.shape[1]); linear_config.AutoCon_wnorm="MeanStd"; linear_config.moving_avg=25; linear_config.individual=False
    model=Hybrid(mamba,DLinear.Model(linear_config).cuda(),NLinear.Model(linear_config).cuda(),linear_only=args.linear_only).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    best,best_state,history=float("inf"),None,[]
    for epoch in range(args.epochs):
        frozen=not args.linear_only and epoch<args.freeze_mamba_epochs
        for p in model.mamba.parameters(): p.requires_grad=not frozen
        model.train(); losses=[]
        for x,y in train_loader:
            loss=F.mse_loss(model(x.cuda()),y.cuda()); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); errors=[]
        with torch.no_grad():
            gate_values=[]
            for x,y in val_loader:
                x=x.cuda(); errors.append(F.mse_loss(model(x),y.cuda(),reduction="none").mean(dim=(1,2)).cpu()); gate_values.append(torch.sigmoid(model.base_gate(model._features(x))).mean().cpu())
        val=float(torch.cat(errors).mean()); history.append({"epoch":epoch+1,"train_mse":float(np.mean(losses)),"validation_mse":val,"mamba_frozen":frozen,"residual_scale":float(torch.tanh(model.residual_scale).detach()),"dlinear_gate_mean":float(torch.stack(gate_values).mean())})
        if val<best: best,best_state=val,{k:v.detach().cpu() for k,v in model.state_dict().items()}
        print(json.dumps(history[-1],sort_keys=True),flush=True)
    payload={"model":best_state,"config":json_compatible(vars(args)),"best_validation_mse":best,"history":history,"transfer":transfer,"target_channels":int(raw.shape[1]),"few_train_rows":few_end,"original_train_rows":train_end,"validation_rows":val_end-train_end,"test_metrics_produced":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.output); args.output.with_suffix(".json").write_text(json.dumps({**payload,"model":None},indent=2)+"\n",encoding="utf-8"); print(json.dumps({"best_validation_mse":best,"test_metrics_produced":False},sort_keys=True))


if __name__ == "__main__": main()
