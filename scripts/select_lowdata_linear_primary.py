#!/usr/bin/env python3
"""Choose DLinear or NLinear using only an inner target-training calibration tail."""

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

from models import DLinear, NLinear
from data_provider.samplers import ACFLagPairBatchSampler
from utils.acf_guidance import fit_signed_acf_prior


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values=torch.as_tensor(values,dtype=torch.float32); self.seq_len=seq_len; self.pred_len=pred_len; self.starts=np.asarray(starts,dtype=np.int64)
        if not len(self.starts): raise ValueError("no windows")
    def __len__(self): return len(self.starts)
    def __getitem__(self,index):
        start=int(self.starts[index]); return self.values[start:start+self.seq_len],self.values[start+self.seq_len:start+self.seq_len+self.pred_len]


class OriginWindows(Windows):
    def __getitem__(self,index):
        start=int(self.starts[index]); return start,self.values[start:start+self.seq_len],self.values[start+self.seq_len:start+self.seq_len+self.pred_len]


def paired_starts(total, max_windows, lags):
    starts=np.arange(total,dtype=np.int64)
    if len(starts)<=max_windows: return starts
    lags=sorted({int(lag) for lag in lags if 0<int(lag)<len(starts)}) or [1]
    selected=set()
    for index in range(max_windows//2):
        lag=lags[index%len(lags)]
        anchor=int(round(index*(len(starts)-lag-1)/max(1,max_windows//2-1)))
        selected.update((anchor,anchor+lag))
    for start in np.linspace(0,len(starts)-1,max_windows*2,dtype=np.int64):
        if len(selected)>=max_windows: break
        selected.add(int(start))
    return np.asarray(sorted(selected)[:max_windows],dtype=np.int64)


def config(args, channels):
    return SimpleNamespace(task_name="long_term_forecast",seq_len=args.seq_len,pred_len=args.pred_len,enc_in=channels,dec_in=channels,c_out=channels,moving_avg=25,dropout=.1,individual=False,AutoCon_wnorm="MeanStd")


def run(model_class, cfg, train_loader, calibration_loader, epochs, paired=False):
    model=model_class(cfg).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5); best=float("inf")
    for _ in range(epochs):
        model.train()
        for packed in train_loader:
            _,x,y=packed if paired else (None,*packed)
            loss=F.mse_loss(model(x.cuda(),None,None,None),y.cuda()); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); errors=[]
        with torch.no_grad():
            for x,y in calibration_loader: errors.append(F.mse_loss(model(x.cuda(),None,None,None),y.cuda(),reduction="none").mean(dim=(1,2)).cpu())
        best=min(best,float(torch.cat(errors).mean()))
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seq-len",type=int,default=96); parser.add_argument("--pred-len",type=int,default=96); parser.add_argument("--few-shot-fraction",type=float,choices=[.1,.25],required=True); parser.add_argument("--train-fraction",type=float,default=.6); parser.add_argument("--calibration-fraction",type=float,default=.2); parser.add_argument("--batch-size",type=int,default=4); parser.add_argument("--epochs",type=int,default=5); parser.add_argument("--max-train-windows",type=int,default=64); parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not 0<args.calibration_fraction<.5: raise ValueError("calibration-fraction must be in (0,.5)")
    cpu=os.environ.get("AUTOCON_CPU_THREAD_LIMIT","").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw=pd.read_csv(args.dataset).iloc[:,1:].to_numpy(dtype=np.float32); full_train=int(len(raw)*args.train_fraction); few_end=int(full_train*args.few_shot_fraction); selection_end=int(few_end*(1-args.calibration_fraction))
    mean=raw[:selection_end].mean(axis=0,keepdims=True); std=raw[:selection_end].std(axis=0,keepdims=True); values=(raw-mean)/np.sqrt(std*std+1e-5)
    train_starts=np.arange(0,selection_end-args.seq_len-args.pred_len+1)
    if len(train_starts)>args.max_train_windows: train_starts=np.linspace(0,len(train_starts)-1,args.max_train_windows,dtype=np.int64)
    calibration_starts=np.arange(max(0,selection_end-args.seq_len),few_end-args.seq_len-args.pred_len+1); calibration_starts=calibration_starts[calibration_starts+args.seq_len>=selection_end]
    train_loader=DataLoader(Windows(values,args.seq_len,args.pred_len,train_starts),batch_size=args.batch_size,shuffle=True); calibration_loader=DataLoader(Windows(values,args.seq_len,args.pred_len,calibration_starts),batch_size=args.batch_size)
    prior=fit_signed_acf_prior(raw[:selection_end],max_lag=min(720,selection_end-1),top_k=8,min_lag=1,min_abs_acf=.05,confidence_z=1.96,min_separation=2,detrend="mean",channel_chunk_size=64)
    lags=list(prior.common_lags(top_k=8)) or [1]
    paired_starts_value=paired_starts(selection_end-args.seq_len-args.pred_len+1,args.max_train_windows,lags)
    paired_dataset=OriginWindows(values,args.seq_len,args.pred_len,paired_starts_value)
    paired_loader=DataLoader(paired_dataset,batch_sampler=ACFLagPairBatchSampler(paired_dataset,batch_size=args.batch_size,lags=lags,lag_balance="balanced",seed=args.seed,origins=paired_starts_value),num_workers=0)
    scores={}
    for name,klass in (("DLinear",DLinear.Model),("NLinear",NLinear.Model)):
        for sampler_name,loader,paired in (("uniform",train_loader,False),("acf_paired",paired_loader,True)):
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            scores[f"{name}_{sampler_name}"]=run(klass,config(args,raw.shape[1]),loader,calibration_loader,args.epochs,paired)
    selected=min(scores,key=scores.get)
    model_name,sampler_name=selected.split("_",1)
    result={"selected_model":model_name,"selected_sampler":sampler_name,"inner_calibration_mse":scores,"selection_lags":lags,"few_train_rows":few_end,"selection_train_rows":selection_end,"calibration_rows":few_end-selection_end,"test_metrics_produced":False,"main_validation_accessed":False,"config":vars(args)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,default=str)+"\n",encoding="utf-8"); print(json.dumps(result,sort_keys=True,default=str))


if __name__=="__main__": main()
