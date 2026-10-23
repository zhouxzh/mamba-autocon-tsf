#!/usr/bin/env python3
"""Frozen variable-token Mamba target-head adaptation."""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from data_provider.data_loader import load_numeric_array_file
from utils.acf_guidance import fit_signed_acf_prior
from models.NLinearVariableMambaAutoCon import Model

class Windows(Dataset):
    def __init__(self,v,seq,pred,starts): self.v=torch.as_tensor(v,dtype=torch.float32);self.seq,self.pred,self.starts=seq,pred,starts
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=int(self.starts[i]);return self.v[s:s+self.seq],self.v[s+self.seq:s+self.seq+self.pred]

def cfg(a,c):
    return SimpleNamespace(seq_len=a.seq_len,pred_len=a.pred_len,enc_in=c,c_out=c,d_model=a.d_model,d_ff=a.d_ff,AutoCon=True,AutoCon_wnorm="MeanStd",mamba_d_state=a.mamba_d_state,mamba_d_conv=a.mamba_d_conv,mamba_ts_residual_init=0.,mamba_variable_mixer=False,mamba_variable_repr_fusion=bool(a.variable_repr_fusion),mamba_use_acf_memory=False,mamba_bidirectional=False,mamba_acf_channel_order=a.acf_channel_order)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag,typ,default in (("--seq-len",int,96),("--pred-len",int,96),("--d-model",int,32),("--d-ff",int,64),("--mamba-d-state",int,8),("--mamba-d-conv",int,2),("--train-fraction",float,.6),("--val-fraction",float,.2),("--batch-size",int,4),("--epochs",int,5),("--max-train-windows",int,64),("--seed",int,None)):
        p.add_argument(flag,type=typ,default=default,required=flag=="--seed")
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--source-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--few-shot-fraction",type=float,choices=[.1,.25],required=True);p.add_argument("--variable-repr-fusion",action="store_true");p.add_argument("--acf-channel-order",choices=["input","strength","similarity"],default="input")
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    cpu=os.environ.get("AUTOCON_CPU_THREAD_LIMIT","").strip()
    if cpu:torch.set_num_threads(int(cpu))
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    raw=pd.read_csv(a.dataset).iloc[:,1:].to_numpy(dtype=np.float32) if a.dataset.suffix==".csv" else load_numeric_array_file(a.dataset,data_format="auto")
    te=int(len(raw)*a.train_fraction);ve=int(len(raw)*(a.train_fraction+a.val_fraction));fe=int(te*a.few_shot_fraction);mean=raw[:fe].mean(0,keepdims=True);std=raw[:fe].std(0,keepdims=True);v=(raw-mean)/np.sqrt(std*std+1e-5)
    st=np.arange(fe-a.seq_len-a.pred_len+1,dtype=np.int64)
    if not len(st):raise ValueError("target prefix too short")
    if len(st)>a.max_train_windows:st=np.linspace(0,len(st)-1,a.max_train_windows,dtype=np.int64)
    va=np.arange(max(0,te-a.seq_len),ve-a.seq_len-a.pred_len+1,dtype=np.int64);va=va[va+a.seq_len>=te]
    tr=DataLoader(Windows(v,a.seq_len,a.pred_len,st),batch_size=a.batch_size,shuffle=True);vl=DataLoader(Windows(v,a.seq_len,a.pred_len,va),batch_size=a.batch_size)
    state=torch.load(a.source_checkpoint,map_location="cpu",weights_only=False);m=Model(cfg(a,raw.shape[1])).cuda();m.load_state_dict(state["model"],strict=True)
    prior=fit_signed_acf_prior(raw[:fe],max_lag=min(720,fe-1),top_k=8,min_lag=1,min_abs_acf=.05,confidence_z=1.96,min_separation=2,detrend="mean",channel_chunk_size=64)
    m.set_acf_prior(prior.acf,prior.common_lags(top_k=8) or [1])
    heads={"base_head.weight","base_head.bias","variable_head.weight","variable_head.bias","forecast_scale"}
    for name,param in m.named_parameters():param.requires_grad=name in heads
    opt=torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=1e-3,weight_decay=1e-5);best=float("inf");best_state=None;hist=[]
    for epoch in range(a.epochs):
        m.train();ls=[]
        for x,y in tr:
            pred=m(x.cuda(),return_repr=False)[0];loss=F.mse_loss(pred,y.cuda());opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls.append(float(loss.detach()))
        m.eval();errs=[]
        with torch.no_grad():
            for x,y in vl:errs.append(F.mse_loss(m(x.cuda(),return_repr=False)[0],y.cuda(),reduction="none").mean(dim=(1,2)).cpu())
        score=float(torch.cat(errs).mean());hist.append({"epoch":epoch+1,"train_loss":float(np.mean(ls)),"validation_mse":score,"variable_mamba_frozen":True,"forecast_heads_trainable":True})
        if score<best:best=score;best_state={k:v.detach().cpu() for k,v in m.state_dict().items()}
        print(json.dumps(hist[-1],sort_keys=True),flush=True)
    result={"model":best_state,"config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},"best_validation_mse":best,"history":hist,"architecture":"variable_token_mamba_acf_shared_encoder","test_metrics_produced":False};a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output);a.output.with_suffix(".json").write_text(json.dumps({**result,"model":None},indent=2)+"\n",encoding="utf-8");print(json.dumps({"best_validation_mse":best,"test_metrics_produced":False},sort_keys=True))
if __name__=="__main__":main()
