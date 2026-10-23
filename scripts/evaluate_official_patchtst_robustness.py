#!/usr/bin/env python3
"""Read-only validation stress evaluation for a locked official PatchTST checkpoint."""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.official_patchtst_adapter import config, digest, official_model
from data_provider.data_loader import load_numeric_array_file

class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, starts):
        self.values=torch.as_tensor(values,dtype=torch.float32);self.seq_len=int(seq_len);self.pred_len=int(pred_len);self.starts=np.asarray(starts,dtype=np.int64)
    def __len__(self): return len(self.starts)
    def __getitem__(self,index):
        start=int(self.starts[index]);return self.values[start:start+self.seq_len],self.values[start+self.seq_len:start+self.seq_len+self.pred_len]

def evaluate(model,loader,mask_fraction,noise_scale,seed):
    generator=torch.Generator(device="cuda").manual_seed(seed);errors=[]
    for x,y in loader:
        x,y=x.cuda(),y.cuda()
        if mask_fraction:
            count=max(1,int(round(x.shape[-1]*mask_fraction)));chosen=torch.randperm(x.shape[-1],generator=generator,device=x.device)[:count];x=x.clone();x[:,:,chosen]=0
        if noise_scale:
            scale=x.std(dim=1,keepdim=True,unbiased=False).clamp_min(1e-6);x=x+noise_scale*scale*torch.randn(x.shape,generator=generator,device=x.device,dtype=x.dtype)
        errors.append(F.mse_loss(model(x),y,reduction="none").mean(dim=(1,2)).cpu())
    return float(torch.cat(errors).mean())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--dataset",type=Path,required=True);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--batch-size",type=int,default=4);parser.add_argument("--seed",type=int,required=True);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    cpu=os.environ.get("AUTOCON_CPU_THREAD_LIMIT","").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if saved.get("test_metrics_produced"):raise ValueError("checkpoint was test-exposed")
    cfg=saved["config"];root=Path(saved["official_root"])
    for relative,key in (("models/PatchTST.py","official_model_sha256"),("layers/PatchTST_backbone.py","official_backbone_sha256")):
        if digest(root/relative)!=saved[key]:raise ValueError("official source hash mismatch")
    raw=load_numeric_array_file(args.dataset);train_end=int(len(raw)*float(cfg["train_fraction"]));val_end=int(len(raw)*(float(cfg["train_fraction"])+float(cfg["val_fraction"])));few_end=int(train_end*float(cfg["few_shot_fraction"]));mean=raw[:few_end].mean(axis=0,keepdims=True);std=raw[:few_end].std(axis=0,keepdims=True);values=(raw-mean)/np.sqrt(std*std+1e-5)
    starts=np.arange(max(0,train_end-int(cfg["seq_len"])),val_end-int(cfg["seq_len"])-int(cfg["pred_len"])+1);starts=starts[starts+int(cfg["seq_len"])>=train_end];late=starts[len(starts)//2:]
    model=official_model(root)(config(argparse.Namespace(**cfg),raw.shape[1])).cuda().eval();model.load_state_dict(saved["model"],strict=True)
    conditions={"in_distribution":(starts,0.,0.),"late_validation_shift":(late,0.,0.),"missing_channels_10":(starts,.1,0.),"missing_channels_30":(starts,.3,0.),"noise_0p05":(starts,0.,.05),"noise_0p10":(starts,0.,.1)}
    result={"checkpoint":str(args.checkpoint),"dataset":str(args.dataset),"official_commit":saved["official_commit"],"official_model_sha256":saved["official_model_sha256"],"official_backbone_sha256":saved["official_backbone_sha256"],"test_metrics_produced":False,"conditions":{}}
    with torch.no_grad():
        for index,(name,(condition_starts,mask,noise)) in enumerate(conditions.items()):
            loader=DataLoader(Windows(values,cfg["seq_len"],cfg["pred_len"],condition_starts),batch_size=args.batch_size)
            result["conditions"][name]={"mse":evaluate(model,loader,mask,noise,args.seed+index),"windows":int(len(condition_starts)),"mask_fraction":mask,"noise_scale":noise}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
