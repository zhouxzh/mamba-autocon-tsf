#!/usr/bin/env python3
"""Cross-domain variable-token Mamba representation pretraining with ACF pairs."""
from __future__ import annotations

import argparse
import hashlib
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

from data_provider.data_loader import load_numeric_array_file
from data_provider.samplers import ACFLagPairBatchSampler
from layers.losses import ACFPeakPairInfoNCELoss, ACFRelationLoss, ForecastLagPairContrastiveLoss
from models.NLinearVariableMambaAutoCon import Model
from utils.acf_guidance import fit_signed_acf_prior


class Windows(Dataset):
    def __init__(self, values, seq_len, pred_len, max_windows, lags):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        starts = np.arange(len(values) - self.seq_len - self.pred_len + 1, dtype=np.int64)
        if not len(starts):
            raise ValueError("source prefix is too short")
        self.starts = self._select(starts, max_windows, lags)

    @staticmethod
    def _select(starts, max_windows, lags):
        if len(starts) <= max_windows:
            return starts
        usable = [int(v) for v in sorted(set(lags)) if 0 < int(v) < len(starts)] or [1]
        selected = set()
        for index in range(max_windows // 2):
            lag = usable[index % len(usable)]
            anchor = int(round(index * (len(starts) - lag - 1) / max(1, max_windows // 2 - 1)))
            selected.update((int(starts[anchor]), int(starts[anchor + lag])))
        for value in np.linspace(0, len(starts) - 1, max_windows * 2, dtype=np.int64):
            if len(selected) >= max_windows:
                break
            selected.add(int(value))
        return np.asarray(sorted(selected)[:max_windows], dtype=np.int64)

    def __len__(self): return len(self.starts)
    def __getitem__(self, index):
        start = int(self.starts[index])
        return start, self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def source_spec(value):
    name, path, fmt = value.split(":", 2)
    return name, Path(path).resolve(), fmt


def load_values(path, fmt):
    return (pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=np.float32)
            if fmt == "csv" else load_numeric_array_file(path, data_format=fmt))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_config(args, channels):
    return SimpleNamespace(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=channels, c_out=channels,
                           d_model=args.d_model, d_ff=args.d_ff, AutoCon=True, AutoCon_wnorm="MeanStd",
                           mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv,
                           mamba_ts_residual_init=0.0, mamba_variable_mixer=False,
                           mamba_variable_repr_fusion=bool(args.variable_repr_fusion), mamba_use_acf_memory=False,
                           mamba_bidirectional=False, mamba_acf_channel_order=args.acf_channel_order)


def shared_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def copy_state(target, source):
    state = target.state_dict(); master = source.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            if key in master and value.shape == master[key].shape:
                value.copy_(master[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--calibration-source")
    parser.add_argument("--calibration-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96); parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=32); parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--mamba-d-state", type=int, default=8); parser.add_argument("--mamba-d-conv", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=.6); parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--max-windows", type=int, default=192)
    parser.add_argument("--relation-loss-fraction", type=float, default=.1)
    parser.add_argument("--representation-relation-fraction", type=float, default=.05)
    parser.add_argument("--calibration-relation-loss-fraction", type=float, default=.3)
    parser.add_argument("--calibration-representation-relation-fraction", type=float, default=.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variable-repr-fusion", action="store_true")
    parser.add_argument("--acf-channel-order", choices=["input", "strength", "similarity"], default="input")
    parser.add_argument("--representation-peak-cpc", action="store_true",
                        help="use selected ACF peaks as instance CPC positives and mask high-ACF negatives")
    parser.add_argument("--representation-peak-cpc-positive-mode", choices=["selected", "high_acf"], default="selected")
    parser.add_argument("--representation-peak-cpc-threshold", type=float, default=.3)
    parser.add_argument("--representation-channelwise-acf", action="store_true",
                        help="apply each variable's own train-prefix ACF peaks to its Mamba token state")
    args = parser.parse_args()
    if (args.calibration_source is None) != (args.calibration_fraction is None):
        raise ValueError("calibration source/fraction must pair")
    if args.representation_peak_cpc and args.representation_channelwise_acf:
        raise ValueError("peak CPC and channelwise ACF representation objectives are mutually exclusive")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu: torch.set_num_threads(int(cpu))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    specs = [(source_spec(item), "source", None) for item in args.source]
    if args.calibration_source: specs.append((source_spec(args.calibration_source), "target_calibration", args.calibration_fraction))
    sources=[]
    for (name,path,fmt), role, fraction in specs:
        raw=load_values(path,fmt); rows=int(len(raw)*args.train_fraction*(fraction if role=="target_calibration" else 1.0)); train=raw[:rows]
        prior=fit_signed_acf_prior(train,max_lag=min(720,rows-1),top_k=8,min_lag=1,min_abs_acf=.05,confidence_z=1.96,min_separation=2,detrend="mean",channel_chunk_size=64)
        lags=prior.common_lags(top_k=8) or [1]
        channel_peak_lags=np.zeros((prior.n_channels,8),dtype=np.int64)
        for channel in range(prior.n_channels):
            selected=prior.channel_lags(channel,top_k=8)
            channel_peak_lags[channel,:len(selected)]=selected
        sources.append({"dataset":Windows(train,args.seq_len,args.pred_len,args.max_windows,lags),"acf":prior.acf.astype(np.float32),"mean_acf":prior.acf.mean(axis=0).astype(np.float32),"lags":lags,"channel_peak_lags":channel_peak_lags,"forecast_fraction":args.calibration_relation_loss_fraction if role=="target_calibration" else args.relation_loss_fraction,"repr_fraction":args.calibration_representation_relation_fraction if role=="target_calibration" else args.representation_relation_fraction,"metadata":{"name":name,"path":str(path),"sha256":sha256(path),"role":role,"train_rows":rows,"channels":int(raw.shape[1]),"channel_peak_slots":8,"forecast_relation_fraction":args.calibration_relation_loss_fraction if role=="target_calibration" else args.relation_loss_fraction,"representation_relation_fraction":args.calibration_representation_relation_fraction if role=="target_calibration" else args.representation_relation_fraction}})
    models=[Model(model_config(args,source["metadata"]["channels"])).cuda().train() for source in sources]
    for model, source in zip(models, sources):
        model.set_acf_prior(source["acf"], source["lags"])
    master=models[0]; optimizer=torch.optim.AdamW(master.parameters(),lr=args.learning_rate,weight_decay=1e-5)
    history=[]
    for epoch in range(args.epochs):
        values=[]; forecast_active=[]; repr_active=[]
        for source,model in zip(sources,models):
            loader=DataLoader(source["dataset"],batch_sampler=ACFLagPairBatchSampler(source["dataset"],args.batch_size,source["lags"],seed=args.seed+epoch,origins=source["dataset"].starts,lag_balance="balanced"),num_workers=0)
            forecast_relation=ForecastLagPairContrastiveLoss(source["acf"],selected_lags=source["lags"]).cuda()
            repr_relation = (
                ACFPeakPairInfoNCELoss(source["mean_acf"], source["lags"],
                                       exclude_abs_acf=args.representation_peak_cpc_threshold,
                                       positive_mode=args.representation_peak_cpc_positive_mode).cuda()
                if args.representation_peak_cpc else
                ACFRelationLoss(
                    source["acf"] if args.representation_channelwise_acf else source["mean_acf"],
                    selected_lags=source["lags"], relation_scope="global",
                    target_mode="hard_selected",
                    channel_selected_lags=(source["channel_peak_lags"] if args.representation_channelwise_acf else None),
                    normalize_global_active_anchors=args.representation_channelwise_acf,
                ).cuda()
            )
            for origins,x,y in loader:
                if model is not master: copy_state(model,master)
                origins,x,y=origins.long().cuda(),x.cuda(),y.cuda(); optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
                output=model(x,return_repr=True,return_variable_tokens=args.representation_channelwise_acf)
                if args.representation_channelwise_acf:
                    prediction,representation,variable_tokens=output
                else:
                    prediction,representation=output
                    variable_tokens=None
                predictive=F.smooth_l1_loss(prediction,y)
                forecast_loss=forecast_relation(prediction,origins).mean()
                repr_loss = (
                    repr_relation(F.normalize(representation,dim=-1),origins).mean()
                    if args.representation_peak_cpc else
                    (
                        repr_relation.channelwise_global_contrastive_loss(
                            F.normalize(variable_tokens,dim=-1),origins
                        ).mean()
                        if args.representation_channelwise_acf else
                        repr_relation.global_contrastive_loss(F.normalize(representation,dim=-1),origins).mean()
                    )
                )
                fw=source["forecast_fraction"]*predictive.detach()/forecast_loss.detach().abs().clamp_min(1e-6); rw=source["repr_fraction"]*predictive.detach()/repr_loss.detach().abs().clamp_min(1e-6)
                loss=predictive+fw*forecast_loss+rw*repr_loss; loss.backward()
                repr_active.append(float(
                    repr_relation.last_active_anchor_fraction
                    if args.representation_peak_cpc
                    else repr_relation.last_global_contrastive_active_anchor_fraction.detach().cpu()
                ))
                if model is not master:
                    master_parameters=dict(master.named_parameters())
                    for key,param in model.named_parameters():
                        if key in master_parameters: master_parameters[key].grad=None if param.grad is None else param.grad.detach().clone()
                torch.nn.utils.clip_grad_norm_(master.parameters(),1.0); optimizer.step(); values.append(float(loss.detach())); forecast_active.append(float(forecast_relation.last_active_anchor_fraction))
        row={"epoch":epoch+1,"mean_loss":float(np.mean(values)),"forecast_relation_active_fraction":float(np.mean(forecast_active)),"representation_relation_active_fraction":float(np.mean(repr_active)),"representation_peak_cpc":bool(args.representation_peak_cpc),"representation_peak_cpc_positive_mode":args.representation_peak_cpc_positive_mode,"representation_peak_cpc_threshold":args.representation_peak_cpc_threshold,"representation_channelwise_acf":bool(args.representation_channelwise_acf)}; history.append(row); print(json.dumps(row,sort_keys=True),flush=True)
    payload={"model":shared_state(master),"config":{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},"sources":[source["metadata"] for source in sources],"history":history,"architecture":"variable_token_mamba_acf_shared_encoder","mamba_scan_count":1,"test_metrics_produced":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.output); args.output.with_suffix(".json").write_text(json.dumps({key:value for key,value in payload.items() if key!="model"},indent=2)+"\n",encoding="utf-8"); print(json.dumps({"checkpoint":str(args.output),"test_metrics_produced":False},sort_keys=True))


if __name__=="__main__": main()
