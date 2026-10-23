#!/usr/bin/env python3
"""Validation-only d32/24 cross-domain SSL bridge probe for Traffic."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_traffic_mamba1_overlap_probe_20261006"
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_traffic_mamba1_d32_stride24_transfer_probe_20261006"
SEED=6006
INITIAL=(
 ROOT/"artifacts/exploratory/pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_20260926/null/checkpoints/long_term_forecast_plan_pems_h96_fullcover_p24_s24_d32_autocon_null_pems03_MambaTSv3AutoCon_h96_s2171_l0_c1351a657278d_MambaTSv3AutoCon_pems03_ftM_sl96_ll48_pl96__hb38645f3b46ce635/checkpoint.pth",
 ROOT/"artifacts/exploratory/pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_20260926/null/checkpoints/long_term_forecast_plan_pems_h96_fullcover_p24_s24_d32_autocon_null_pems04_MambaTSv3AutoCon_h96_s2171_l0_c96ca22b3fcaf_MambaTSv3AutoCon_pems04_ftM_sl96_ll48_pl96__h457439223b99675c/checkpoint.pth",
)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def replace_repeated(c,flag,values):
 out=[];i=0;n=0
 while i<len(c):
  if c[i]==flag:out += [flag,str(values[n])];n+=1;i+=2
  else:out.append(c[i]);i+=1
 if n!=len(values):raise ValueError("initial checkpoint count mismatch")
 return out
def append(c,f,v):return replace(c,f,v) if f in c else list(c)+[f,v]
def commands(arm):
 ref=REFERENCE/f"s5941_{arm}";prefix=OUTPUT/f"s{SEED}_{arm}"
 pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];pre[0]=sys.executable;pre=replace_repeated(pre,"--initial-checkpoint",INITIAL);pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED));pre=replace(pre,"--patch-stride","24");pre=replace(pre,"--d-model","32");pre=replace(pre,"--d-ff","64");pre=replace(pre,"--mamba-headdim","16");pre=append(pre,"--calibration-source","Traffic:dataset/traffic/traffic.csv:csv");pre=append(pre,"--calibration-fraction",".25");pre=append(pre,"--calibration-relation-loss-fraction",".3" if arm=="active" else "0")
 ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];ft[0]=sys.executable;ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(SEED));ft=replace(ft,"--patch-stride","24");ft=replace(ft,"--d-model","32");ft=replace(ft,"--d-ff","64");ft=replace(ft,"--mamba-headdim","16")
 if "--evaluate-test" in pre+ft:raise ValueError("probe cannot read test rows")
 return pre,ft,prefix
def guard(c,p,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 for p in INITIAL:
  if not p.is_file():raise FileNotFoundError(p)
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seed":SEED,"source":"Electricity plus Traffic chronological calibration prefix","initial_checkpoints":[str(p) for p in INITIAL],"architecture":{"mixer":"mamba1","patch_len":24,"patch_stride":24,"d_model":32,"d_ff":64,"mamba_scan_count":1},"protocol":"cross-domain next-patch/forecast-InfoNCE SSL with target core frozen; separate from historical highdim end-to-end release","relation_schedule":{"Electricity_source":.1,"Traffic_calibration_active":.3,"Traffic_calibration_null":0.0}},indent=2)+"\n",encoding="utf-8")
 for arm in ("active","null"):
  pre,ft,prefix=commands(arm);guard(pre,prefix,"pretrain");guard(ft,prefix,"finetune")
 return 0
if __name__=="__main__":raise SystemExit(main())
