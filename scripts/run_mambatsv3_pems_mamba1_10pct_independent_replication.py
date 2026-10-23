#!/usr/bin/env python3
"""Independent five-seed PEMS-BAY 10% Mamba-1 active/null validation replication."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006";OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006";SEEDS=(6001,6002,6003,6004,6005)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def commands(seed,arm):
 ref=REFERENCE/f"s5960_{arm}";prefix=OUTPUT/f"s{seed}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];pre[0]=sys.executable;pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed));ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];ft[0]=sys.executable;ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(seed))
 if "--evaluate-test" in pre+ft:raise ValueError("replication cannot read test rows")
 return pre,ft,prefix
def guard(c,p,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def complete(p):return p.is_file() and load(p).get("status")=="completed" and load(p).get("trigger") is None
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"seeds":list(SEEDS),"reference_protocol":str(REFERENCE),"replication_status":"independent five-seed validation replication after the completed three-seed external-stress extension","mamba_contract":{"mixer":"mamba1","patch_len":24,"patch_stride":8,"target_core_frozen":True},"source_pool":["PEMS07","PEMS08","PEMS04"],"no_result_combination":"reported separately from seeds 5960-5962"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  for arm in ("active","null"):
   pre,ft,prefix=commands(seed,arm)
   if not(args.resume and prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))):guard(pre,prefix,"pretrain")
   if not(args.resume and prefix.with_name(prefix.name+"_frozen_core.pt").is_file() and complete(prefix.with_name(prefix.name+"_finetune_event.json"))):guard(ft,prefix,"finetune")
 return 0
if __name__=="__main__":raise SystemExit(main())
