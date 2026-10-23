#!/usr/bin/env python3
"""Validation-only multi-positive ACF CPC gate for variable-token Mamba."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/variable_mamba_acf_cpc_probe_20261007"
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_multipositive_probe_20261008"
SEED=6107
ARMS=("selected_cpc","high_acf_cpc")
def load(p):return json.loads(p.read_text())
def replace(c,f,v):r=list(c);r[r.index(f)+1]=v;return r
def append(c,*items):return c if items[0] in c else [*c,*items]
def complete(p):return p.is_file() and load(p).get("status")=="completed" and load(p).get("trigger") is None
def guard(c,p,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seed":SEED,"arms":list(ARMS),"question":"Do all train-prefix high-ACF lag relations as multi-positive CPC targets improve variable-token transfer over sparse selected peaks?","fixed_contract":{"backbone":"one variable-axis Mamba scan","d_model":32,"d_ff":64,"target_variable_mamba_frozen":True,"representation_cpc_fraction":.05,"high_acf_threshold":.3},"only_difference":"selected_cpc uses top-k ACF peaks as positives; high_acf_cpc treats every lag with abs(mean train-prefix ACF)>=.3 as a signed positive and only lower-ACF lags as negatives","screen_rule":"expand high_acf only if validation MSE is at least .25 percent below selected CPC"}
 (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
 ref=REFERENCE/"s6106_peak_cpc"
 for arm in ARMS:
  prefix=OUTPUT/f"s{SEED}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];pre[0]=ft[0]=sys.executable;pre=append(pre,"--representation-peak-cpc-positive-mode","high_acf" if arm=="high_acf_cpc" else "selected");pre=append(pre,"--representation-peak-cpc-threshold",".3");pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED));ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_head.pt")));ft=replace(ft,"--seed",str(SEED))
  if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))):guard(pre,prefix,"pretrain")
  if not(prefix.with_name(prefix.name+"_head.pt").is_file() and complete(prefix.with_name(prefix.name+"_finetune_event.json"))):guard(ft,prefix,"finetune")
if __name__=="__main__":main()
