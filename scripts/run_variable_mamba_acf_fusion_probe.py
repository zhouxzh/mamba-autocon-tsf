#!/usr/bin/env python3
"""Validation-only direct-representation variable Mamba ACF gate."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/variable_mamba_acf_traffic_probe_20261007"
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_fusion_probe_20261007"
SEED=6104
ARMS=("fusion_forecast","fusion_representation_acf")
def load(p):return json.loads(p.read_text())
def replace(c,f,v):r=list(c);r[r.index(f)+1]=v;return r
def append(c,f):return c if f in c else [*c,f]
def complete(p):return p.is_file() and load(p).get("status")=="completed" and load(p).get("trigger") is None
def guard(c,p,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seed":SEED,"arms":list(ARMS),"question":"Does a prediction-connected variable representation let ACF pretraining improve variable-token Mamba transfer?","fixed_contract":{"backbone":"one variable-axis Mamba scan with zero-gated representation-to-token FiLM","d_model":32,"d_ff":64,"target_variable_mamba_frozen":True,"forecast_relation_fractions":{"source":.1,"calibration":.3}},"only_difference":"representation arm enables source and calibration representation ACF InfoNCE at .05; direct fusion is enabled in both arms","screen_rule":"expand only if fusion representation ACF is at least .25 percent below fusion forecast"}
 (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
 ref=REFERENCE/"s6102_representation_acf"
 for arm in ARMS:
  prefix=OUTPUT/f"s{SEED}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];pre[0]=ft[0]=sys.executable;fraction=".05" if arm=="fusion_representation_acf" else "0";pre=replace(pre,"--representation-relation-fraction",fraction);pre=replace(pre,"--calibration-representation-relation-fraction",fraction);pre=append(pre,"--variable-repr-fusion");ft=append(ft,"--variable-repr-fusion");pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED));ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_head.pt")));ft=replace(ft,"--seed",str(SEED))
  if "--evaluate-test" in pre+ft:raise ValueError("fusion probe cannot read test")
  if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))):guard(pre,prefix,"pretrain")
  if not(prefix.with_name(prefix.name+"_head.pt").is_file() and complete(prefix.with_name(prefix.name+"_finetune_event.json"))):guard(ft,prefix,"finetune")
if __name__=="__main__":main()
