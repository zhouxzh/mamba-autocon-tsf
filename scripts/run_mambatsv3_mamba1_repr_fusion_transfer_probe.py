#!/usr/bin/env python3
"""Validation-only deployment-fusion probe for forecast-only versus repr-trained cores."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"artifacts/exploratory/mambatsv3_mamba1_representation_relation_probe_20261006"
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_mamba1_repr_fusion_transfer_probe_20261006"
SEED=5990
TARGETS=("pems_bay","metr_la")
ARMS=("plain_fusion","representation_fusion")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def commands(target,arm):
 source_arm="plain" if arm=="plain_fusion" else "representation";source_pretrain=SOURCE/target/f"s{SEED}_{source_arm}_pretrain.pt";reference=SOURCE/target/f"s{SEED}_{source_arm}_finetune_event.json";prefix=OUTPUT/target/f"s{SEED}_{arm}"
 command=load(reference)["command"];command[0]=sys.executable;command=replace(command,"--source-checkpoint",str(source_pretrain));command=replace(command,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")))
 if "--mamba-ts-repr-forecast-fusion" not in command:command.append("--mamba-ts-repr-forecast-fusion")
 if "--evaluate-test" in command:raise ValueError("fusion probe cannot read test rows")
 return command,prefix
def guard(c,p):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+"_telemetry.csv")),"--event-log",str(p.with_name(p.name+"_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"seed":SEED,"targets":list(TARGETS),"arms":list(ARMS),"source_checkpoints":str(SOURCE),"question":"Does a direct-ACF-trained transferable repr_ improve a zero-gated target forecast fusion over forecast-only pretraining?","fixed_contract":{"mixer":"mamba1","patch_len":24,"patch_stride":8,"few_shot_fraction":.1,"target_core_frozen":True,"mamba_scan_count":1,"inference":"representation computed internally; no AutoCon loss or added Mamba scan"}},indent=2)+"\n",encoding="utf-8")
 for target in TARGETS:
  (OUTPUT/target).mkdir(parents=True,exist_ok=True)
  for arm in ARMS:
   command,prefix=commands(target,arm);guard(command,prefix)
 return 0
if __name__=="__main__":raise SystemExit(main())
