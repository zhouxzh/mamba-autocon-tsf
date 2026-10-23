#!/usr/bin/env python3
"""Registered fair 10-epoch screen for d32 channelwise gate versus PatchTST."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CANDIDATE_REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006"
PATCH_REFERENCE=ROOT/"artifacts/exploratory/official_patchtst_electricity_d32_channelwise_gate_20261006"
OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_gate_epoch10_screen_20261006";SEED=6047
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seed":SEED,"question":"Does equal 10-epoch budget resolve the near-tie between channelwise gate and official PatchTST?","candidate_pretrain":str(CANDIDATE_REFERENCE/f"s{SEED}_channel_patch_pretrain.pt"),"contract":{"patch":"24/24","d_model":32,"d_ff":64,"target_windows":64,"epochs":10,"mamba_scan_count":1,"channelwise_residual_scale":True}},indent=2)+"\n",encoding="utf-8")
 source=CANDIDATE_REFERENCE/f"s{SEED}_channel_patch_pretrain.pt";candidate=load(CANDIDATE_REFERENCE/f"s{SEED}_channelwise_gate_event.json")["command"];candidate[0]=sys.executable;prefix=OUTPUT/f"s{SEED}_channelwise_gate_epoch10";candidate=replace(candidate,"--source-checkpoint",str(source));candidate=replace(candidate,"--output",str(prefix.with_suffix(".pt")));candidate=replace(candidate,"--epochs","10")
 if "--evaluate-test" in candidate:raise ValueError("candidate screen cannot read test")
 guard(candidate,prefix)
 patch=load(PATCH_REFERENCE/f"electricity_h96_s{SEED}_official_patchtst_event.json")["command"];patch[0]=sys.executable;prefix=OUTPUT/f"electricity_h96_s{SEED}_official_patchtst_epoch10";patch=replace(patch,"--output",str(prefix.with_suffix(".pt")));patch=replace(patch,"--epochs","10")
 if "--evaluate-test" in patch:raise ValueError("PatchTST screen cannot read test")
 guard(patch,prefix)
if __name__=="__main__":main()
