#!/usr/bin/env python3
"""Validation-only horizon/channel bias gate over the d32 channelwise scale."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006"
OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_bias_probe_20261006";SEED=6047
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def main():
 source=REFERENCE/f"s{SEED}_channel_patch_pretrain.pt";event=REFERENCE/f"s{SEED}_channelwise_gate_event.json"
 if not source.is_file():raise FileNotFoundError(source)
 command=list(load(event)["command"]);command[0]=sys.executable;prefix=OUTPUT/f"s{SEED}_channelwise_bias"
 command=replace(command,"--source-checkpoint",str(source));command=replace(command,"--output",str(prefix.with_suffix(".pt")));command += ["--mamba-channelwise-residual-bias"]
 if "--evaluate-test" in command:raise ValueError("channelwise bias probe cannot read test rows")
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seed":SEED,"source_pretraining":str(source),"control":str(REFERENCE/f"s{SEED}_channelwise_gate.pt"),"mechanism":"zero-initialized target horizon/channel bias after the zero-initialized channelwise correction scale","mamba_scan_count":1,"shared_encoder_change":False},indent=2)+"\n",encoding="utf-8")
 subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
if __name__=="__main__":main()
