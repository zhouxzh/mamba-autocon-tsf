#!/usr/bin/env python3
"""Validation-only PEMS-BAY external gate probe for channel-patch SSL."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay";OUTPUT=ROOT/"artifacts/exploratory/pems_channelwise_residual_gate_probe_20261006";SEED=6026
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def main():
 source=REFERENCE/f"s{SEED}_channel_patch_pretrain.pt";event=REFERENCE/f"s{SEED}_channel_patch_finetune_event.json";prefix=OUTPUT/f"s{SEED}_channelwise_gate"
 if not source.is_file():raise FileNotFoundError(source)
 command=list(load(event)["command"]);command[0]=sys.executable;command=replace(command,"--source-checkpoint",str(source));command=replace(command,"--output",str(prefix.with_suffix(".pt")));command += ["--mamba-channelwise-residual-scale"]
 if "--evaluate-test" in command:raise ValueError("PEMS gate probe cannot read test")
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"seed":SEED,"source_pretraining":str(source),"control":str(REFERENCE/f"s{SEED}_channel_patch_frozen_core.pt"),"selection_boundary":"target-channel gate was designed on Electricity; PEMS-BAY is external to that design","mechanism":"zero-initialized target-channel scale for existing correction","mamba_scan_count":1},indent=2)+"\n",encoding="utf-8")
 subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
if __name__=="__main__":main()
