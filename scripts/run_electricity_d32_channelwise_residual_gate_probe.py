#!/usr/bin/env python3
"""Validation-only target channelwise-residual gate on d32 channel-patch SSL."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_replication_20261006"
OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_probe_20261006"
SEED=6044
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def main():
 source=REFERENCE/f"s{SEED}_channel_patch_pretrain.pt";event=REFERENCE/f"s{SEED}_channel_patch_finetune_event.json"
 if not source.is_file():raise FileNotFoundError(source)
 command=list(load(event)["command"]);command[0]=sys.executable;prefix=OUTPUT/f"s{SEED}_channelwise_gate"
 command=replace(command,"--source-checkpoint",str(source));command=replace(command,"--output",str(prefix.with_suffix(".pt")));command += ["--mamba-channelwise-residual-scale"]
 if "--evaluate-test" in command:raise ValueError("channelwise gate probe cannot read test rows")
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seed":SEED,"source_pretraining":str(source),"control":str(REFERENCE/f"s{SEED}_channel_patch_frozen_core.pt"),"mechanism":"target-only zero-initialized per-channel scale for existing Mamba correction","mamba_scan_count":1,"shared_encoder_change":False},indent=2)+"\n",encoding="utf-8")
 subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
if __name__=="__main__":main()
