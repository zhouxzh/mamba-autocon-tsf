#!/usr/bin/env python3
"""Validation-only active/null Traffic d32 channelwise target-gate probe."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_traffic_mamba1_d32_stride24_transfer_probe_20261006"
OUTPUT=ROOT/"artifacts/exploratory/traffic_d32_channelwise_residual_gate_probe_20261006";SEED=6006
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seed":SEED,"question":"Can target-channel correction scaling recover the active/null relation effect in the d32/24 Traffic bridge?","reference":str(REFERENCE),"arms":["active","null"],"fixed_contract":{"mixer":"mamba1","patch":"24/24","d_model":32,"d_ff":64,"target_core_frozen":True,"mamba_scan_count":1,"target_gate":"zero-initialized per-channel correction scale"}},indent=2)+"\n",encoding="utf-8")
 for arm in ("active","null"):
  source=REFERENCE/f"s{SEED}_{arm}_pretrain.pt";event=REFERENCE/f"s{SEED}_{arm}_finetune_event.json";prefix=OUTPUT/f"s{SEED}_{arm}_channelwise_gate"
  if not source.is_file():raise FileNotFoundError(source)
  command=list(load(event)["command"]);command[0]=sys.executable;command=replace(command,"--source-checkpoint",str(source));command=replace(command,"--output",str(prefix.with_suffix(".pt")));command += ["--mamba-channelwise-residual-scale"]
  if "--evaluate-test" in command:raise ValueError("Traffic gate probe cannot read test")
  guard(command,prefix)
if __name__=="__main__":main()
