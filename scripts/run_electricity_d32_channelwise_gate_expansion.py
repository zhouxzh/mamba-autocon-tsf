#!/usr/bin/env python3
"""Two fresh seeds expanding the d32 channelwise-gate/PatchTST comparison to five."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRETRAIN_REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_probe_20261006"
GATE_REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_probe_20261006"
PATCH_REFERENCE=ROOT/"artifacts/exploratory/official_patchtst_electricity_d32_channelwise_gate_20261006"
OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_gate_expansion_20261006";SEEDS=(6050,6051)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def complete(path):return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seeds":list(SEEDS),"question":"Does two-seed expansion resolve the pre-registered near-tie between d32 channelwise gates and official PatchTST?","existing_three_seed_reference":"electricity_d32_channelwise_residual_gate_replication_rerun_20261006","architecture":{"mixer":"mamba1","patch":"24/24","d_model":32,"d_ff":64,"mamba_scan_count":1},"comparison":"same seed, source/checkpoint initialization, target budget, and official PatchTST geometry; no test reader","no_result_combination":"report separately and combined only by this registered five-seed audit"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  pre_prefix=OUTPUT/f"s{seed}_channel_patch";pre=load(PRETRAIN_REFERENCE/"s6043_channel_patch_pretrain_event.json")["command"];pre[0]=sys.executable;pre=replace(pre,"--output",str(pre_prefix.with_name(pre_prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed));pre_event=pre_prefix.with_name(pre_prefix.name+"_pretrain_event.json")
  if "--evaluate-test" in pre:raise ValueError("candidate cannot read test")
  if not(args.resume and pre_prefix.with_name(pre_prefix.name+"_pretrain.pt").is_file() and complete(pre_event)):guard(pre,pre_prefix,"pretrain")
  gate_prefix=OUTPUT/f"s{seed}_channelwise_gate";gate=load(GATE_REFERENCE/"s6044_channelwise_gate_event.json")["command"];gate[0]=sys.executable;gate=replace(gate,"--source-checkpoint",str(pre_prefix.with_name(pre_prefix.name+"_pretrain.pt")));gate=replace(gate,"--output",str(gate_prefix.with_suffix(".pt")));gate=replace(gate,"--seed",str(seed));gate_event=gate_prefix.with_name(gate_prefix.name+"_finetune_event.json")
  if "--evaluate-test" in gate:raise ValueError("candidate cannot read test")
  if not(args.resume and gate_prefix.with_suffix(".pt").is_file() and complete(gate_event)):guard(gate,gate_prefix,"finetune")
  patch_prefix=OUTPUT/f"electricity_h96_s{seed}_official_patchtst";patch=load(PATCH_REFERENCE/"electricity_h96_s6047_official_patchtst_event.json")["command"];patch[0]=sys.executable;patch=replace(patch,"--output",str(patch_prefix.with_suffix(".pt")));patch=replace(patch,"--seed",str(seed));patch_event=patch_prefix.with_name(patch_prefix.name+"_event.json")
  if "--evaluate-test" in patch:raise ValueError("PatchTST cannot read test")
  if not(args.resume and patch_prefix.with_suffix(".pt").is_file() and complete(patch_event)):guard(patch,patch_prefix,"patch")
if __name__=="__main__":main()
