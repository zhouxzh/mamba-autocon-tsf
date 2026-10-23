#!/usr/bin/env python3
"""Current-source same-machine clean-interpreter channel-patch validation run."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay"
OUTPUT=ROOT/"artifacts/reproducibility/channel_patch_clean_interpreter_validation_20261006"
SEED=6042
TRACKED=("models/MambaTS.py","layers/losses.py","scripts/pretrain_mambatsv3_core_autocon.py","scripts/finetune_mambatsv3_core_transfer.py","scripts/finetune_mambatsv3_dlinear_hybrid.py","scripts/run_gpu_guarded.py","dataset/PEMS-BAY/PEMS-BAY.csv","dataset/PEMS07/PEMS07.npz","dataset/PEMS08/PEMS08.npz","dataset/PEMS04/PEMS04.npz")
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):
 value=hashlib.sha256()
 with Path(path).open("rb") as handle:
  for block in iter(lambda:handle.read(1024*1024),b""):value.update(block)
 return value.hexdigest()
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def guard(python,command,prefix,stage):subprocess.run([str(python),str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--python",type=Path,required=True);args=parser.parse_args();python=args.python.resolve()
 if not python.is_file():raise FileNotFoundError(python)
 OUTPUT.mkdir(parents=True,exist_ok=True)
 environment=json.loads(subprocess.check_output([str(python),str(ROOT/"scripts/verify_clean_install_environment.py"),"--output",str(OUTPUT/"environment_check.json")],text=True).strip().splitlines()[-1])
 manifest={"status":"registered_before_training","label":"current-source clean-interpreter channel-patch validation replication","same_machine":True,"independent_machine":False,"validation_only":True,"test_metrics_produced":False,"interpreter":str(python),"environment":environment,"seed":SEED,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"source_pool":["PEMS07","PEMS08","PEMS04"],"arms":["forecast","channel_patch"],"source_hashes":{name:digest(ROOT/name) for name in TRACKED}}
 (OUTPUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 for arm in ("forecast","channel_patch"):
  reference=REFERENCE/f"s6026_{arm}";prefix=OUTPUT/f"s{SEED}_{arm}"
  pre=load(reference.with_name(reference.name+"_pretrain_event.json"))["command"];ft=load(reference.with_name(reference.name+"_finetune_event.json"))["command"];pre[0]=str(python);ft[0]=str(python)
  pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED));ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(SEED))
  if "--evaluate-test" in pre+ft:raise ValueError("clean channel-patch replication cannot read test rows")
  guard(python,pre,prefix,"pretrain");guard(python,ft,prefix,"finetune")
if __name__=="__main__":main()
