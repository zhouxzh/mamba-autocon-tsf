#!/usr/bin/env python3
"""Run current PEMS-BAY SSL active/null validation replication in a clean interpreter."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006"
DEFAULT_OUTPUT=ROOT/"artifacts/reproducibility/current_clean_interpreter_validation_20261006"
SEED=6011
TRACKED=("models/MambaTS.py","scripts/pretrain_mambatsv3_core_autocon.py","scripts/finetune_mambatsv3_dlinear_hybrid.py","scripts/run_gpu_guarded.py","dataset/PEMS-BAY/PEMS-BAY.csv","dataset/PEMS07/PEMS07.npz","dataset/PEMS08/PEMS08.npz","dataset/PEMS04/PEMS04.npz")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def commands(python,output,arm):
 ref=REFERENCE/f"s6001_{arm}";prefix=output/f"s{SEED}_{arm}"
 pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];pre[0]=str(python);pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED))
 ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];ft[0]=str(python);ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(SEED))
 if "--evaluate-test" in pre+ft:raise ValueError("clean replication must not read test rows")
 return pre,ft,prefix
def guard(python,c,p,stage):subprocess.run([str(python),str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--python",type=Path,required=True);parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args();python=args.python.resolve()
 if not python.is_file():raise FileNotFoundError(python)
 args.output.mkdir(parents=True,exist_ok=True)
 environment=json.loads(subprocess.check_output([str(python),str(ROOT/"scripts/verify_clean_install_environment.py"),"--output",str(args.output/"environment_check.json")],text=True).strip().splitlines()[-1])
 manifest={"status":"registered_before_training","label":"current-source clean-interpreter validation replication","same_machine":True,"independent_machine":False,"validation_only":True,"test_metrics_produced":False,"interpreter":str(python),"environment":environment,"seed":SEED,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"source_pool":["PEMS07","PEMS08","PEMS04"],"protocol_reference":str(REFERENCE),"source_hashes":{name:digest(ROOT/name) for name in TRACKED}}
 (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 for arm in ("active","null"):
  pre,ft,prefix=commands(python,args.output,arm);guard(python,pre,prefix,"pretrain");guard(python,ft,prefix,"finetune")
 return 0
if __name__=="__main__":raise SystemExit(main())
