#!/usr/bin/env python3
"""Independent five-seed official PatchTST PEMS-BAY 10% validation replication."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REFERENCE=ROOT/"artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006";OUTPUT=ROOT/"artifacts/exploratory/official_patchtst_pems_mamba1_10pct_independent_replication_20261006";SEEDS=(6001,6002,6003,6004,6005)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def guard(c,p):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+"_telemetry.csv")),"--event-log",str(p.with_name(p.name+"_event.json")),"--",*c],cwd=ROOT,check=True)
def complete(p):return p.is_file() and load(p).get("status")=="completed" and load(p).get("trigger") is None
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"seeds":list(SEEDS),"reference_protocol":str(REFERENCE),"official":"PatchTST commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9","no_result_combination":"reported separately from seeds 5960-5962"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  ref=REFERENCE/"pems_bay_h192_s5960_official_patchtst_event.json";c=load(ref)["command"];c[0]=sys.executable;prefix=OUTPUT/f"pems_bay_h192_s{seed}_official_patchtst";c=replace(c,"--output",str(prefix.with_suffix(".pt")));c=replace(c,"--seed",str(seed))
  if "--evaluate-test" in c:raise ValueError("baseline cannot read test rows")
  if not(args.resume and prefix.with_suffix(".pt").is_file() and complete(prefix.with_name(prefix.name+"_event.json"))):guard(c,prefix)
 return 0
if __name__=="__main__":raise SystemExit(main())
