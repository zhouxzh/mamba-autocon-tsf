#!/usr/bin/env python3
"""Same-seed official PatchTST comparator for current-geometry 25% SSL."""

from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/official_patchtst_channel_patch_25pct_replication_20261006"
REFERENCE={
 "pems_bay":(ROOT/"artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006","pems_bay_h192_s5960_official_patchtst_event.json",(6036,6037,6038)),
 "metr_la":(ROOT/"artifacts/exploratory/official_patchtst_metr_mamba1_10pct_20261006","metr_la_h192_s5970_official_patchtst_event.json",(6039,6040,6041)),
}
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def complete(path):return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",choices=sorted(REFERENCE),action="append",default=None);parser.add_argument("--resume",action="store_true");args=parser.parse_args();targets=tuple(args.target or sorted(REFERENCE))
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"question":"Same-seed official PatchTST comparator for 25% channel-patch ACF SSL","targets":{name:{"seeds":list(REFERENCE[name][2])} for name in targets},"official":"PatchTST commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9","contract":"official model with local chronological adapter, 25% labels, no test evaluation"},indent=2)+"\n",encoding="utf-8")
 for target in targets:
  root,event,seeds=REFERENCE[target]
  for seed in seeds:
   prefix=OUTPUT/target/f"{target}_h192_s{seed}_official_patchtst";command=load(root/event)["command"];command[0]=sys.executable;command=replace(command,"--output",str(prefix.with_suffix(".pt")));command=replace(command,"--seed",str(seed));command=replace(command,"--few-shot-fraction",".25")
   if "--evaluate-test" in command:raise ValueError("baseline cannot read test rows")
   prefix.parent.mkdir(parents=True,exist_ok=True)
   if not(args.resume and prefix.with_suffix(".pt").is_file() and complete(prefix.with_name(prefix.name+"_event.json"))):guard(command,prefix)
 return 0
if __name__=="__main__":raise SystemExit(main())
