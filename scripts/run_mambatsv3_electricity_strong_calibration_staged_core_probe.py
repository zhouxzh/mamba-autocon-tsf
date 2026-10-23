#!/usr/bin/env python3
"""Validation-only staged shared-core adaptation for Electricity strong calibration."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_staged_core_probe_20261006"
SEED=5948
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def remove(c,f):
 out=list(c)
 while f in out:out.remove(f)
 return out
def command(arm):
 ref=REFERENCE/f"s{SEED}_{arm}_finetune_event.json";prefix=OUTPUT/f"s{SEED}_{arm}";c=load(ref)["command"];c[0]=sys.executable;c=replace(c,"--output",str(prefix.with_name(prefix.name+"_staged_core.pt")));c=replace(c,"--freeze-mamba-epochs","1");c=remove(c,"--freeze-pretrained-core");c += ["--mamba-core-learning-rate","0.0001"]
 if "--evaluate-test" in c:raise ValueError("probe cannot read test rows")
 return c,prefix
def guard(c,p):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+"_telemetry.csv")),"--event-log",str(p.with_name(p.name+"_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seed":SEED,"source_checkpoints":str(REFERENCE),"architecture":"unchanged one-scan Mamba-1 24/8 shared encoder","adaptation":"one frozen target-head warm-up epoch then shared transferred core LR=1e-4 versus target LR=1e-3","paired_control":"active/null use their matched strong-calibration pretraining checkpoints; target adaptation is otherwise identical"},indent=2)+"\n",encoding="utf-8")
 for arm in ("active","null"):
  c,p=command(arm);guard(c,p)
 return 0
if __name__=="__main__":raise SystemExit(main())
