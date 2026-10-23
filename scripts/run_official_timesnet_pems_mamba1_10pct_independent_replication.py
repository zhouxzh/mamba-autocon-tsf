#!/usr/bin/env python3
"""Independent five-seed official TimesNet PEMS-BAY 10% validation baseline."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OFFICIAL_ROOT=Path("/tmp/tslib-adapter-84W4AM/Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e")
COMMIT="4e938a1767106324dd753b2a44832bf870a0252e";OUTPUT=ROOT/"artifacts/exploratory/official_timesnet_pems_mamba1_10pct_independent_replication_20261006";SEEDS=(6001,6002,6003,6004,6005)
def guard(c,p):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+"_telemetry.csv")),"--event-log",str(p.with_name(p.name+"_event.json")),"--",*c],cwd=ROOT,check=True)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def complete(p):return p.is_file() and load(p).get("status")=="completed" and load(p).get("trigger") is None
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args()
 if not (OFFICIAL_ROOT/"models/TimesNet.py").is_file():raise FileNotFoundError(OFFICIAL_ROOT)
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"seeds":list(SEEDS),"official_root":str(OFFICIAL_ROOT),"official_commit":COMMIT,"budget":{"seq_len":96,"epochs":5,"max_train_windows":64,"d_model":24,"d_ff":48,"e_layers":1,"top_k":3,"num_kernels":3,"batch_size":2},"comparison":"same split, labels, seeds, epoch and target-window budget as independent Mamba-1 replication; TimesNet batch size follows official adapter memory-safe setting"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  prefix=OUTPUT/f"pems_bay_h192_s{seed}_official_timesnet";out=prefix.with_suffix(".pt")
  if args.resume and out.is_file() and prefix.with_name(prefix.name+"_event.json").is_file() and complete(prefix.with_name(prefix.name+"_event.json")):continue
  guard([sys.executable,str(ROOT/"scripts/official_timesnet_adapter.py"),"--official-root",str(OFFICIAL_ROOT),"--official-commit",COMMIT,"--dataset","dataset/PEMS-BAY/PEMS-BAY.csv","--output",str(out),"--seq-len","96","--pred-len","192","--few-shot-fraction",".1","--train-fraction",".6","--val-fraction",".2","--batch-size","2","--epochs","5","--max-train-windows","64","--d-model","24","--d-ff","48","--e-layers","1","--top-k","3","--num-kernels","3","--dropout",".1","--seed",str(seed)],prefix)
 return 0
if __name__=="__main__":raise SystemExit(main())
