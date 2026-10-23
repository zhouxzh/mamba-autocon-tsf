#!/usr/bin/env python3
"""Validation-only official PatchTST comparator for the d32/24 Electricity bridge."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OFFICIAL_ROOT=Path("/tmp/patchtst-adapter-oaFMCU/PatchTST-204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised")
COMMIT="204c21efe0b39603ad6e2ca640ef5896646ab1a9";OUTPUT=ROOT/"artifacts/exploratory/official_patchtst_electricity_d32_stride24_comparator_20261006";SEEDS=(5997,5998,5999)
def guard(c,p):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+"_telemetry.csv")),"--event-log",str(p.with_name(p.name+"_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seeds":list(SEEDS),"official_commit":COMMIT,"geometry":{"d_model":32,"d_ff":64,"patch_len":24,"patch_stride":24},"comparison":"same chronological split, target windows, epochs and seeds as d32/24 Mamba bridge"},indent=2)+"\n",encoding="utf-8")
 if not OFFICIAL_ROOT.is_dir():raise ValueError("official root missing")
 for seed in SEEDS:
  prefix=OUTPUT/f"electricity_h96_s{seed}_official_patchtst";output=prefix.with_suffix(".pt")
  if args.resume and output.is_file() and prefix.with_suffix(".json").is_file():continue
  guard([sys.executable,str(ROOT/"scripts/official_patchtst_adapter.py"),"--official-root",str(OFFICIAL_ROOT),"--official-commit",COMMIT,"--dataset","dataset/electricity/electricity.csv","--output",str(output),"--seq-len","96","--pred-len","96","--few-shot-fraction",".25","--train-fraction",".6","--val-fraction",".2","--batch-size","4","--epochs","5","--max-train-windows","64","--d-model","32","--d-ff","64","--n-heads","4","--e-layers","1","--patch-len","24","--patch-stride","24","--dropout",".1","--fc-dropout",".05","--head-dropout","0","--revin","1","--affine","0","--subtract-last","0","--individual","0","--seed",str(seed)],prefix)
 return 0
if __name__=="__main__":raise SystemExit(main())
