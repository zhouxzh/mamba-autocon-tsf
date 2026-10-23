#!/usr/bin/env python3
"""Validation-only task-aligned ACF coefficient gate for residual Mamba transfer."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_forecast_target_acf_probe_20261008"
SEED=6120
ARMS=("raw_signal_acf","forecast_target_acf")
INITIAL=(
    ROOT/"artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems03_MambaTSv3AutoCon_h12_s2131_l0p001_ca786fbe2493c_MambaTSv3AutoCon_pems03_ftM_sl96_ll48_pl12_dm2_h00dcc565bb3086f9/checkpoint.pth",
    ROOT/"artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems04_MambaTSv3AutoCon_h12_s2131_l0p001_c22b7150b652d_MambaTSv3AutoCon_pems04_ftM_sl96_ll48_pl12_dm2_h179e4641a67e8f06/checkpoint.pth",
)

def complete(path):
    return path.is_file() and json.loads(path.read_text()).get("status")=="completed" and json.loads(path.read_text()).get("trigger") is None

def guard(command,prefix,stage):
    subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)

def main():
    if not all(path.is_file() for path in INITIAL): raise FileNotFoundError("frozen initialization checkpoint missing")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.1,"seed":SEED,"arms":list(ARMS),"question":"Does a train-only future-residual ACF table improve residual-target Mamba transfer over raw-signal ACF while holding peak lags, source windows, architecture, and auxiliary weight fixed?","fixed_contract":{"backbone":"channel-agnostic MambaTSv3 one scan","prediction_target":"last_value_residual","source_relation_view":"forecast_affine","source_relation_fraction":.1,"source_domains":["Electricity","Weather","METR-LA"],"target_primary":"DLinear","exact_prediction_head_transfer":True},"only_difference":"forecast_target_acf replaces the raw-signal coefficient at each fixed raw ACF peak with the train-prefix empirical correlation of centered H96 last-value residual future trajectories","target_calibration_boundary":"target validation and test remain unread; target adaptation uses only the first 10% of its 60% chronological train partition","screen_rule":"expand only if task-aligned ACF is at least .25 percent below raw-signal ACF on validation MSE"}
    (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    for arm in ARMS:
        prefix=OUTPUT/f"s{SEED}_{arm}"
        pre=["env","AUTOCON_CPU_THREAD_LIMIT=4","OMP_NUM_THREADS=4","MKL_NUM_THREADS=4","OPENBLAS_NUM_THREADS=4","NUMEXPR_THREADS=4",sys.executable,"scripts/pretrain_mambatsv3_core_autocon.py","--source","Electricity:dataset/electricity/electricity.csv:csv","--source","Weather:dataset/weather/weather.csv:csv","--source","METR-LA:dataset/METR-LA/METR-LA.csv:csv"]
        for checkpoint in INITIAL: pre.extend(("--initial-checkpoint",str(checkpoint)))
        pre.extend(("--output",str(prefix.with_name(prefix.name+"_pretrain.pt")),"--batch-size","4","--epochs","3","--max-windows","64","--source-pred-len","96","--prediction-target","last_value_residual","--pretrain-task","next_patch","--reconstruction-weight","0","--relation-view","forecast_affine","--mamba-lambda",".1","--relation-loss-fraction",".1","--seed",str(SEED)))
        if arm=="forecast_target_acf": pre.append("--forecast-target-acf")
        ft=["env","AUTOCON_CPU_THREAD_LIMIT=4","OMP_NUM_THREADS=4","MKL_NUM_THREADS=4","OPENBLAS_NUM_THREADS=4","NUMEXPR_THREADS=4",sys.executable,"scripts/finetune_mambatsv3_dlinear_hybrid.py","--linear-model","DLinear","--mamba-correction-output","--dataset","dataset/traffic/traffic.csv","--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")),"--transfer-prediction-head","--prediction-head-transfer-mode","exact","--output",str(prefix.with_name(prefix.name+"_hybrid.pt")),"--few-shot-fraction",".1","--batch-size","2","--epochs","5","--max-train-windows","64","--freeze-mamba-epochs","1","--channel-agnostic-scan","--seed",str(SEED)]
        if "--evaluate-test" in pre+ft: raise ValueError("forecast-target ACF probe cannot read test")
        if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))): guard(pre,prefix,"pretrain")
        if not(prefix.with_name(prefix.name+"_hybrid.pt").is_file() and complete(prefix.with_name(prefix.name+"_hybrid_event.json"))): guard(ft,prefix,"hybrid")

if __name__=="__main__": main()
