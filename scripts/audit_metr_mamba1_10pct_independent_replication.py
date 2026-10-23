#!/usr/bin/env python3
"""Read-only audit of independent METR-LA 10% Mamba-1/PatchTST replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];MAMBA=ROOT/'artifacts/exploratory/mambatsv3_metr_mamba1_10pct_independent_replication_20261006';PATCH=ROOT/'artifacts/exploratory/official_patchtst_metr_mamba1_10pct_independent_replication_20261006';SEEDS=(6007,6008,6009,6010,6011)
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def summary(left,right):
 ratio=left/right
 return {'left_mean_mse':float(left.mean()),'right_mean_mse':float(right.mean()),'percent':float((left.mean()/right.mean()-1)*100),'wins':int((left<right).sum()),'ratio_upper_1se':float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),'noninferior_at_one_se':bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 rows=[]
 for seed in SEEDS:
  a,n,p=load(MAMBA/f's{seed}_active_frozen_core.json'),load(MAMBA/f's{seed}_null_frozen_core.json'),load(PATCH/f'metr_la_h192_s{seed}_official_patchtst.json')
  if any(x.get('test_metrics_produced') for x in (a,n,p)):raise ValueError('test boundary')
  cfg=a['config'];pc=p['config']
  if (cfg.get('mamba_ts_mixer_type'),int(cfg['patch_len']),int(cfg['patch_stride']),float(cfg['few_shot_fraction']),bool(cfg['freeze_pretrained_core'])) != ('mamba1',24,8,.1,True):raise ValueError(f'candidate: {seed}')
  if (int(pc['patch_len']),int(pc['patch_stride']),float(pc['few_shot_fraction'])) != (24,8,.1):raise ValueError(f'PatchTST: {seed}')
  rows.append({'seed':seed,'active':float(a['best_validation_mse']),'null':float(n['best_validation_mse']),'official_patchtst':float(p['best_validation_mse'])})
 active=np.asarray([r['active'] for r in rows]);null=np.asarray([r['null'] for r in rows]);patch=np.asarray([r['official_patchtst'] for r in rows])
 result={'validation_only':True,'test_metrics_produced':False,'seeds':list(SEEDS),'rows':rows,'active_vs_null':summary(active,null),'active_vs_official_patchtst':summary(active,patch),'submission_ready':False}
 result['external_stress_eligible']=bool(result['active_vs_null']['noninferior_at_one_se'] and result['active_vs_official_patchtst']['noninferior_at_one_se'])
 (MAMBA/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
