# Official-style MambaTS VPT Local-search Preflight

## What Changed

The repository previously implemented MambaTS v11 variable permutation training with a greedy evaluation route. This preflight adds the public MambaTS route-solver family through `python-tsp`: a train-derived directed transition-cost matrix is decoded with local search after the checkpoint is restored. Random per-sample variable paths, Mamba scan, representation fusion, AutoCon loss, data, budget, and seed remain fixed.

This is an adaptation of the official [MambaTS implementation](https://github.com/XiudingCai/MambaTS-pytorch), not a bitwise reproduction of its vendored mixer.

## Integrity

All `42/42` validation-only GPU jobs completed. The greedy VPT, local-search VPT, v3 fusion reference, and external baseline phases all have `valid=true` phase validators and one source snapshot. Every v11 run has positive VPT transition count and near-complete transition coverage. All runs use `skip_test=true`.

## Result

| Dataset | Local search vs greedy | Local-search wins | External decision |
| --- | ---: | ---: | --- |
| ETTh1 H96 | -0.051% | 1/3 | Fail: +4.389% vs PatchTST |
| Weather H96 | +0.233% | 0/3 | External baselines pass, but mechanism fails |

ETTh1 local-search MSE is `0.716456`, compared with PatchTST `0.686335`, TimeMixer `0.700938`, iTransformer `0.684409`, and Mamba2 `0.738182`. Weather local-search MSE is `0.448503`, below PatchTST `0.493253`, TimeMixer `0.458740`, iTransformer `0.481306`, and Mamba2 `0.472295`, but it is worse than v11 greedy `0.447457`.

## Decision

Stop this branch. The local-search solver does not improve its exact greedy VPT control in either dataset under the pre-registered paired rule. It cannot be promoted to a test release or claimed as an AutoCon/MambaTS architecture improvement. The full strict table is [`readout.md`](../../artifacts/exploratory/mambatsv11_official_vpt_ls_h96_preflight_20260927/readout/readout.md).
