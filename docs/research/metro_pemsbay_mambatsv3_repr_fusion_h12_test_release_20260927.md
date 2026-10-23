# MambaTSv3 AutoCon Fusion H12 Held-out Results

## Scope

This release evaluates the frozen prediction-connected AutoCon fusion candidate on non-graph METR-LA and PEMS-BAY at horizon H12. It contains 6 validation-only selection jobs, 10 candidate held-out jobs, and 60 same-source baseline held-out jobs. All 76 jobs completed on `cuda:0` with `failed_jobs=0`; all three strict phase validators are valid.

The candidate is `MambaTSv3AutoCon` with `mamba_ts_repr_forecast_fusion=true`, `d_model=24`, `d_ff=48`, `seq_len=96`, `pred_len=12`, and `AutoCon_lambda=.001`. Baselines use the same data split, budget, optimizer, scheduler, batch size, and five held-out seeds (`2921--2925`).

## Held-out accuracy

The primary metric is streaming scaled MSE. Ratios are paired by dataset, model, horizon, and seed. The one-sided 95% upper ratio is compared with the registered non-inferiority threshold `1.02`.

| Dataset | Baseline | Candidate | Baseline | Change | Wins | Upper ratio | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| METR-LA | MambaTSv3 | 0.389206 | 0.398196 | -2.257% | 5/5 | 0.9814 | Pass |
| METR-LA | TimeMixer | 0.389206 | 0.417066 | -6.674% | 5/5 | 0.9432 | Pass |
| PEMS-BAY | MambaTSv3 | 0.310211 | 0.315941 | -1.816% | 5/5 | 0.9864 | Pass |
| PEMS-BAY | TimeMixer | 0.310211 | 0.371971 | -16.579% | 5/5 | 0.8506 | Pass |

All 12 candidate-versus-registered-baseline cells pass the `1.02` gate and are `5/5` paired wins. The six-baseline complete table is in [`paired_test_summary.csv`](../../artifacts/paper_final/metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927/readout/paired_test_summary.csv).

## Comparison with validation

Against the strongest same-source plain MambaTSv3, the previous validation confirmation reported improvements of `3.455%` on METR-LA and `1.554%` on PEMS-BAY. The held-out improvements are `2.257%` and `1.816%`, respectively. The direction is consistent across both datasets; the held-out test therefore supports replication of the validation signal rather than a selection-only result.

## Efficiency boundary

Held-out training elapsed time is not an inference benchmark. Candidate mean training elapsed time was `27.7 s` (METR-LA) and `102.1 s` (PEMS-BAY), versus `22.1 s` and `52.5 s` for plain MambaTSv3. A fair representation-path GPU benchmark measured approximately `+4.4%` and `+5.2%` overhead for fusion-on versus fusion-off. The evidence supports a small representation-path overhead, not an end-to-end training or universal speed claim.

## Paper-usable claim

The defensible claim is restricted to: *On non-graph METR-LA and PEMS-BAY H12 under a frozen five-seed held-out protocol, prediction-connected AutoCon fusion is non-inferior to the registered same-source baselines and improves over plain MambaTSv3 by 2.257% and 1.816% in streaming scaled MSE.* This does not establish long-horizon, graph-aware, causal, universal, or hardware-independent conclusions.
