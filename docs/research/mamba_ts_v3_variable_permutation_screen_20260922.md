# MambaTS v3 variable permutation screen (2026-09-22)

## Purpose

This screen tests the MambaTS-style train-time variable permutation as a possible
regularizer for the prediction-connected AutoCon representation route. The
permutation is sampled per training batch, applied before the shared variable
scan, and inverted before the public output. Evaluation remains deterministic.

Plan: [`mamba_ts_v3_variable_permutation_screen_20260922.json`](./mamba_ts_v3_variable_permutation_screen_20260922.json)

Readout: [`readout.md`](../../artifacts/exploratory/mamba_ts_v3_variable_permutation_screen_20260922/readout/readout.md)

## Protocol

- MambaTS v3, H96, four datasets: ETTh1, ETTm1, Electricity, Traffic.
- Three seeds: 2021, 2022, 2023; validation-only; test metrics were not read.
- Three arms, 36/36 completed jobs:
  - global representation fusion, no permutation, `AutoCon_lambda=0.003`;
  - permutation + representation fusion, `AutoCon_lambda=0`;
  - permutation + representation fusion, `AutoCon_lambda=0.003`.
- The primary comparison is permutation active versus permutation null. The
  global-fusion arm is an architecture contrast, not a matched ablation.

## Results

| Dataset | Permuted active vs permuted null | Wins / 3 |
| --- | ---: | ---: |
| ETTh1 | `+0.220%` | 1 |
| ETTm1 | `-0.182%` | 2 |
| Electricity | `-1.111%` | 3 |
| Traffic | `-0.037%` | 2 |

The sign convention is percentage change in validation MSE, so positive values
are worse. The result is not a uniform improvement: the high-dimensional
Electricity case degrades on all three seeds, while the other datasets are close
to noise-level changes. Against the no-permutation global-fusion arm, the
permuted active arm changes by `+0.138%`, `+0.173%`, `-0.260%`, and `+0.046%`
for ETTh1, ETTm1, Electricity, and Traffic respectively; this is an
architecture contrast and should not be used as an ablation claim.

## Resource check

The prediction-plus-representation path was measured on RTX 5090 D with
synthetic device-resident input, B=8, C=862, H=96, 20 warm-up and 80 timed
forwards. The permutation flag is training-only, so it is inactive in this
evaluation path. The record reports 136,700 parameters, median latency
`0.6933 ms`, `11,530` sequences/s, and a peak allocated increment of
`153.5 MB`.

Raw record: [`model_resources.json`](../../artifacts/exploratory/mamba_ts_v3_variable_permutation_resource_20260922/model_resources.json)

## Decision

Do not promote variable permutation to the paper mainline. It is implemented,
contract-tested, and reproducible, but this screen provides no cross-dataset
accuracy gain and no independent inference-speed benefit. It may remain as a
negative ablation or a future robustness study after a stronger prediction
objective is established.

