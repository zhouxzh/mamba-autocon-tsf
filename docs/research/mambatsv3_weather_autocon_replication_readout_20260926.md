# MambaTS v3 Weather AutoCon replication readout (2026-09-26)

## Scope

This is a current-source, validation-only replication of the earlier Weather
H96 MambaTS v3 pilot. It compares an exact AutoCon null (`lambda=0`) with the
active objective (`lambda=.001`) under identical data, optimizer, model and
random-seed contracts. All six jobs used `--skip_test --no_save`; no test
metric was used.

## Results

| seed | lambda=0 best validation MSE | active best validation MSE | active change | active wins |
|---:|---:|---:|---:|:---:|
| 2021 | 0.448000 | 0.446500 | -0.3348% | yes |
| 2022 | 0.443500 | 0.440900 | -0.5862% | yes |
| 2023 | 0.448600 | 0.447900 | -0.1560% | yes |
| mean | 0.446700 | 0.445100 | -0.3582% | 3/3 |

The run logs and checkpoint loss logs are under
`artifacts/exploratory/mambatsv3_weather_autocon_replication_20260926/`.
The Weather file hash is
`773b97c4495ed044b390241cea9a9c39b79c80b2436dbcc0fbd32e289c86abf5`.

## Decision

The pre-registered continuation rule was met: active had lower mean validation
MSE, won all three paired seeds, and all jobs completed without numerical or
contract failures. The candidate is therefore allowed into a same-snapshot
Weather strong-baseline matrix. This is a screening gate, not evidence of a
general AutoCon improvement or a test-set result. The gain is small, so the
baseline matrix must retain the exact null arm and report paired variability.

## Reproduction contract

- Weather multivariate features, H96, `MeanStd`, batch 32;
- MambaTS v3, `d_model=32`, one layer, patch `16/8`, causal embedding, one
  scale, Mamba state/convolution/expand `16/4/2`;
- eight epochs maximum, patience 3, cosine learning-rate schedule;
- seeds `2021/2022/2023`, `--skip_test`, GPU `cuda:0`.

