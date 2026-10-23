# MambaTS v3 Weather H96 MeanStd baseline readout (2026-09-26)

## Scope and validity

This current-source screen uses the Weather multivariate split, H96, identical
three seeds (`2021/2022/2023`), eight-epoch maximum training budget, and an
explicit `AutoCon_wnorm=MeanStd` contract for every arm. The candidate active
and lambda-zero null logs are from the immediately preceding paired
replication. The six external baselines were retrained after the audit found
that an earlier command batch had omitted the explicit normalization setting.
Every result directory records `skip_test=true` and status
`trained_without_test`; no test outputs exist.

## Best validation MSE

| model | seed 2021 | seed 2022 | seed 2023 | mean | active relative change | active wins |
|---|---:|---:|---:|---:|---:|:---:|
| MambaTS v3 AutoCon, lambda=0 | .448000 | .443500 | .448600 | .446700 | -0.3582% | 3/3 |
| **MambaTS v3 AutoCon, lambda=.001** | **.446500** | **.440900** | **.447900** | **.445100** | - | - |
| DLinear | .544271 | .544360 | .544312 | .544314 | -18.2274% | 3/3 |
| NLinear | .544449 | .544941 | .544767 | .544719 | -18.2882% | 3/3 |
| PatchTST | .495106 | .494441 | .494334 | .494627 | -10.0130% | 3/3 |
| TimesNet | .452541 | .460593 | .466737 | .459957 | -3.2301% | 3/3 |
| iTransformer | .482081 | .484724 | .483615 | .483473 | -7.9370% | 3/3 |
| Mamba2 | .475612 | .475495 | .471776 | .474294 | -6.1553% | 3/3 |

Negative relative changes favor the active MambaTS v3 model. The only causal
AutoCon contrast in this table is active versus the exact lambda-zero null.
The external-model contrasts establish an H96 validation ranking under the
frozen small-budget contract; they do not prove generalization or replace
held-out evaluation.

## Provenance

- Active/null logs and configurations:
  `artifacts/exploratory/mambatsv3_weather_autocon_replication_20260926/`.
- Fair external baseline logs and configurations:
  `artifacts/exploratory/mambatsv3_weather_strong_baseline_matrix_meanstd_20260926/`.
- Weather SHA-256:
  `773b97c4495ed044b390241cea9a9c39b79c80b2436dbcc0fbd32e289c86abf5`.

## Decision

Advance only to the registered H336 active/null confirmation. Do not release a
test evaluation, claim a universal speed advantage, or generalize this single
H96 dataset coordinate into a paper-level conclusion.

