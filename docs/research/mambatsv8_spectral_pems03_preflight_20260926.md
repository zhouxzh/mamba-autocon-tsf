# MambaTS v8 spectral PEMS03 preflight

Date: 2026-09-26

## Question

Can a causal, input-window Fourier residual close the PEMS periodic-model gap
while preserving a direct AutoCon representation-to-forecast path?

## Architecture

`MambaTSv8AutoCon` retains the shared MambaTS v3 patch scan
(`patch_len=stride=24`, `d_model=32`, `d_ff=64`, Mamba-1 `d_state=8`,
`d_conv=2`, `expand=1`). It computes an `rfft` only over each observed
normalized 96-step window, keeps the four largest non-DC bins per channel,
and evaluates their Fourier reconstruction only at future steps 96--191.
There is no fitted validation/test statistic or future input in this path.

The residual is multiplied by a zero-initialized scalar ReZero parameter. In
the AutoCon arm, a horizon gate and a shared channel gate both read the same
time-indexed representation used by the AutoCon objective. Consequently,
the contrastive representation has a direct prediction-facing route, rather
than affecting the forecast only indirectly through the encoder.

The public MambaTS implementation identifies variable-aware temporal scan and
variable-permutation training as its central structural ideas; the local v8
keeps that MambaTS-inspired shared scan as its base and tests only the new
periodic residual. It does not claim equivalence to the external code.
[Official MambaTS repository](https://github.com/XiudingCai/MambaTS-pytorch)

## Matched preflight

Both arms used PEMS03, M features, H96, seed `2181`, three epochs, batch 32,
MeanStd normalization, AutoCon lambda `0.001`, `skip_test=true`, and the same
optimizer schedule. This is a validation-only, one-seed, low-budget rejection
screen, not a confirmatory comparison.

| Arm | Best pure forecast validation MSE | Relative to v3 |
| --- | ---: | ---: |
| MambaTSv3AutoCon | 0.241888 | reference |
| MambaTSv8AutoCon | 0.423632 | +75.136% |

Artifacts are under
[`mambatsv8_pilot_20260926`](../../artifacts/exploratory/mambatsv8_pilot_20260926/).
No test split was read or evaluated.

## Decision

Stop v8 before multi-dataset, multi-seed, resource, or test expansion. The
large one-seed regression fails the purpose of a low-cost preflight, so
further tuning would be outcome-driven. This result does not alter the v3
historical evidence and does not support any periodic-residual or AutoCon
accuracy claim.

## Implementation checks

The implementation is registered in the model runner, experiment entry
points, plan runner, and resource benchmark CLI. Contract, causality-shape,
and resource-registration checks were run as part of the complete suite:
`702 passed, 44 warnings`.
