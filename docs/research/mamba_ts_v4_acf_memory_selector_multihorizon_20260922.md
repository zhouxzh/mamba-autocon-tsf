# MambaTS v4 ACF Memory Selector Long-Horizon Screen (2026-09-22)

## Protocol

This is the frozen H336/H720 continuation of the H96 selector screen. The
only treatment difference is `mamba_ts_acf_memory_selector`; both arms use the
same MambaTS v4 AutoCon backbone, raw train-only ACF prior, lags `[24, 48]`,
`AutoCon_lambda=0.001`, `MeanStd`, optimizer, short budget, and three seeds.
All 48 jobs are validation-only. Both phase validators report 24 unique jobs,
valid provenance, matching artifact hashes, and no test artifacts.

Plan: [`mamba_ts_v4_acf_memory_selector_multihorizon_20260922.json`](./mamba_ts_v4_acf_memory_selector_multihorizon_20260922.json)

Readout: [`readout.md`](../../artifacts/exploratory/mamba_ts_v4_acf_memory_selector_multihorizon_20260922/readout/readout.md)

## Results

| Dataset | H336 | H720 |
| --- | ---: | ---: |
| ETTh1 | `-0.266%` (2/3) | `+0.367%` (1/3) |
| ETTm1 | `-0.641%` (3/3) | `-0.734%` (2/3) |
| Electricity | `-0.418%` (2/3) | `-0.829%` (3/3) |
| Traffic | `+0.039%` (1/3) | `-0.046%` (2/3) |

Values are selector versus exact static-memory control; negative changes favor
the selector. It preserves useful signal on ETTm1 and Electricity, but fails
the cross-dataset/cross-horizon gate because ETTh1/H720 and Traffic/H336
degrade. The Traffic/H720 change is also too small to support a robust claim.

## Decision

Do not run same-snapshot strong baselines, test evaluation, or further tuning
for this exact selector. The H96 gain does not generalize uniformly to long
horizons. The current evidence supports only a conditional research finding:
an AutoCon representation can selectively improve train-prior ACF memory in
some dataset/horizon regimes, not a general-purpose forecasting architecture.

