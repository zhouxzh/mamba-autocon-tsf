# Channel-Patch Strength Boundary

The channel-patch objective is a prediction-scaled, train-only representation
loss. To test whether its canonical `0.05` fraction is merely underweighted,
this strict PEMS03 H192/25% target-excluded screen compares a same-seed `0.05`
arm with `0.10`. The Mamba-1 `24/8` backbone, ACF positives, source and
initialization exclusion, forecasting loss, frozen target core, and all target
data boundaries are fixed.

| Channel-patch fraction | Validation MSE | Change from 0.05 |
| ---: | ---: | ---: |
| 0.05 | 1.193764 | -- |
| 0.10 | 1.192291 | -0.123% |

Both objectives have channel-patch active-anchor fraction `1.0` in every
source epoch. Doubling the representation component changes its calibrated
weighted contribution from approximately `0.77` to `1.53`, so the null result
is not caused by an inactive loss. The `-0.123%` result misses the registered
`-0.25%` expansion threshold. No multi-seed, external-baseline, robustness, or
test extension is warranted.

Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_strength_probe_20261007/audit.json`
