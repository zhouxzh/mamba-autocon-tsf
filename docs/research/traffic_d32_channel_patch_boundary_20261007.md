# Traffic d32 Channel-Patch Boundary

The prior d32/24 Electricity-to-Traffic transfer bridge found that its
forecast-level ACF relation schedule was unfavorable relative to its exact
forecast-prediction null. That result did not test the current full
channel-patch representation objective. This validation-only screen keeps the
same single Mamba-1 scan, `d_model=32`, `24/24` patching, Electricity source,
Traffic 25% chronological training-prefix calibration, forecast InfoNCE, and
frozen target core. The candidate alone adds train-only full trajectory
Channel-Patch InfoNCE at fraction `.05` for both source and calibration.

| Arm | Validation MSE | Difference |
| --- | ---: | ---: |
| Forecast calibration | 0.819623 | -- |
| Forecast + Channel-Patch calibration | 0.822247 | +0.320% |

The Channel-Patch objective was active rather than bypassed: its active-anchor
fraction was `0.906`, `0.917`, and `0.875` over the three pretraining epochs,
with nonzero prediction-scaled relation loss for Electricity and Traffic.
All four guarded stages completed without intervention; the highest observed
temperature was 49C. No test rows or external baseline results were read.

The pre-registered expansion rule required a `-0.25%` validation improvement.
This screen fails that condition and is not expanded. Together with the
existing forecast-level negative bridge, it shows that the current
Channel-Patch representation does not repair the high-dimensional
Electricity-to-Traffic transfer limitation under this frozen-core protocol.

Machine-readable evidence:
`artifacts/exploratory/traffic_d32_channel_patch_probe_20261007/audit.json`.
