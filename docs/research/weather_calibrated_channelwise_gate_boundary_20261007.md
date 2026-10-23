# Weather Calibrated Channelwise Gate Boundary

The target-channel residual gate is a zero-initialized target-head refinement
over an unchanged frozen calibrated Channel-Patch core. It introduces no
additional Mamba scan and no source-domain channel identity, but permits the
existing Mamba correction to vary across Weather's 21 output channels.

On registered Weather H192/10% seed `6073`, it reaches `4.507823` validation
MSE, versus `4.505757` for scalar Channel-Patch and `4.480501` for same-seed
official TimeMixer. The learned gate is active (maximum absolute scale
`0.07769`), so this is a negative capacity result rather than a dormant zero-
gate result. It misses both scalar and TimeMixer screen gates; no replication
or external expansion is warranted.

Machine-readable audit:
`artifacts/exploratory/weather_calibrated_channelwise_gate_probe_20261007/audit.json`
