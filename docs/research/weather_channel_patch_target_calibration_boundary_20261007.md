# Weather Target-Prefix Calibration Boundary

Weather has lower train-prefix ACF transfer compatibility than the traffic
targets. This registered H192/10% screen adds only the earliest 6% of its raw
timeline (`60%` chronological train region times `10%` few-shot fraction) as a
target-calibration source during pretraining. Its validation and test segments
remain closed. Both arms use identical source data, next-patch prediction,
forecast InfoNCE, calibration fraction, and frozen target core; Channel-Patch
alone adds full patch-trajectory InfoNCE.

| Arm | Validation MSE | Change |
| --- | ---: | ---: |
| Uncalibrated forecast reference | 4.547938 | -- |
| Forecast with target-prefix calibration | 4.421606 | -2.778% |
| Channel-Patch with target-prefix calibration | 4.413717 | -2.951% vs uncalibrated; -0.178% vs calibrated forecast |

The calibration control confirms a real domain-adaptation effect. The extra
Channel-Patch gain misses the pre-registered `-0.25%` expansion threshold, so
this `.1` calibration-relation setting is not expanded. The target-prefix role,
3,161 rows, and active Channel-Patch anchors are recorded in the artifact.

Machine-readable audit:
`artifacts/exploratory/weather_channel_patch_target_calibration_probe_20261007/audit.json`

A registered stronger target-prefix relation schedule (`0.3`) is also closed.
It yields Channel-Patch MSE `4.462932`, only `-0.027%` versus its same-seed
forecast calibration and `+1.115%` worse than the completed `.1` calibration
Channel-Patch screen. It does not meet the expansion rule.

A full-pool train-prefix source-selection check confirms that the existing
METR-LA/PEMS-BAY/Electricity source pool remains the top three candidates, so
no post-hoc source replacement is justified.
