# Weather Calibrated 25% Boundary

The Weather target-prefix calibrated Channel-Patch protocol passes its
registered 10% label replication but must also be assessed at 25% labels.
This fresh three-seed H192 validation replication keeps the same source pool,
Mamba-1 `24/8` one-scan core, source and target calibration objectives, and
full Channel-Patch trajectory loss. The Weather calibration domain is only the
earliest 15% of the raw timeline (`60%` chronological train region times `25%`
few-shot fraction); validation and test remain closed.

| Arm | Mean validation MSE | Channel-Patch difference | Wins |
| --- | ---: | ---: | ---: |
| Forecast calibration | 2.489742 | -- | -- |
| Channel-Patch calibration | 2.488316 | -0.057% | 2/3 |

The one-SE ratio upper is `0.99998`, so noninferiority holds, but the
pre-registered promotion condition requires all three wins. It therefore does
not authorize external-baseline, robustness, or test expansion. The calibrated
Weather mechanism is label-budget dependent and cannot be described as a
uniform 10%/25% success.

Machine-readable audit:
`artifacts/exploratory/weather_channel_patch_calibration_25pct_replication_20261007/audit.json`
