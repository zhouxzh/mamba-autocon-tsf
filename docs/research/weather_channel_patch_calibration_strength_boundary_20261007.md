# Weather Strong Calibration Boundary

The Weather target-prefix calibration control improves substantially at a
calibration relation fraction of `0.1`, but the Channel-Patch increment is
below its materiality threshold. This registered screen tests a stronger
target-prefix forecast-relation fraction of `0.3`, while retaining the same
Mamba-1 `24/8` one-scan model, sources, earliest Weather 10%-label training
prefix, frozen target core, and Channel-Patch fraction `0.05`.

| Arm | Validation MSE |
| --- | ---: |
| Forecast calibration, relation 0.3 | 4.464125 |
| Channel-Patch calibration, relation 0.3 | 4.462932 |
| Prior Channel-Patch calibration, relation 0.1 | 4.413717 |

The `.3` Channel-Patch arm is only `-0.027%` below its forecast-calibration
control and `+1.115%` worse than the completed `.1` calibration candidate.
It fails the registered expansion rule. No multi-seed, robustness, external
baseline, or test extension is justified.

Machine-readable audit:
`artifacts/exploratory/weather_channel_patch_calibration_strength_probe_20261007/audit.json`
