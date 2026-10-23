# Weather Calibrated TimeMixer Boundary

Official TimeMixer is a material external baseline for the new Weather
target-prefix calibrated Channel-Patch protocol. The three paired validation
seeds use the same H192, 10% labels, chronological boundary, target-window
budget, and seed identifiers.

| Metric | Channel-Patch calibration | Official TimeMixer |
| --- | ---: | ---: |
| Mean validation MSE | 4.510126 | 4.508596 |
| Paired Channel-Patch wins | 1/3 | -- |
| Relative mean | +0.034% | -- |
| One-SE ratio upper | 1.00889 | -- |

TimeMixer is marginally lower on mean error and fails the candidate's
noninferiority gate. No further seed expansion is justified because the mean
and pair wins are both unfavorable. This validation-only boundary does not
open test data.

Machine-readable audit:
`artifacts/exploratory/official_timemixer_weather_channel_patch_calibration_20261007/audit.json`
