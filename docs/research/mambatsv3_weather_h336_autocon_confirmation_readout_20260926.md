# MambaTS v3 Weather H336 AutoCon confirmation readout (2026-09-26)

## Scope

This is the pre-registered current-source Weather H336 validation-only active
versus exact lambda-zero confirmation. It uses the contract in
`mambatsv3_weather_h336_autocon_confirmation_protocol_20260926.md`: same data
fingerprint, model, optimizer, `MeanStd`, seeds, and `--skip_test`; only
`AutoCon_lambda` changes.

## Results

| seed | lambda=0 best validation MSE | active best validation MSE | active change | active wins |
|---:|---:|---:|---:|:---:|
| 2021 | .615700 | .616300 | +.0974% | no |
| 2022 | .613100 | .612100 | -.1631% | yes |
| 2023 | .612300 | .614000 | +.2776% | no |
| mean | .613700 | .614133 | +.0706% | 1/3 |

All `6/6` jobs returned success and explicitly skipped test evaluation. Logs,
result configurations, and checkpoint loss logs are in
`artifacts/exploratory/mambatsv3_weather_h336_autocon_confirmation_20260926/`.

## Decision

The registered continuation gate failed: active has a higher mean validation
MSE than the exact null and wins only one of three seed pairs. Stop this
MambaTS v3 Weather branch at H336. Do not allocate the H336 baseline matrix,
perform post-hoc lambda tuning, or release a test evaluation. The H96
active/null signal remains a localized short-horizon observation and must not
be generalized to long-horizon or paper-level AutoCon claims.

