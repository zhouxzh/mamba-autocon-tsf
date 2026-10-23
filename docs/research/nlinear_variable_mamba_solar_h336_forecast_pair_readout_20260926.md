# Solar H336 forecast-lag-pair readout (2026-09-26)

## Result

This validation-only screen held the variable-axis Mamba architecture, the
prediction-connected AutoCon representation, raw train-only ACF prior,
selected-lag global relation loss, lag-pair sampler, ACF lags, optimization
budget, and `AutoCon_lambda=.0001` fixed. The only treatment difference was
the forecast-level signed ACF lag-pair loss with pre-registered weight `.01`.

| seed | pair term off | pair term on | on change | on better |
|---:|---:|---:|---:|:---:|
| 2031 | .154332 | .154062 | -.1751% | yes |
| 2032 | .153442 | .153351 | -.0593% | yes |
| 2033 | .153755 | .153426 | -.2138% | yes |
| mean | .153843 | .153613 | -.1495% | 3/3 |

All six jobs completed successfully at eight maximum epochs (one early-stop
checkpoint was selected at epoch seven in each arm). The enabled arm had
nonzero prediction-pair supervision in every seed: final-epoch pair loss
`.01539/.01542/.01475`, mean valid-pair fraction about `2.90%`, active-anchor
fraction about `89.7%`, and about `14.39` valid unordered pairs per batch.
The balanced sampler exposed the two resolved train-only lags `[168, 336]`.

No job evaluated test data: every training log contains
`test skipped by --skip_test`, and the test-result directories contain no
files.

## Interpretation

The result passes the pre-registered initial gate: lower mean validation MSE,
three of three exact seed-pair improvements, and valid nonzero telemetry. It
is a small effect (`.000230` absolute MSE) on one dataset and horizon, so it
does **not** establish a paper claim or justify test release. It only licenses
an independent dataset/horizon confirmation with the same frozen weight.

The training wall-clock mean was `49.37 s` with the pair term versus `42.04 s`
without it (`+17.4%` under this short GPU training protocol). The new loss is
not part of the model and is never called in evaluation/inference, so this is
not an inference-latency change. A later resource comparison must use the
same prediction-only timing protocol as the baseline table.

The execution manifest, source snapshots, configurations, checkpoints and
loss telemetry are retained under
[`artifacts/exploratory/nlinear_variable_mamba_solar_h336_forecast_pair_screen_20260926`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h336_forecast_pair_screen_20260926/).
