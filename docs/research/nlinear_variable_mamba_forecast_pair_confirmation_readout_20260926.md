# Forecast-lag-pair independent confirmation readout (2026-09-26)

## Solar H720

The forecast-pair term was frozen at weight `.01` after the Solar H336
initial screen. The variable-axis Mamba, prediction-connected AutoCon
representation, train-only raw ACF, relation objective, lag-pair sampler,
optimizer, and budget were unchanged between arms.

| seed | pair term off | pair term on | on change | on better |
|---:|---:|---:|---:|:---:|
| 2041 | .153766 | .153975 | +.1359% | no |
| 2042 | .154314 | .154195 | -.0775% | yes |
| 2043 | .153624 | .154007 | +.2492% | no |
| mean | .153902 | .154059 | +.1022% | 1/3 |

All six jobs completed successfully with eight epochs and `returncode=0`.
Every enabled seed had valid prediction-pair supervision: final-epoch pair
loss `.01392/.01381/.01352`, valid-pair fraction about `2.90%`, active-anchor
fraction about `89.6%`, and about `14.38` valid unordered pairs per batch.
The resolved train-only relation lags were `[168, 336]`.

Every log contains `test skipped by --skip_test`; no files were written under
the test-result directories.

## Decision

The independent longer-horizon confirmation fails the frozen continuation
rule: the treatment is worse on average and wins only one of three paired
seeds. The initial H336 improvement (`-.1495%`) is therefore a local
validation signal, not evidence of a transferable mechanism. Stop forecast
lag-pair expansion, weight tuning, additional datasets, and test release.

The treatment adds no inference parameters or evaluation path, but its
training wall-clock mean increased from `48.43 s` to `54.39 s` on this H720
screen (`+12.3%`). Thus it also does not support a speed claim.

Artifacts, manifests, source snapshots and telemetry are under
[`forecast-pair confirmation artifacts`](../../artifacts/exploratory/nlinear_variable_mamba_forecast_pair_confirmation_20260926/).
