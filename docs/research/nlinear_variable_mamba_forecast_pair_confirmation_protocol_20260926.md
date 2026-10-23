# Forecast-Lag-Pair Independent Confirmation (2026-09-26)

Solar H336 passed the initial three-seed screen with forecast-pair weight
`.01`. This follow-up freezes that weight and tests Solar H720 as an
independent longer-horizon validation-only coordinate.

Every off/on pair keeps the variable-axis Mamba, prediction-connected AutoCon
representation, train-only raw ACF prior, selected-lag global relation loss,
balanced lag-pair batching, and `AutoCon_lambda=.0001` fixed. Solar retains
the fixed lag bank. Fresh seeds are `2041/2042/2043`; all jobs use
`--skip_test --no_save`.

Each coordinate passes only when pair-on improves mean validation MSE, wins at
least two of three exact seed pairs, and records nonzero pair telemetry in all
enabled jobs. A failing coordinate cannot proceed to baseline expansion or
test evaluation.

The executable registration is
[`nlinear_variable_mamba_forecast_pair_confirmation_screen_20260926.json`](nlinear_variable_mamba_forecast_pair_confirmation_screen_20260926.json).
