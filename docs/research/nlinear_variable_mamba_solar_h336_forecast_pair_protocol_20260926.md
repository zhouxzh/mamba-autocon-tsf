# Solar H336 forecast-lag-pair protocol (2026-09-26)

## Question

The preceding exact Solar H336 replication showed that the existing
representation-level AutoCon relation loss was not a stable prediction
improvement. This protocol tests a distinct, prediction-connected mechanism:
for windows paired at selected train-only ACF lags, match the signed
correlation of their forecast trajectories to the signed train ACF target.

## Fixed Contract

- Solar AL TXT, 137 variables, SHA-256
  `230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2`;
- `L=96`, label `48`, H336, `LastVal`, batch 32, eight epochs maximum,
  patience 3, learning rate `.001`, fresh seeds `2031/2032/2033`;
- `NLinearVariableMambaAutoCon`: one variable-axis Mamba-1 scan, `d_model=64`,
  `d_ff=128`, two layers, `d_state=8`, `d_conv=2`, expand 1, residual
  initialization `.1`, and prediction-connected representation fusion;
- both arms retain raw train-only ACF, selected-lag masked global AutoCon
  relation loss, balanced lag-pair batching, minimum lag 96, and
  `AutoCon_lambda=.0001` with two warm-up epochs;
- the only difference is the training-only forecast-pair objective:
  disabled with weight `0` versus enabled with a pre-registered weight `.01`.

The treatment has no additional parameter or inference path. Each job uses
`--skip_test --no_save`; test data cannot be used for this screen.

## Gate

Advance only if enabled has lower mean best validation MSE, wins at least two
of three exact seed pairs, and records nonzero pair loss/count/active-anchor
telemetry in all seeds. A failure stops the mechanism; it does not license a
weight search or a test release.

The executable registration is
[`nlinear_variable_mamba_solar_h336_forecast_pair_screen_20260926.json`](nlinear_variable_mamba_solar_h336_forecast_pair_screen_20260926.json).
