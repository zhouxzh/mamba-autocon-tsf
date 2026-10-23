# Variable-axis Mamba Solar H336 AutoCon replication protocol (2026-09-26)

## Question

The compact variable-axis Mamba has a strong long-horizon Solar validation
signal and favorable prediction-only latency versus iTransformer/PatchTST at
H336. Its existing long-horizon matrix did not include an exact relation-loss
null, so it cannot support the required AutoCon contribution. This current
source replication isolates that objective while preserving the
prediction-connected representation fusion.

## Fixed architecture and training contract

- Solar AL TXT, `137` variables, SHA-256
  `230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2`;
- `L=96`, label `48`, H336, `LastVal`, batch 32; eight epochs maximum,
  patience 3, learning rate `.001`, seeds `2022/2023/2024`;
- `NLinearVariableMambaAutoCon`: `d_model=64`, `d_ff=128`, two layers,
  one variable-axis Mamba-1 scan (`d_state=8`, `d_conv=2`, expand 1),
  residual initialization `.1`, prediction-connected representation fusion;
- train-only raw ACF prior, selected-lag masked global relation loss,
  lag-pair balanced sampler, minimum lag 96, encoder representation,
  AutoCon warm-up 2; all settings are identical between arms;
- exact arms: `AutoCon_lambda=0` and `.0001`.

Every job runs with `--skip_test --no_save`; neither validation selection nor
training reads test metrics.

## Decision rule

Only expand this current-source Solar candidate to new baseline/resource or
test-isolated work if the active arm has lower mean validation MSE than null,
wins at least two of three seed pairs, and shows valid non-degenerate relation
telemetry. Otherwise stop the variable-axis Mamba Solar branch; the retained
candidate-versus-baseline result cannot be attributed to AutoCon.

