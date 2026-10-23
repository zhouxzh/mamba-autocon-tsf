# Variable-axis Mamba Solar H336 AutoCon replication readout (2026-09-26)

## Results

Current-source, validation-only, exact paired H336 results for the
prediction-connected `NLinearVariableMambaAutoCon` are:

| seed | lambda=0 best validation MSE | active best validation MSE | active change | active wins |
|---:|---:|---:|---:|:---:|
| 2022 | .153900 | .154100 | +.1300% | no |
| 2023 | .154200 | .154500 | +.1946% | no |
| 2024 | .154600 | .154400 | -.1294% | yes |
| mean | .154233 | .154333 | +.0648% | 1/3 |

All six jobs completed successfully with `--skip_test --no_save`. Relation
telemetry was non-degenerate: the finished null run reported about `50.36%`
contrastive anchors and `39.32%` degenerate positives. Thus the negative
forecast result is not attributable to the previously observed all-positive
relation-target failure.

## Decision

The active arm fails the pre-registered continuation rule. Stop current
Solar H336 expansion of this variable-axis Mamba branch: do not treat its
existing candidate-versus-baseline advantage as evidence that AutoCon improves
accuracy, and do not perform post-hoc lambda tuning or test release. The
compact Mamba forecast path can remain an architectural baseline, but it is
not a paper candidate satisfying the required contrastive-learning
contribution.

Logs and current-source configurations are under
`artifacts/exploratory/nlinear_variable_mamba_solar_h336_autocon_replication_20260926/`.

