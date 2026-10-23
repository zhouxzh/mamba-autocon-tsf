# MambaACF relation-mask repair and channel-gate screen (PEMS03 H96, 2026-09-26)

## Why this screen was needed

The earlier gate preflight omitted `--mamba_acf_relation_lag_mask`. Under the
repository contract, `hard_selected` without that flag intentionally treats
all in-range pairs as positive targets. The resulting `100%` degenerate-positive
diagnostic is therefore an invalid representation-learning configuration for a
selected-peak contrastive claim. That earlier result remains archived as a
diagnostic, but is not used to judge the gate.

This screen enables the explicit selected-lag positive mask and lag-pair
sampler. All other settings are matched: PEMS03 H96, `d_model=32`, one scale,
three epochs, batch size 32, `AutoCon_lambda=0.001`, rank-4 raw channel gate,
and GPU `cuda:0`. Test evaluation was intentionally skipped.

## Paired validation result

| seed | AutoCon relation null | + rank-4 channel gate | relative gate delta |
|---:|---:|---:|---:|
| 2181 | 0.215840 | 0.212300 | -1.640% |
| 2027 | 0.227869 | 0.225475 | -1.051% |
| 31415 | 0.211166 | 0.212346 | +0.559% |
| mean | 0.218292 | 0.216707 | -0.711% |

The gate wins on 2/3 seeds, but the mean effect is below a 2% practical
threshold and the sample is too small for a confirmatory claim. The gate adds
5,017 parameters (6.39%). No test-set metric or universal speed claim is
licensed by this screen.

## AutoCon causal null

Keeping the corrected relation-mask geometry and backbone fixed, an additional
three-seed `AutoCon_lambda=0` null was run. The active relation-loss arm uses
`AutoCon_lambda=0.001` and is the same no-gate arm in the table above.

| seed | lambda-zero null | active relation loss | relative active delta |
|---:|---:|---:|---:|
| 2181 | 0.216679 | 0.215840 | -0.387% |
| 2027 | 0.228078 | 0.227869 | -0.092% |
| 31415 | 0.211499 | 0.211166 | -0.157% |
| mean | 0.218752 | 0.218292 | -0.212% |

The active objective improves all three seeds directionally, but the mean
effect is only `-0.212%`. It validates that the repaired objective is no
longer mechanically degenerate; it does not establish a practically material
forecasting contribution.

## Normalization and sampler confounding audit

The original plain MambaACF context run used `LastVal` normalization while the
AutoCon runs used `MeanStd`; those values are not comparable. A corrected
plain `MeanStd` three-seed reference produced validation MSE
`0.219573/0.214805/0.218623` (mean `0.217667`). The corrected active AutoCon
arm mean is `0.218292`, or `+0.287%` relative to this ordinary forecasting
reference.

As a single-seed sampler diagnostic, `AutoCon_lambda=0` with explicit random
batching gave `0.220413` at seed 2181, versus `0.219573` for plain MeanStd.
This does not prove a sampler effect, but it removes the earlier basis for
attributing the plain/AutoCon gap to lag-pair batching. It is enough to stop
post-hoc lambda tuning on PEMS03: `lambda=.01` and `.03` gave `0.215997` and
`0.218869` respectively at seed 2181, both worse than `.001` (`0.215840`).

Consequently, the selected-lag relation objective is mechanically valid but
not a forecasting improvement on this coordinate. A successor must be chosen
on a new preregistered screen with different data/horizons, not by further
PEMS03 H96 tuning.

## Relation-learning diagnostics

With the mask enabled, the final training diagnostics are consistent across
seeds:

- global contrastive anchors: approximately `73--74%`;
- degenerate positive anchors: approximately `19%`;
- candidate anchors: approximately `99.5%`;
- sign accuracy: approximately `32--34%` for this signed target;
- local target coverage: `1.58%`.

This is materially different from the no-mask run (`100%` degenerate positives,
`0%` contrastive anchors), so the corrected configuration does create a real
positive/negative relation objective. It does not yet demonstrate that the
objective improves forecasting accuracy.

## Decision

Promote the explicit relation-mask contract for future AutoCon MambaACF
experiments. Keep the channel gate in exploratory status only; do not tune it
or open the test set until it passes a preregistered multi-dataset, multi-seed
non-inferiority screen against the exact null and ordinary forecasting
baselines.

Artifacts:

- `artifacts/exploratory/mambaacf_relation_mask_pems03_20260926/null.log`
- `artifacts/exploratory/mambaacf_relation_mask_pems03_20260926/gate.log`
- `artifacts/exploratory/mambaacf_relation_mask_pems03_20260926/null_s2027.log`
- `artifacts/exploratory/mambaacf_relation_mask_pems03_20260926/gate_s2027.log`
- `artifacts/exploratory/mambaacf_relation_mask_pems03_20260926/null_s31415.log`
- `artifacts/exploratory/mambaacf_relation_mask_pems03_20260926/gate_s31415.log`
