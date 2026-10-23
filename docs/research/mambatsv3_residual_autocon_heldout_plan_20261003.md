# Residual-Target AutoCon Held-Out Release Plan

## Frozen Candidate

The candidate is the few-shot-only ACF source-selected residual-target
pretraining recipe validated across three seeds. It uses one channel-agnostic
MambaTSv3 scan, an exact `P*d -> H96` residual prediction head, affine
train-only source ACF trajectory supervision calibrated to 10% of predictive
loss, and a zero-gated correction to a target-primary linear forecast.

For each new held-out seed, the target few-shot prefix is used only for two
inner selections before target training: ACF source ranking and a DLinear/NLinear
selection calibration tail. The target validation block selects the best target
epoch. The target test block is evaluated once from that selected checkpoint.

## Coordinates

| Target | Primary selection candidates | Source candidates | Held-out seeds |
| --- | --- | --- | --- |
| Traffic H96, 10% | DLinear, NLinear | PEMS03/04/07/08, METR-LA, PEMS-BAY, Electricity, Weather | 5201, 5202, 5203 |
| Weather H96, 10% | DLinear, NLinear | PEMS03/04/07/08, METR-LA, PEMS-BAY, Electricity, Traffic | 5204, 5205, 5206 |

Primary release comparison is hybrid versus the selected linear-primary
control. The one-sided non-inferiority gate is a paired upper ratio below
1.02; material improvement requires mean MSE below the selected control and
at least two paired wins per target. Test metrics are not read while computing
source ranks, primary selection, or target best epoch.

The release remains incomplete until all source-selection artifacts, target
inner-selection artifacts, target checkpoints, test summaries, and GPU guard
events are present and the post-run reader validates every coordinate.

## Invalid Smoke Boundary

One implementation smoke invoked `--evaluate-test` with only one target epoch
and eight target windows. Its payload correctly records a test metric, but it
does not satisfy this plan's frozen coordinates and is excluded from all
selection, comparison, summary, and manuscript claims. No confirmatory
held-out coordinate has been run under this plan.
