# Channel-Coverage InfoNCE Candidate

## Mechanism Change

The forecast-level InfoNCE objective is channel-specific: a pair is positive
for a channel only when the pair origin distance equals one of that channel's
train-prefix ACF peaks.  The prior source sampler used the union of all
channel peak lags.  With 64 windows and batch size four, many selected lags
were supported by only one or a few channels, leaving most channels without a
positive in a batch.

The candidate retains the same train-only per-channel ACF labels and the same
one-scan shared MambaTSv3 encoder.  It changes only the source batch lag set:
for `forecast_contrastive`, select lags found among at least 50% of source
channels; if none qualifies, use the most supported lag.  PEMS07, PEMS08, and
PEMS04 each select lags 288 and 576 with support 0.782--0.933.

## Fixed Validation-Only Pilot

| Item | Value |
| --- | --- |
| Target | PEMS-BAY, H=192, 10% labels, chronological 60/20/20 split |
| Source pool | PEMS07, PEMS08, PEMS04 |
| Source pairing | `channel_coverage`, threshold 0.5, local `contrastive_cluster` geometry |
| Control | Identical source and target sampling, `mamba_lambda=0` |
| Active | `mamba_lambda=0.1`, `relation_loss_fraction=0.1` |
| Paired seed | 5503 |
| Test access | Prohibited until the active validation MSE is no larger than the null validation MSE |

## Decision Boundary

The earlier `channel_coverage` attempt at seed 5502 is also diagnostic-only:
it used historical window selection, so its local pair groups had no valid
InfoNCE negatives.  It is not fine-tuned or evaluated.  This is a
mechanism-repair candidate.  It may be expanded only if the saved
validation checkpoint shows: (1) active InfoNCE anchor fraction at least 0.8
on each source epoch on average, and (2) active validation MSE no larger than
the exact-null control.  It cannot be merged into the frozen release table.

## Seed 5503 Readout

The registered gates passed: active anchor coverage was 1.0 in all three source
epochs; validation MSE was 2.765715 versus 2.811656 for the exact-null arm
(-1.634%); the subsequently opened held-out test MSE was 2.634552 versus
2.670468 (-1.345%). All guarded jobs completed, with the maximum observed
temperature at 54C and fan speed at 0%.

This is a single development seed, not confirmation evidence. The next
experiment must freeze this configuration and use a newly registered multi-seed
confirmation set, without reusing seed 5503 to make an inferential claim.
