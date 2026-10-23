# ACF-Guided Source Selection Protocol

## Motivation

All-source affine ACF pretraining improved Traffic but degraded Weather. This
protocol tests whether negative transfer comes from sources whose train-prefix
autocorrelation signatures are unlike the target.

The first implementation used the target chronological 60% training prefix
to form its signature. That is train-only but violates the stricter 10%/25%
low-data claim, so its source-selected results are retained as audit-only.
The replacement selector uses only the declared target few-shot prefix.

For each target, source selection uses only the target chronological 60%
training prefix. The signature is the channel-mean signed ACF at lags 1--336,
augmented with the mean and standard deviation of per-channel absolute ACF,
then L2 normalized. Sources are ranked by cosine similarity. No validation or
test row is read during source selection.

## Frozen Source Pools

| Target | Excluded target | Top-3 train-only ACF sources | Similarity |
| --- | --- | --- | --- |
| Traffic | Traffic | Electricity, Weather, METR-LA | .950, .084, .079 |
| Weather | Weather | METR-LA, PEMS-BAY, Electricity | .889, .573, .361 |

Both arms retain the exact-H96 `P*d -> 96` transferable head, the
channel-agnostic one-Mamba-block encoder, direct signed affine future-trajectory
relation, 10% calibrated relation-loss share, three source epochs, and the
same 10% target adaptation budget. The only changed factor relative to the
all-source affine treatment is the registered source pool.

Promotion requires improvement over both the all-source affine treatment and
the DLinear-primary control in both target validation domains. Target tests
remain closed.
