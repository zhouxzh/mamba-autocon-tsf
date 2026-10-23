# METR-LA 25% Label Confirmation

## Registered Question

Does the frozen local-cluster InfoNCE recipe remain non-inferior to exact-null
predictive pretraining on METR-LA H=192 with 25% target labels?

## Frozen Protocol

The target remains METR-LA with the previously registered train-prefix source
pool PEMS-BAY, PEMS07, and PEMS08. The shared one-scan core, channel-support
threshold 0.5, `contrastive_cluster` geometry, residual next-patch task,
active/null contrast, target NLinear residual hybrid, exact H192 head transfer,
and source/target budgets are unchanged. The sole experimental factor is
`few_shot_fraction=0.25`.

New paired seeds: 5580, 5581, 5582. Test stays closed until every active source
run reaches 0.8 anchor coverage and every active validation MSE is no greater
than its exact-null pair. A failed all-seed gate leaves the test unread.
