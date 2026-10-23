# Weather 25% Label Confirmation

## Registered Question

Does the frozen local-cluster InfoNCE recipe remain non-inferior to exact-null
predictive pretraining on Weather H=192 with 25% target labels?

## Frozen Protocol

The source policy is fixed from the previously recorded Weather 10%-prefix
selection: METR-LA, PEMS-BAY, and Electricity. It is deliberately not
reselected from the larger 25% target prefix. The one-scan shared core,
channel-support threshold 0.5, `contrastive_cluster` geometry, residual
next-patch task, active/null contrast, target NLinear residual hybrid, exact
head transfer, and source/target budgets remain unchanged. The only changed
factor is `few_shot_fraction=0.25`.

New paired seeds: 5590, 5591, 5592. All source and validation-only jobs finish
before test access. Tests are read only if all active anchor coverages are at
least 0.8 and all active validation MSE values are no larger than null.
