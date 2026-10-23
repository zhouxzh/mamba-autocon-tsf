# PEMS-BAY 25% Label Confirmation

## Registered Question

After the frozen local-cluster InfoNCE protocol succeeds at 10% labels, is it
still non-inferior to exact-null predictive pretraining at 25% target labels
on PEMS-BAY H=192?

## Fixed Protocol

The source pool remains PEMS07, PEMS08, and PEMS04. The shared one-scan
MambaTSv3 encoder, train-only channel ACF positive construction, support
threshold 0.5, local `contrastive_cluster` batch geometry, next-patch task,
active weight 0.1, exact-null control, target head, and all epochs/window
budgets are identical to the confirmed 10% protocol. The only changed
experimental factor is `few_shot_fraction=0.25`.

New paired seeds are 5540, 5541, and 5542. All six source/target validation
jobs finish before test access. Tests are read only if every active source run
has coverage >= 0.8 and every active validation MSE is no larger than null.
This is a new confirmation, not a pooled extension of the 10% result.
