# Local-Cluster InfoNCE Weather Confirmation

## Registered Question

Does the frozen locally identifiable forecast-InfoNCE recipe improve Weather
H=192 forecasting at 10% target labels over exact-null predictive pretraining
with the Weather target excluded from all source pretraining?

## Fixed Protocol

| Item | Value |
| --- | --- |
| Target | Weather; chronological 60/20/20 split; 10% of train labels |
| Source selection | Weather train-prefix selection record; no validation/test access |
| Source pool | METR-LA, PEMS-BAY, Electricity; target excluded |
| Shared backbone | Frozen channel-count-independent one-scan MambaTSv3 core |
| Source objective | H=192 residual next-patch prediction, 3 epochs, 64 windows/source |
| Pair geometry | Channel peak support >= 0.5, local `contrastive_cluster`, batch size 4 |
| Active | Signed forecast InfoNCE, `mamba_lambda=0.1`, relation fraction `0.1` |
| Exact null | Identical source windows and target tuning, with both relation weights zero |
| Target tuning | NLinear primary, Mamba residual correction, exact head transfer; 5 epochs, 64 windows |
| New paired seeds | 5530, 5531, 5532 |

## Lifecycle

All source-pretraining and validation-only target jobs complete before a test
reader starts. Held-out test results are read only if every active source run
has anchor coverage >= 0.8 and every active validation MSE is no larger than
its paired exact-null control. A failure preserves the full validation matrix
as an out-of-domain negative result. GPU jobs are serial and guarded at 78C /
85% fan.
