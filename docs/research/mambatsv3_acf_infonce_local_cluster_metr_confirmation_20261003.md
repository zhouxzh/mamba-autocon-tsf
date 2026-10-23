# Local-Cluster InfoNCE METR-LA Confirmation

## Registered Question

Does the frozen local-identifiable forecast InfoNCE pretraining recipe improve
METR-LA H=192 forecasting at 10% target labels over exact-null predictive
pretraining when the target is excluded from the source pool?

## Fixed Protocol

| Item | Value |
| --- | --- |
| Target | METR-LA; chronological 60/20/20 split; 10% of train labels |
| Source selection | Train-prefix ACF selection recorded in `metr_la_sources.json`; no validation/test access |
| Source pool | PEMS-BAY, PEMS07, PEMS08; target excluded |
| Shared backbone | Frozen channel-count-independent one-scan MambaTSv3 core |
| Source objective | H=192 residual next-patch prediction, 3 epochs, 64 windows/source |
| Pair geometry | Channel peak support >= 0.5, local `contrastive_cluster`, batch size 4 |
| Active | Signed forecast InfoNCE, `mamba_lambda=0.1`, relation fraction `0.1` |
| Exact null | Identical source windows and tuning, with both relation weights zero |
| Target tuning | NLinear primary, Mamba residual correction, exact head transfer; 5 epochs, 64 windows |
| New paired seeds | 5520, 5521, 5522 |

## Lifecycle

All six pretraining and six validation-only target jobs complete before any
test reader runs. Test is opened for all six checkpoints only when every active
source run has anchor coverage >= 0.8 and every active validation MSE is no
larger than its paired null control. A failure leaves test closed and records a
negative cross-domain result. GPU jobs are serial and guarded at 78C / 85% fan.
