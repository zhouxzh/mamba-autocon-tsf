# PEMS-BAY H336 Local-Cluster Confirmation

## Registered Question

Does the frozen local-identifiable forecast InfoNCE protocol improve PEMS-BAY
H=336 forecasting at 10% target labels relative to exact-null predictive
pretraining?

## Fixed Protocol

| Item | Value |
| --- | --- |
| Target | PEMS-BAY, chronological 60/20/20 split, 10% labels |
| Source pool | PEMS07, PEMS08, PEMS04; selected before prior H192/H336 releases |
| Source and target horizon | 336, exact head transfer |
| Shared core | Existing channel-count-independent one-scan MambaTSv3 |
| Pair geometry | Train-only ACF, support >= 0.5, local `contrastive_cluster` batch size 4 |
| Active / null | `mamba_lambda=0.1`, fraction `0.1` / both zero |
| Source / target budgets | 3 epochs x 64 windows/source / 5 epochs x 64 target windows |
| New paired seeds | 5600, 5601, 5602 |

All source and validation-only target runs finish before any test evaluation.
The six held-out test readers run only if every active source run has coverage
>= 0.8 and every active validation MSE is no larger than the paired null.
The historical H336 result is not pooled into this confirmation.
