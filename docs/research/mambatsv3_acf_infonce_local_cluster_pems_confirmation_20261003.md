# Local-Cluster InfoNCE PEMS-BAY Confirmation

## Registered Question

After making forecast-level ACF InfoNCE locally identifiable in every source
mini-batch, does the active source-pretraining arm improve PEMS-BAY H=192
forecasting with 10% target labels relative to its exact-null pretraining
control?

The prior seed 5503 is a development result and is excluded from this
confirmation set. The configuration below was fixed before starting these
three new seeds. No architecture, source pool, ACF threshold, batch geometry,
loss weight, or training duration may be changed as a consequence of their
validation results.

## Fixed Arms

| Item | Value |
| --- | --- |
| Target | PEMS-BAY; chronological 60/20/20 split; target prefix 10% of train |
| Source pool | PEMS07, PEMS08, PEMS04; target excluded |
| Shared encoder | Existing channel-count-independent one-scan MambaTSv3 core |
| Source task | H=192 residual next-patch prediction, 3 epochs, 64 windows/source |
| Pair geometry | Train-only channel ACF; support threshold 0.5; local `contrastive_cluster`; batch size 4 |
| Active | Forecast signed InfoNCE, `mamba_lambda=0.1`, calibrated fraction `0.1` |
| Exact null | Identical source windows, pairing, seed, prediction task, and target tuning; both relation weights zero |
| Target tuning | NLinear-primary residual correction, exact source head transfer, 5 epochs, 64 windows |
| New paired seeds | 5510, 5511, 5512 |
| Resources | One guarded GPU job at a time; four CPU threads; 78C / 85% limits |

## Locked Lifecycle

1. Complete all six source-pretraining and all six validation-only target
   fine-tuning jobs before opening any confirmation test data.
2. Proceed to the six read-only held-out test evaluations only if each active
   seed has validation MSE no larger than its paired null seed and each active
   source run has mean anchor coverage at least 0.8.
3. If the gate fails, record the complete validation matrix as a negative or
   mixed confirmation and leave all confirmation test partitions unread.
4. A successful confirmation establishes only the PEMS-BAY H=192/10% claim.
   METR-LA, Weather, Electricity, and Traffic remain independent follow-up
   domains; no pooled universal claim is permitted.
