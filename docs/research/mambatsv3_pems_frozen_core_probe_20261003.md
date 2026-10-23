# PEMS-BAY Frozen-Core Representation Probe

This fresh seed 5800 validation-only probe applies frozen shared-core and
target-trainable forecast-head adaptation to the PEMS-BAY H192, 10% label
setting. It uses the same pre-registered PEMS07, PEMS08, and PEMS04 sources,
source geometry, and local forecast-level ACF InfoNCE as the prior positive
PEMS-BAY confirmation, but reruns source pretraining from scratch with a new
seed and does not access test rows.

Active and exact-null arms share source data, initialization, training budget,
and target adaptation. During target adaptation, transferable core parameters
remain frozen while the target forecast head, NLinear head, and residual scale
train. A favorable one-seed validation result can only justify a separately
registered three-seed validation matrix; it cannot open test evaluation.

## Completed Fresh Validation Matrix

Seeds 5800--5802 all favored active over exact-null while preserving the
frozen-core/target-head contract. Mean validation MSE was 2.86389 for active
and 2.90325 for null, a relative change of -1.356% (3/3 active wins). These
are fresh validation results only; no PEMS-BAY test rows were opened.

The deployed active checkpoint (`s5800_active_frozen_core.pt`) was benchmarked
with prediction-only execution: one Mamba scan, no returned AutoCon
representation, no relation loss, 0.321ms median latency at batch size four,
and 18.49MB peak allocated memory. The pretraining relation objective and
frozen-core adaptation policy therefore add no inference-time scan or loss.
