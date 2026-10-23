# PEMS-BAY Frozen-Core 25% Validation Matrix

This registered matrix applies the frozen shared-core, target-trainable
forecast-head protocol to PEMS-BAY H192 with 25% target labels. It uses the
same PEMS07, PEMS08, and PEMS04 source pool and source pretraining geometry as
the 10% frozen-core matrix, but uses fresh seeds 5810--5812 and a separate
validation-only output root. Active and exact-null differ only in source
AutoCon relation weights. No command in the matrix is allowed to evaluate
target test rows.
