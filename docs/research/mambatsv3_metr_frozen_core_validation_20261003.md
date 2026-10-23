# METR-LA Frozen-Core Validation Matrix

This fresh three-seed validation-only matrix evaluates the frozen shared-core,
target-trainable forecast-head protocol on METR-LA H192 with 10% labels.
It reuses the METR-LA train-prefix selected PEMS-BAY, PEMS07, and PEMS08 source
pool and local forecast-level ACF InfoNCE geometry, but starts new active/null
source pretraining runs at seeds 5820--5822. Target test rows are prohibited.
