# Weather Frozen-Core Validation Matrix

This fresh three-seed validation-only matrix tests the frozen shared-core,
target-trainable forecast-head protocol on Weather H192 with 10% labels.
It reuses Weather's train-prefix selected METR-LA, PEMS-BAY, and Electricity
source pool and local forecast-level ACF InfoNCE geometry, while using new
active/null source pretraining seeds 5830--5832. Target test rows are
prohibited.

## Completed Validation Readout

All three seeds favored active over exact-null. Mean validation MSE was
4.48383 for active and 4.49810 for null, a -0.317% relative difference. The
smaller effect than the traffic-network matrices is retained. This matrix does
not authorize test access.
