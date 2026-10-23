# PEMS-BAY Frozen-Core Robustness Matrix

This registered validation-only extension evaluates the locked fresh
PEMS-BAY frozen-core active/null checkpoints from seeds 5800--5802. It uses
the six fixed validation conditions: in-distribution, late validation shift,
10% and 30% channel masking, and signal-relative Gaussian noise at 0.05 and
0.10. The evaluator reads only the chronological validation block and mutates
inference inputs only. It cannot access test rows or modify checkpoint weights.

## Completed Validation Readout

All six conditions favored active over exact-null for every seed. Mean relative
MSE changes were -1.356% in distribution, -1.256% under late-validation
shift, -1.227% with 10% masked channels, -0.957% with 30% masked channels,
-1.355% at 0.05 noise, and -1.354% at 0.10 noise. This is a validation-only
robustness extension of a protocol selected before these stress evaluations;
it does not open PEMS-BAY test rows.
