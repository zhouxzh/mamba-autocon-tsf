# PEMS-BAY H336 Robustness Extension

This post-confirmation extension reads only the chronological validation block
of the frozen H336/10%-label checkpoints for seeds 5600--5602. It evaluates
ordinary validation, late origins, 10%/30% channel masking, and 0.05/0.10
input noise. It does not train, select, update, or retest any checkpoint.

The results are reported separately from the H336 held-out causal confirmation.
All conditions are retained, including any active degradation.
