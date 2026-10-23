# Pooled-Patch Anti-Collapse Probe

The initial pooled-patch probe produced nearly identical representations across
Traffic source windows: the active representation's off-diagonal cosine was
0.9955 and its mean cross-window feature variance was 0.00097. Therefore its
small validation gain could not be interpreted as useful representation
learning.

This follow-up adds a pretraining-only batch-variance floor to the pooled patch
projection. The relation loss remains calibrated to 10% of the predictive
objective; the variance term changes the composition of that relation gradient
without changing the Mamba scan count, prediction head, or inference graph.

For Traffic seed 5770, the active pretraining feature standard deviation rose
from 0.344 to 0.498 across the three epochs, above the 0.2 target. Yet active
validation MSE was 0.818614 versus 0.818512 for exact-null (+0.012%). The
screen is validation-only and does not justify a held-out test or a claim that
the anti-collapse term improves forecasting. It distinguishes an achieved
representation-geometry correction from absent downstream gain.
