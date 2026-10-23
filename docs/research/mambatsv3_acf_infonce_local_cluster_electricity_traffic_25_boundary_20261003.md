# Electricity and Traffic 25% Boundary Matrix

## Registered Scope

This validation-only matrix tests the frozen local-cluster InfoNCE recipe at
25% target labels on two high-dimensional domains where earlier 10% studies
showed weak or negative transfer. It is deliberately a boundary experiment,
not a search for a favorable result.

| Target | Horizon | Source pool | New seeds |
| --- | ---: | --- | --- |
| Electricity | 96 | Traffic, METR-LA, Weather | 5550--5552 |
| Traffic | 96 | Electricity, METR-LA, PEMS-BAY | 5560--5562 |

The source lists are fixed by the target train-prefix selectors already stored
in the repository. The target is excluded from source pretraining. Active and
exact-null use identical windows, initialization, epochs, transfer head, and
target tuning; only the relation objective weights differ. The source geometry
is channel-support threshold 0.5 with local `contrastive_cluster` batches.

All source and target jobs are validation-only. Test access is prohibited in
this boundary matrix, regardless of whether active wins or loses validation.
This prevents a negative result from being hidden and keeps the matrix useful
for the paper's applicability boundary.
