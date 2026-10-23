# Learned Channel Set-Pooling Replication

Status: completed frozen validation replication. No target test reader ran.

The candidate replaces the fixed shared channel mean/std summary with a
channel-count-independent learned set pool: a shared `D -> 1` score produces
per-patch channel attention, then a weighted mean and weighted standard
deviation are projected by a shared `2D -> D -> D` residual before the
existing single Mamba-1 scan. The pooling tensors are source-pretrained,
transferred, and frozen with the shared core. It is permutation-equivariant,
adds no second scan, and has no channel-indexed parameter.

Three new seeds compared pooling active, pooling exact-null, and ordinary
active InfoNCE under the frozen 10%-label PEMS-BAY/METR-LA protocol.

| Target | Pool active vs pool null | Pool active vs ordinary active | Decision |
| --- | ---: | ---: | --- |
| PEMS-BAY H192 | -3.128%, 3/3, upper .97116 | +0.344%, 2/3, upper 1.00725 | Does not replace ordinary active |
| METR-LA H192 | -0.428%, 3/3, upper .99738 | -1.410%, 3/3, upper .99033 | Positive on this target |

The pooling branch preserves a causal AutoCon improvement over its exact
predictive-pretraining null on both targets, but it is not a cross-domain
improvement over the ordinary shared encoder. The pre-registered replication
gate therefore fails and it is not eligible for a test release or the primary
method. The implementation remains as a reusable, audited high-dimensional
mechanism candidate, where fixed channel summaries are most likely to be
insufficient.

All 36 guarded stages completed serially without a trigger. Peak temperature
was 54C and peak fan was 32%.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_replication_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_replication_20261006/audit.json`
- `scripts/run_mambatsv3_mamba1_channel_set_pooling_replication.py`
- `scripts/audit_mambatsv3_mamba1_channel_set_pooling_replication.py`
