# Channel-Patch Representation Geometry

This read-only diagnostic evaluates the forecast-only and channel-patch source
checkpoints on PEMS07, PEMS08, and PEMS04 chronological training prefixes
only. For each source channel, its own train-only ACF peak supplies a signed
positive window pair; a low-absolute-ACF lag supplies a contrastive control.
The diagnostic reads encoder patch trajectories and performs no target
validation/test read or model update.

| Source | Forecast margin | Channel-patch margin | Margin change |
| --- | ---: | ---: | ---: |
| PEMS07 | 0.1581 | 0.2488 | +0.0908 |
| PEMS08 | 0.2611 | 0.3170 | +0.0560 |
| PEMS04 | 0.1556 | 0.2333 | +0.0776 |

The margin is signed ACF-positive cosine minus low-ACF control absolute cosine.
Across sources the mean improvement is `+0.0748`. The channel-patch objective
therefore produces the intended representational separation on every source
pool member, consistent with its downstream PEMS-BAY/METR-LA gains. This is a
mechanism diagnostic, not independent evidence of target accuracy or a test
result.

Artifact: `artifacts/exploratory/channel_patch_representation_geometry_20261007/audit.json`.
