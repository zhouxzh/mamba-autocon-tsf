# Traffic d32 Channel-Patch Weighting Boundary

The high-dimensional Electricity-to-Traffic screen showed that equal-weight
full Channel-Patch InfoNCE was unfavorable. This registered follow-up changes
only the aggregation over active channels: `acf_strength` uses the median
absolute train-prefix ACF at each channel's selected positive lags, normalized
to mean one and clipped to `[0.25, 4.0]`. It does not alter the ACF pairs,
encoder, source/calibration data, forecast loss, Mamba scan count, or
inference graph.

| Aggregation | Validation MSE | Difference |
| --- | ---: | ---: |
| Uniform active channels | 0.821575 | -- |
| ACF-strength active channels | 0.821763 | +0.023% |

The weighting was nontrivial: the Electricity weights range from `0.25` to
`1.15`, and Traffic weights range from `0.25` to `1.31`. Channel-Patch
anchors were active in every epoch. The effect is nevertheless nearly zero and
unfavorable, missing the registered `-0.25%` expansion threshold. It is not
expanded to additional seeds or test data.

Both arms completed their four guarded stages without intervention, peaking at
54C. The weighting exists only in pretraining loss aggregation and is absent
from inference. Machine-readable evidence:
`artifacts/exploratory/traffic_d32_channel_patch_weighting_probe_20261007/audit.json`.
