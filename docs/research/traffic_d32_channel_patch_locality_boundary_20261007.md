# Traffic d32 Channel-Patch Locality Boundary

In the high-dimensional Electricity-to-Traffic d32/24 protocol, the usual
candidate applies Channel-Patch InfoNCE to both the Electricity source and the
Traffic 25% training-prefix calibration role. This validation-only screen
tests whether the target-prefix representation loss interferes with source
transfer. Both arms retain the same source Channel-Patch fraction `.05`,
forecast relation objectives, target forecast calibration, target frozen core,
single Mamba scan, and prediction head; the source-only arm changes the
Traffic Channel-Patch fraction from `.05` to `0`.

| Channel-Patch placement | Validation MSE | Difference |
| --- | ---: | ---: |
| Electricity source and Traffic calibration | 0.826616 | -- |
| Electricity source only | 0.826262 | -0.043% |

The source-only direction is favorable but misses the registered `-0.25%`
expansion threshold by a large margin. Its artifact confirms the intended
role split: Electricity receives Channel-Patch fraction `.05`, while Traffic
receives `0`, and Traffic forecast calibration remains `.3`. No test rows are
opened. All four guarded stages completed without intervention, peaking at
55C.

This result does not support promoting a source-only high-dimensional method,
but it rules out a large target-prefix Channel-Patch interference explanation
for the current Traffic limitation. Machine-readable evidence:
`artifacts/exploratory/traffic_d32_channel_patch_locality_probe_20261007/audit.json`.
