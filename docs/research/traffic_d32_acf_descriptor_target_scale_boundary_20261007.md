# Traffic d32 Target-Tuned ACF Descriptor Boundary

The prior ACF-descriptor screen found a small reduction in the ordinary
Channel-Patch regression but still missed forecast calibration. This follow-up
keeps the transferred descriptor projection, patch encoder, single Mamba scan,
and target descriptor buffer frozen. The sole added trainable parameter is the
existing scalar ReZero gate on the ACF descriptor; it is learned from Traffic's
25% chronological train prefix.

| Arm | Validation MSE | Difference |
| --- | ---: | ---: |
| Forecast calibration | 0.815086 | -- |
| Descriptor, frozen target gate | 0.826051 | +1.345% vs forecast |
| Descriptor, target-tuned scalar | 0.826049 | -0.00025% vs frozen gate |

The target scalar is active and changes from `-0.00082` to `0.01836` over five
epochs. Every history record confirms that the shared core remains frozen;
only this scalar is unfrozen. Despite the intended target adaptation, the
effect is negligible and remains far above the forecast control, so the
registered `-0.25%` expansion gate fails. No additional seeds, stress tests,
external baselines, or test metrics are authorized.

All five guarded stages completed without intervention, with a peak
temperature of 56C. The audit artifact is
`artifacts/exploratory/traffic_d32_acf_descriptor_scale_probe_20261007/audit.json`.
