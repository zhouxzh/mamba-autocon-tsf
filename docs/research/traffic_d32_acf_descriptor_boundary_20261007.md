# Traffic d32 ACF Descriptor Boundary

This validation-only experiment moves train-prefix ACF information from loss
labels into the shared patch encoder itself. For every source or target
calibration role, a six-value per-channel descriptor is computed only from
that role's chronological training prefix: leading signed peak, median peak
magnitude, mean and maximum ACF magnitude, normalized leading lag, and peak
coverage. A shared `6 -> D` projection is zero-gated into patch tokens before
the existing single Mamba-1 scan.

The descriptor buffer is role-local rather than transferable. Pretraining
transfers only the shared projection and gate; target adaptation rebuilds a
fresh `[862, 6]` Traffic buffer from its 25% train prefix. The trained gate is
nonzero (`0.01068`), so this is not a dormant zero-initialized branch.

| Arm | Validation MSE | Difference |
| --- | ---: | ---: |
| Forecast calibration | 0.817925 | -- |
| Ordinary Channel-Patch | 0.819953 | +0.248% vs forecast |
| ACF-descriptor Channel-Patch | 0.819027 | +0.135% vs forecast; -0.113% vs ordinary Channel-Patch |

The descriptor reduces the ordinary Channel-Patch regression but does not
surpass the forecast control and misses both pre-registered `-0.25%` gates.
It is not expanded to further seeds, robustness, external baselines, or test
data. At inference it adds a fixed descriptor projection but no AutoCon loss
and no additional Mamba scan.

All six guarded stages completed without intervention, peaking at 55C. The
machine-readable audit is
`artifacts/exploratory/traffic_d32_acf_descriptor_probe_20261007/audit.json`.
