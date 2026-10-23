# Channel-Patch ACF-Magnitude Weighting Boundary

This registered PEMS03 H192/25% screen retained the full Channel-Patch
trajectory objective but weighted only its positive InfoNCE terms by absolute
train-prefix ACF magnitude. Sources, target exclusion, initialization,
prediction loss, Mamba-1 `24/8` backbone, and frozen target head matched the
same-seed ordinary Channel-Patch arm.

On seed `6064`, ordinary Channel-Patch reaches `1.191174` validation MSE and
magnitude weighting reaches `1.191015`, a negligible `-0.013%` change. It
does not meet the pre-registered `-0.25%` expansion threshold. The weighted
loss remains active throughout pretraining, so this is not a zero-support
failure.

No multi-seed, external-comparator, robustness, or test extension is justified
for this negative screen. Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_confidence_probe_20261007/audit.json`
