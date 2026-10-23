# Channel-Patch Stability Weighting Boundary

The ordinary Channel-Patch objective treats each selected per-channel ACF peak
equally. This screen tested an alternative that weights only its positive
InfoNCE terms by signed-ACF agreement between the two chronological halves of
each source training prefix. It retains the same sources, Mamba-1 `24/8`
backbone, one scan, ACF positives, prediction loss, and frozen PEMS03 H192/25%
target adaptation. Forecast-level InfoNCE remained unweighted in both arms.

On registered PEMS03 seed `6063`, ordinary Channel-Patch reaches `1.185252`
validation MSE and stability weighting reaches `1.185348` (`+0.008%`). It
misses the pre-registered `-0.25%` improvement threshold and is not expanded.
The source stability weights average `0.964`, `0.955`, and `0.962` for PEMS04,
PEMS08, and PEMS07, respectively, so this specific data-derived weighting is
nearly uniform and cannot materially reallocate the loss.

No external comparator, stress matrix, or test read is warranted for this
negative screen. Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_stability_probe_20261007/audit.json`
