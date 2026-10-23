# Traffic d32 ACF-Grouped Patch Boundary

This validation-only experiment tests a loss-level, channel-count-independent
representation alternative. Each source or target-calibration training prefix
sorts channels by its dominant train-only ACF lag and splits them into four
balanced fixed groups. Patch states are mean-pooled within each group, a
separate group-level ACF prior supplies positives and negatives, and the
resulting grouped-patch InfoNCE replaces, rather than supplements, ordinary
per-channel Channel-Patch InfoNCE. It adds no parameter, Mamba scan, or
inference computation.

| Arm | Validation MSE | Difference |
| --- | ---: | ---: |
| Forecast calibration | 0.818652 | -- |
| Four-group ACF patch representation | 0.825248 | +0.806% |

The grouping is not degenerate: Electricity group sizes are `81/80/80/80`,
Traffic group sizes are `216/216/215/215`, and group-level active-anchor
coverage is `0.906`, `0.917`, and `0.940` over the three pretraining epochs.
The candidate nevertheless fails the registered `-0.25%` expansion threshold
by a wide margin, so it is not expanded to seeds, robustness, external
baselines, or test data.

All guarded stages completed normally with a peak temperature of 55C. The
group construction and group InfoNCE exist only during pretraining; deployed
inference is unchanged. Machine-readable evidence:
`artifacts/exploratory/traffic_d32_grouped_patch_probe_20261007/audit.json`.
