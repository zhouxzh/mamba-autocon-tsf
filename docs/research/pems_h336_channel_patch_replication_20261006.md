# PEMS-BAY H336 Channel-Patch Replication

Status: fresh three-seed, validation-only long-horizon extension. No test row
was read.

The current channel-patch ACF representation mechanism was selected before
this extension from H192 PEMS-BAY/METR-LA evidence. This replication uses the
same Mamba-1 v3 `24/8` architecture, channel-count-agnostic shared encoder,
PEMS07/PEMS08/PEMS04 source pool, 10% chronological labels, and frozen target
core. Source next-patch prediction and target forecasting both use horizon
336. All channel-patch pretraining epochs have active ACF anchor fraction 1.0.

| Comparison | MSE | Relative change | Wins | One-SE ratio upper |
| --- | ---: | ---: | ---: | ---: |
| Channel-patch vs null | 2.381133 vs 2.469274 | -3.569% | 3/3 | 0.96759 |
| Channel-patch vs forecast-only | 2.381133 vs 2.409260 | -1.167% | 3/3 | 0.98870 |
| Channel-patch vs official PatchTST | 2.381133 vs 2.443893 | -2.568% | 3/3 | 0.98010 |

The six read-only validation stresses preserve external noninferiority in all
cells. Candidate versus official PatchTST has three wins per condition, with
one-SE ratio upper bounds from `0.97826` to `0.98686`. Long-horizon gains
therefore are not explained by only the ordinary validation origin.

This result is validation-only, uses a new seed set, and is not pooled with
the earlier H336 historical-release artifacts.

Artifacts:

- `artifacts/exploratory/pems_h336_channel_patch_replication_20261006/audit.json`
- `artifacts/exploratory/pems_h336_channel_patch_replication_20261006/robustness_validation/audit.json`
