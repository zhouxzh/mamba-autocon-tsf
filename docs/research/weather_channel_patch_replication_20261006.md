# Weather Channel-Patch External Replication

Status: independent five-seed validation-only external-domain replication. No
test rows were read.

The channelwise patch-trajectory ACF InfoNCE mechanism and its `0.05` fraction
were selected using PEMS-BAY and METR-LA evidence only. Weather uses its
previously registered independent seeds `5991--5995`, the frozen source pool
(METR-LA, PEMS-BAY, Electricity), and already locked same-seed forecast-only,
null, and official PatchTST checkpoints. Only the new candidate is trained.
All source-pretraining epochs had channel-patch active anchor fraction `1.0`.

| Comparison | Validation MSE | Relative change | Wins | One-SE ratio upper |
| --- | ---: | ---: | ---: | ---: |
| Channel-patch vs forecast-only | 4.503699 vs 4.510360 | -0.148% | 4/5 | 0.99943 |
| Channel-patch vs null | 4.503699 vs 4.536690 | -0.727% | 5/5 | 0.99336 |
| Channel-patch vs official PatchTST | 4.503699 vs 4.618591 | -2.488% | 5/5 | 0.98375 |

Six validation stresses preserve the mechanism's external noninferiority:
later validation origins, 10%/30% missing channels, and 0.05/0.10
signal-relative noise all have one-SE PatchTST ratio upper bounds below one
(`0.97835--0.99356`). The candidate beats the frozen forecast-only control in
four of five seeds for every stress condition and beats its null in all five.

This verifies that the representation mechanism is not confined to traffic
sensor targets, although the incremental gain over forecast-only is smaller
than on PEMS-BAY and METR-LA. It remains validation-only and is not pooled
with any test release.

Artifacts:

- `artifacts/exploratory/weather_channel_patch_replication_20261006/audit.json`
- `artifacts/exploratory/weather_channel_patch_replication_20261006/robustness_validation/audit.json`
