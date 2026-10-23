# MambaTS Backbone-Family v13 Screen

Status: completed validation-only negative pilot. No held-out test reader was
called.

## Question And Contract

The release backbone is the channel-count-agnostic, one-scan MambaTS v3. This
registered pilot relaxed that outer-backbone restriction only to include v13:
the same Mamba-1 scan followed by shared depthwise kernel-3/kernel-5 filters
over each channel's patch states. The filters have no channel-ID parameters,
so they can be pretrained, transferred, and frozen under the existing shared
core contract.

All arms use PEMS-BAY H192, 10% chronological labels, Mamba-1, `d=24`,
`d_ff=48`, patch `24/8`, the fixed PEMS07/PEMS08/PEMS04 sources, source
next-patch pretraining, and the same frozen-core target adaptation budget. New
seeds are `6021` and `6022`. For each seed, the pilot compares v3 active,
v13 lambda-zero, and v13 active. The registered continuation rule required
v13 active to beat both v3 active and its v13 null in both seed pairs.

## Result

| Seed | v3 active validation MSE | v13 null validation MSE | v13 active validation MSE |
| ---: | ---: | ---: | ---: |
| 6021 | 2.770511 | 2.841561 | 2.775263 |
| 6022 | 2.775834 | 2.842184 | 2.763839 |
| Mean | 2.773172 | 2.841873 | 2.769551 |

v13 active improves over its exact lambda-zero control by `2.545%` and wins
both seed pairs. It does not robustly improve over the release v3 active:
the mean change is `-0.131%`, it wins only `1/2` pairs, and the one-SE paired
ratio upper bound is `1.00172`. Therefore the v3 comparison gate fails.

## Decision

Stop v13. It has an internally responsive relation objective, but the extra
post-scan filters do not provide enough reproducible gain over the already
channel-agnostic v3 active baseline. No METR-LA extension, stress matrix,
held-out test reader, or post-hoc filter tuning is authorized from this pilot.
The released Mamba-1 v3 configuration remains the selected backbone.

Machine-readable plan and audit:

- `artifacts/exploratory/mambats_backbone_family_v13_screen_20261006/plan.json`
- `artifacts/exploratory/mambats_backbone_family_v13_screen_20261006/audit.json`
