# Shared Channel-Patch Projector Boundary

The Channel-Patch pretraining objective normally applies ACF-guided InfoNCE
directly to the shared MambaTSv3 encoder states. This validation-only,
target-excluded PEMS03 H192/25% screen tested the usual SSL projection-head
alternative: a shared `D -> D` MLP applied independently to every
`[channel, patch]` state before the Channel-Patch loss. The projector has no
channel-indexed parameter, no additional Mamba scan, and is not retained by
the target fine-tuning or inference graph.

| Channel-Patch loss input | Validation MSE | Difference |
| --- | ---: | ---: |
| Encoder states (identity) | 1.178145 | -- |
| Shared train-only MLP projector | 1.183210 | +0.430% |

Both arms use the same PEMS04/08/07 sources, PEMS03-excluded initial
checkpoints, ACF positives, pretraining budget, frozen target core, and seed
6089. The MLP arm has active Channel-Patch anchors for every source epoch and
stores its projector provenance in the pretraining artifact, so this is a
negative effectiveness result rather than an inactive-loss failure.

The pre-registered expansion requirement was at least `-0.25%` validation MSE
relative to identity. The MLP misses it and is not expanded to fresh seeds,
external baselines, robustness conditions, or test data. The deployed default
therefore remains direct encoder-state Channel-Patch InfoNCE.

All four guarded phases completed without a thermal/fan intervention; the
highest observed temperature was 55C. Machine-readable evidence is in
`artifacts/exploratory/pems03_channel_patch_projector_probe_20261007/audit.json`.
