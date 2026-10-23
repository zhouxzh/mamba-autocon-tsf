# Channel-Patch ACF Representation Replication

Status: independently registered, validation-only positive mechanism result.
No held-out test row was read, and no test metric was produced.

## Mechanism

The released MambaTS v3 encoder remains unchanged: one channel-count-agnostic
Mamba-1 scan over shared patch states. Ordinary pretraining applies signed ACF
InfoNCE to predicted future segments. The new pretraining-only term preserves
the encoder output as `[window, channel, patch, embedding]`, flattens only
each channel's own patch trajectory, and applies that channel's train-prefix
ACF peaks to choose positive window pairs. Other in-range windows remain
InfoNCE negatives. The additional term is calibrated to 5% of the detached
next-patch predictive loss. It has no parameters, no channel IDs, no extra
Mamba scan, and no inference-time computation.

This differs from the rejected global `repr_` probe, which pooled heterogeneous
channels before relation lookup, and from the rejected pooled-patch head,
which collapsed channels into one cross-domain vector before contrastive
matching.

## Independent Replication

The registered replication uses fresh seeds, separate from the initial
one-seed mechanism screen: PEMS-BAY `6026--6028`, METR-LA `6029--6031`.
All runs use H192, 10% chronological labels, Mamba-1, `d=24`, `d_ff=48`,
patch `24/8`, source next-patch pretraining, frozen target core, and the
existing forecast-level ACF InfoNCE. All channel-patch source epochs have
active ACF-anchor fraction `1.0`.

| Target | Null MSE | Forecast-only MSE | Channel-patch MSE | Channel-patch vs forecast | Wins | One-SE ratio upper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192 | 2.850772 | 2.772220 | 2.725045 | -1.702% | 3/3 | 0.98374 |
| METR-LA H192 | 1.602232 | 1.593703 | 1.576650 | -1.070% | 3/3 | 0.99037 |

The channel-patch candidate also improves over its exact null by `4.410%`
on PEMS-BAY and `1.597%` on METR-LA, with three wins in each target.

## Robustness And External Baseline

Same-seed official PatchTST checkpoints were trained from pinned commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` using the identical chronological
10% protocol. All checkpoints were then evaluated read-only on ordinary
validation, later validation origins, 10%/30% missing channels, and 0.05/0.10
signal-relative noise.

| Target | Channel-patch vs official PatchTST range | External wins | One-SE upper range |
| --- | ---: | ---: | ---: |
| PEMS-BAY H192 | -2.260% to -3.363% | 3/3 in all six conditions | 0.97360 to 0.98245 |
| METR-LA H192 | -3.257% to -4.749% | 3/3 in all six conditions | 0.95837 to 0.97150 |

The candidate also beats forecast-only and exact-null in every seed of every
registered stress cell. The result establishes low-label validation evidence
for a representation-level ACF objective, but is not a held-out test release
and must not be pooled with earlier seed sets.

At the separately registered 25% current-geometry replication, PEMS-BAY and
METR-LA retain channel-patch improvements over forecast-only of `-1.820%` and
`-0.951%`, respectively (3/3 each), and pass all six external PatchTST stress
cells. Details are in
[`25% external robustness`](channel_patch_25pct_external_robustness_20261006.md).

## Deployment Boundary

The new objective consumes encoder states only during source pretraining.
Target adaptation freezes the transferred core and trains the same target
forecast components as before. The saved target checkpoints do not add a new
forward module or scan; prediction inference retains the existing one-scan
MambaTS v3 graph and computes no AutoCon loss.

Artifacts:

- `artifacts/exploratory/mambats_channel_patch_representation_probe_20261006/audit.json`
- `artifacts/exploratory/mambats_channel_patch_replication_20261006/audit.json`
- `artifacts/exploratory/mambats_channel_patch_replication_20261006/robustness_validation/audit.json`
- `artifacts/exploratory/official_patchtst_channel_patch_replication_20261006/plan.json`

## Frozen Test Boundary

A later frozen-checkpoint, read-only test extension retains the mechanism gain
on PEMS-BAY and METR-LA but reverses it relative to forecast-only on Weather.
The workspace has historical public test access and the extension cannot drive
selection. This failure limits the confirmed causal claim to the two traffic
targets; see
[`frozen test extension`](channel_patch_frozen_test_extension_20261006.md).
