# Mamba-1 High-Dimensional Negative Boundary

The fast frozen Mamba-1 24/8 architecture was tested once on each previously
negative high-dimensional target before deciding whether to extend the released
three-domain method to Electricity and Traffic. Both probes are validation-only
and opened no target test rows.

| Target | Labels | Source | Active MSE | Exact-null MSE | Active vs null |
| --- | ---: | --- | ---: | ---: | ---: |
| Electricity H96 | 25% | Traffic | 8.243153 | 8.143744 | +1.221% |
| Traffic H96 | 25% | Electricity | 0.817744 | 0.816433 | +0.161% |

Both use source-predictive pretraining and the same channel-count-agnostic
shared patch encoder, 24/8 patch geometry, Mamba-1 mixer, strict
coverage-only ACF positives, 192 source windows, exact horizon head transfer,
and frozen target core. The active arm alone adds train-only ACF relation
pretraining. The result therefore rules out this optimized source-only ACF
transfer recipe as a general high-dimensional remedy.

No further high-dimensional seeds or test reads are justified on this failed
direction. These results must remain in the paper as a negative boundary.

## Target-Prefix Calibration Follow-up

Electricity was additionally calibrated with its own declared few-shot training
prefix: the shared encoder first saw Traffic's source-training prefix plus the
earliest 15% of Electricity's raw timeline (60% chronological train region x
25% few-shot fraction). Both active and null arms used this identical
calibration data and next-patch prediction objective; the active arm alone used
ACF relation pairs. The result remained negative: active `8.179109` versus
null `8.132055` (`+0.579%`). The calibration metadata records Electricity's
role, 3,945 prefix rows, and `train_prefix_fraction=0.15`; no validation or
test row was opened.

This failure rules out a simple explanation that high-dimensional degradation
is caused only by an entirely source-only representation. No Electricity test
metric was generated.

## Target Decoder Follow-ups

Two zero-gated shared target decoders were evaluated while keeping the Mamba-1
core frozen and retaining the identical Traffic-plus-Electricity calibration
data in active and null arms. The rank-8 channel adapter pools existing patch
states per channel; the rank-32 target MLP adapter retains the full patch-state
trajectory. Neither invokes another Mamba scan.

| Target decoder | Active MSE | Exact-null MSE | Active vs null |
| --- | ---: | ---: | ---: |
| Rank-8 channel adapter | 8.171434 | 8.109607 | +0.763% |
| Rank-32 target MLP adapter | 8.222950 | 8.214488 | +0.103% |

Both adapters were trainable in every target epoch while the transferred Mamba
core stayed frozen. Neither high-dimensional adapter run opened test data. The
shared target-capacity explanation is therefore also unsupported under this
protocol.

## Prediction-connected Representation Fusion

Finally, a channelwise AutoCon representation fusion was connected directly to
each forecast channel. It reuses the same single Mamba patch-state sequence,
applies a shared channelwise representation projection and a zero-initialized
forecast gate, and does not add a second Mamba scan. Both active and null arms
used the identical target-prefix calibration and fusion capacity. The active
result `8.247402` was worse than exact-null `8.238243` (`+0.111%`).

This closes the direct-representation hypothesis under the current high-
dimensional protocol. No test metric was generated and no further high-
dimensional architecture variants are promoted.

## Transferable Channel Summary

A final channel-count-agnostic context uses each patch's channel mean and
standard deviation, maps the `2*d_model` descriptor through a shared
`2D -> D -> D` projection, and injects it before the existing Mamba scan. Its
three projection/gate tensors were transferred from Traffic-plus-Electricity
pretraining and frozen with the shared target core. The active result
`8.347165` remained worse than exact-null `8.341368` (`+0.070%`). This is the
valid transfer-controlled summary result; an earlier non-transferred summary
probe is excluded from interpretation.

The source-only, target-prefix, target-adapter, direct-representation, and
fully transferable channel-summary hypotheses now all fail the high-dimensional
active/null gate. No more high-dimensional architecture variants are planned
under the current objective.

## Learned Channel Set-Pooling Boundary

A later learned, permutation-equivariant set-pooling context replaced the
fixed channel mean/std summary. It applies a shared `D -> 1` channel attention
per patch, then injects weighted mean/std context before the existing one
Mamba scan; its parameters are independent of channel count and are transferred
and frozen with the core. In the Electricity strong-calibration probe it
retained an active/null improvement of `-0.536%`, but its active MSE `8.181410`
was `+0.181%` worse than the matched plain active `8.166603`. A prior frozen
PEMS-BAY/METR-LA replication likewise showed pooling active favorable to its
own null on both targets but `+0.344%` worse than ordinary active on PEMS-BAY.
The mechanism is therefore not promoted; see
[`set-pooling Electricity probe`](mambatsv3_channel_set_pooling_electricity_probe_20261006.md).

## Multi-Prototype Pooling Boundary

To distinguish a single global summary from heterogeneous channel groups, a
four-query shared prototype set pool was also evaluated before the same one
Mamba scan. It retained a `-0.577%` active/null gain but was `+0.195%` worse
than plain strong-calibration active (`8.188600` versus `8.172686`). Fixed
mean/std, learned single-summary, and four-prototype summaries therefore all
fail to improve the matched plain high-dimensional active model. This entire
scan-pre-context family is closed; see
[`prototype pooling probe`](mambatsv3_prototype_pooling_electricity_probe_20261006.md).

## Strong Target-Calibration Relation Recovery

A different training-only intervention then increased only the Electricity
calibration prefix's relation-loss fraction from 0.1 to 0.3; Traffic remained
at 0.1, and active/null used identical predictive calibration data. Across
three seeds, active improved over null by `0.394%` with 3/3 wins and a one-SE
upper ratio of `0.99682`. The source metadata records the two distinct
fractions and 3,945 Electricity prefix rows.

This is the first positive high-dimensional active/null result under the
current protocol. It does not establish external SOTA: the same-coordinate
official PatchTST validation mean was approximately `7.537`, below the active
mean `8.178`. The next legitimate step would be a separately frozen external
baseline/test plan, not further in-place architecture tuning.

## Staged Core Adaptation Boundary

One frozen warm-up epoch followed by low-rate (`1e-4`) target adaptation of
the transferred shared core retained a small active/null advantage (`-0.132%`)
but made active validation MSE worse than fully frozen active (`8.256705` vs
`8.205705`). This rejects simple target-side core unfreezing as an explanation
for the external gap; see
[`staged core probe`](mambatsv3_staged_core_electricity_probe_20261006.md).

## d32/24 Geometry Bridge

A compatible PEMS d32/24 initialization was transferred through the current
Traffic-plus-Electricity few-shot SSL protocol. Three fresh seeds improved
active MSE over the d24/8 frozen transfer by `1.728%` and retained active/null
`-0.795%`, but remained `+3.462%` behind same-geometry official PatchTST
(0/3; one-SE upper `1.03743`). This isolates a real capacity/patch-geometry
effect without rescuing high-dimensional external parity; see
[`d32/24 bridge`](mambatsv3_d32_stride24_electricity_bridge_20261006.md).

The symmetric Electricity-source plus Traffic-calibration d32/24 bridge
improved Traffic's absolute active error slightly but reversed the active/null
effect (`+0.523%`; `0.816451` versus `0.812202`) despite 0.875 active anchor
coverage. This closes the high-dimensional few-shot bridge for Traffic; see
[`Traffic d32/24 bridge`](mambatsv3_d32_stride24_traffic_bridge_20261006.md).

## Channel-Patch Representation Recovery And External Failure

A distinct pretraining-only objective retains every channel's complete patch
trajectory and uses that channel's own train-prefix ACF peaks for InfoNCE
positives. It avoids the rejected global representation and pooled-channel
paths, has no new scan or inference computation, and is active in every source
epoch. On the strong Electricity calibration protocol, three fresh seeds
improve channel-patch active over forecast-only active by `-0.653%` (3/3,
one-SE upper `0.99481`). This is a valid causal high-dimensional improvement.

However, same-seed official PatchTST remains lower by `8.590%` (candidate
`8.148099`, PatchTST `7.503567`, 0/3, upper `1.08883`). The objective is
therefore retained as a transferable representation mechanism, not promoted as
a high-dimensional external-SOTA solution. No stress or test expansion is
authorized from this external failure; see
[`Electricity channel-patch boundary`](electricity_channel_patch_representation_20261006.md).

## Target Channelwise Correction Gate

The original hybrid used one global Mamba-correction scalar. Replacing it only
at target adaptation by a zero-initialized per-channel gate leaves the shared
encoder, source SSL, and single scan unchanged. In the d32/24 Electricity
bridge, three fresh seeds improve over the scalar-gate control by `-3.362%`
with 3/3 wins. The five-seed same-geometry official PatchTST comparison is a
near-tie but still fails noninferiority (`+0.066%`, 2/5, upper `1.00255`).
This materially reduces the high-dimensional gap without establishing external
superiority. Horizon-specific gates, output bias, and extended budgets are
negative follow-ups; see
[`channelwise residual gate`](electricity_d32_channelwise_residual_gate_20261006.md).
On the symmetric Traffic bridge, the same gate improves both active and null
but leaves active/null negative (`+0.390%`), so it is not a recovery of the
Traffic relation effect.

## Shared Statistic Gate Boundary

To remove target-channel free parameters, a rank-eight shared MLP was applied
to every channel's local normalized-window statistics (last value, mean,
standard deviation, and endpoint trend). It is channel-count agnostic and
zero-initialized, invokes no additional Mamba scan, and is distinct from the
rejected channel-specific gate extensions. On registered Electricity seed
`6047`, it activated but reached `8.266289` MSE, `+5.977%` worse than the
channelwise gate (`7.800090`) and `+2.635%` worse than the scalar gate
(`8.054045`). This branch is stopped before any external or test comparison.
