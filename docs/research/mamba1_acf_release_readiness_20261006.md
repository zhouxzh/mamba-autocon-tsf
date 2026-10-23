# Mamba-1 ACF Release Readiness Audit

This audit maps the stated research objective to current local evidence. It is
not a journal acceptance prediction and does not mark the work submission-ready.

| Objective requirement | Evidence | Status |
| --- | --- | --- |
| One-scan frozen MambaTSv3 backbone | Runtime hooks: one encoder call per prediction; Mamba-1 batch-four latency 0.333/0.315/0.314 ms on PEMS-BAY/METR-LA/Weather | Verified on named shapes |
| Channel-count-agnostic shared patch encoder | Source domains with 21, 207, 321, 325, 307, 862, 883 channels share only patch/position/encoder/norm/representation tensors; target channel embeddings are excluded | Verified by transfer contract |
| Train-only ACF SSL pretraining and few-shot transfer | Frozen test active/null controls: PEMS-BAY -2.449%, METR-LA -0.430%, Weather -0.397%; 3/3 wins each | Verified, scoped to three targets |
| Low-label external comparisons | Frozen test candidate vs official PatchTST: -1.653%, -5.775%, -3.024%; post-release iTransformer: -7.588%, -9.575%, -11.286%; TimeMixer: -3.184%, -4.169%, -2.127% | Verified, three external baselines |
| Independent TimesNet extension | PEMS-BAY H192/10% independent five-seed validation: -10.366%, 5/5 wins, upper 0.91665 | Verified, validation-only official-code extension |
| Independent METR-LA replication | H192/10% five fresh seeds: active/null -0.556% (5/5), active/official PatchTST -4.088% (5/5), upper 0.96354 | Verified, validation-only official-code extension |
| Channel-patch representation ACF SSL | Fresh three-seed PEMS-BAY/METR-LA H192 10% replication: -1.702%/-1.070% versus forecast-only, 3/3 each; all six stresses externally noninferior to same-seed official PatchTST | Verified, validation-only mechanism replication |
| Channel-patch Weather external replication | Five Weather H192/10% seeds: -0.148% versus forecast-only (4/5), -2.488% versus official PatchTST (5/5); all six stresses externally noninferior | Verified, validation-only external-domain replication |
| Channel-patch calibrated Weather replication | Fresh Weather H192/10% target-prefix calibration: -0.280% versus same-calibration forecast, -2.381% versus new same-seed PatchTST, -15.643% versus iTransformer, 3/3 each; all six stresses externally noninferior; TimeMixer noninferiority fails (+0.034%, 1/3) | Verified, validation-only post-registration protocol; does not supersede Weather frozen-test reversal |
| Channel-patch calibrated Weather 25% replication | Fresh H192/25% target-prefix calibration: -0.057% versus forecast calibration, 2/3 wins, upper 0.99998 | Verified label-budget boundary; promotion gate fails, so no external expansion |
| Channel-patch 25% labels | Fresh PEMS-BAY/METR-LA H192 25% seeds: -1.820%/-0.951% versus forecast-only, 3/3 each; all six official PatchTST stress cells noninferior | Verified, validation-only label-budget replication |
| Channel-patch H336 extension | Fresh PEMS-BAY H336/10% seeds: -1.167% versus forecast-only, -2.568% versus official PatchTST, 3/3 each; all six stress cells noninferior | Verified, validation-only long-horizon replication |
| Channel-patch target-excluded PEMS03 transfer | Fresh PEMS03 H192/10% seeds with PEMS03 excluded from source and initialization: -0.333% versus forecast-only, -3.980% versus locked official PatchTST, 3/3 each; all six declared PatchTST stress cells noninferior | Verified, validation-only external target replication |
| Channel-patch target-excluded PEMS03 25% transfer | Fresh PEMS03 H192/25% seeds under the same exclusion: -0.455% versus forecast-only, -14.253% versus locked official PatchTST, -17.395% versus pinned official iTransformer, 3/3 each; all six declared PatchTST stress cells noninferior | Verified, validation-only external label-budget replication |
| PEMS03 25% official TimeMixer precision expansion | Original five seeds and update-matched five-seed audit fail noninferiority. Final pre-registered 15-seed batch/update-matched expansion: -3.350%, 7/15 wins, upper 0.99978 | Verified external noninferiority under matched batch/update geometry; capped expansion complete |
| Channel-patch frozen test extension | PEMS-BAY/METR-LA retain forecast-only gains; Weather reverses by +0.277% vs forecast-only despite 5/5 PatchTST wins | Historical-access-caveated read-only extension; causal claim narrows |
| Independent-machine channel-patch replication | Strict PEMS03 H192/25% replication on verified RTX 3060 Ti host with Torch 2.6/CUDA 12.6/Mamba 2.2.4: -0.406% versus forecast-only, 3/3, upper 0.99662 | Verified independent-host validation-only replication |
| d32/24 channel-patch bridge | Three fresh Electricity seeds reverse the one-seed signal: +0.253% vs forecast-only, 0/3, upper 1.00371 | Stopped negative high-dimensional boundary |
| High-dimensional channel-patch extension | Electricity H96 strong calibration: -0.653% versus forecast active, 3/3; still +8.590% behind same-seed official PatchTST | Verified causal gain, external gate fails |
| High-dimensional target channel gate | d32/24 Electricity: -3.362% vs scalar-gate transfer, 3/3; final ten-seed PatchTST gap +0.274%, 3/10, upper 1.00509 | Strong gap reduction, final external noninferiority failure |
| Shift/missing/noise robustness | Active/null: all registered cells favor active. Independently registered five-seed PEMS-BAY and Weather replications plus the three-seed METR-LA extension all pass six official PatchTST external stress cells | Verified, validation-only, scoped by target |
| Inference has no AutoCon loss/extra scan | Prediction-only hook audit: no representation returned, relation loss false, one Mamba block | Verified |
| Current-source clean interpreter | Separately installed Python/CUDA/Mamba interpreter reproduces PEMS-BAY active/null validation gain with current code/data hashes | Verified same-machine support evidence |

## Negative Boundaries

The method does not presently generalize to all high-dimensional targets.
Electricity and Traffic validation probes remain worse than their exact nulls
under optimized Mamba-1 24/8 source-only transfer. Electricity remains worse
after target-prefix calibration. Those results are valid negative evidence, not
missing cells to be omitted.

A validation-only relation-fraction screen (`0.05/0.10/0.20/0.30`) found that
all active fractions beat their exact null on one fresh paired seed, but no
alternative fraction improved the frozen `0.10` setting on both PEMS-BAY and
METR-LA. The simple hypothesis that AutoCon is merely underweighted is
therefore not supported; see
[`relation fraction screen`](mambatsv3_relation_fraction_screen_20261006.md).
Likewise, weighting forecast InfoNCE positive pairs by raw train-only ACF
magnitude failed to improve the ordinary objective on either PEMS-BAY or
METR-LA; see
[`ACF-confidence probe`](mambatsv3_acf_confidence_weighting_probe_20261006.md).
Filtering the same positives by two-half train-prefix ACF stability also did
not improve the ordinary objective on either target; see
[`ACF-stability probe`](mambatsv3_acf_stability_weighting_probe_20261006.md).
Rotating source-domain optimizer order improves METR-LA but degrades PEMS-BAY;
it is retained only as a supported negative-boundary control, not as the
release default; see
[`source-update probe`](mambatsv3_source_update_policy_probe_20261006.md).
Learned permutation-equivariant channel set-pooling preserves active/null
gains on PEMS-BAY and METR-LA but fails to outperform the ordinary encoder on
PEMS-BAY in a frozen three-seed replication. It remains a high-dimensional
mechanism candidate rather than a new release default; see
[`set-pooling replication`](mambatsv3_channel_set_pooling_replication_20261006.md).
Adding an independently calibrated ACF relation objective directly to the
transferred `repr_` states improves PEMS-BAY but slightly degrades METR-LA,
despite valid relation coverage; it is not promoted beyond its validation
probe. See
[`direct representation probe`](mambatsv3_direct_representation_relation_probe_20261006.md).
Connecting that direct representation through a learned target forecast fusion
also improves PEMS-BAY but slightly degrades METR-LA with demonstrably nonzero
fusion gates; it remains a negative cross-domain mechanism boundary. See
[`deployed fusion probe`](mambatsv3_deployed_representation_fusion_probe_20261006.md).
The broader single-scan v13 family candidate, which adds shared depthwise
multi-scale filters after the same Mamba-1 scan, improved over its v13 null
but failed to reproducibly beat release v3 active on its registered PEMS-BAY
two-seed pilot (one win, one-SE ratio upper `1.00172`). It is stopped before
METR-LA expansion or test reading; see
[`v13 backbone-family screen`](mambats_backbone_family_v13_screen_20261006.md).
Unlike the rejected global representation and pooled-channel objectives, a
channelwise full patch-trajectory ACF InfoNCE objective is positive in an
independent three-seed PEMS-BAY/METR-LA validation replication and all six
validation stresses. It is a promising representation mechanism, but is not
yet part of the frozen test release; see
[`channel-patch representation replication`](mambats_channel_patch_representation_replication_20261006.md).
Its frozen test extension confirms PEMS-BAY/METR-LA but shows a Weather
reversal relative to forecast-only. This is a material cross-domain boundary,
not a result to aggregate away; see
[`channel-patch frozen test extension`](channel_patch_frozen_test_extension_20261006.md).
The Electricity target-channel residual gate is not a shared-method default:
it degrades a PEMS-BAY external gate probe by `+2.485%`. It remains a
target-specific high-dimensional calibration mechanism with a narrow external
gap, not a cross-domain architecture claim.

## Submission Gate

The evidence is strong enough for a constrained manuscript claim about
low-label transfer on PEMS-BAY, METR-LA, and Weather. It is not a complete
SCI-Q1-ready package because:

1. The repository contains historical public test artifacts, so this release
   cannot be presented as a pristine first-open test evaluation.
2. The new release has three official external model families but lacks broader
   official baseline coverage at the same frozen coordinates. Independent-host
   PEMS03 reproduction now exists, but it does not resolve the Weather frozen-
   test reversal or provide independent data.
3. High-dimensional Traffic remains a negative boundary. Electricity now has a
   narrow positive active/null result only when the target calibration prefix
   uses a stronger relation fraction (`0.3`), but it still loses the matched
   PatchTST validation comparator; this is not external SOTA.
4. The eager Mamba-1 candidate is one-scan but not faster than the named
   PatchTST benchmark. The current channel-patch checkpoint is `0.34248 ms`
   versus `0.25438 ms` at batch four. Fixed-shape CUDA Graph device-only replay
   preserves outputs and reaches `0.25824 ms` (`+1.52%` versus PatchTST), but
   does not cover host transfer, dynamic shapes, or end-to-end service latency.
   A complete batch `1/4/8` grid is faster at batch one only (`-40.40%`), near
   parity at batch four, and slower at batch eight; no universal latency claim
   is allowed.
5. The three-domain external robustness evidence is validation-only and uses
   short low-label coordinates. It supports the named PEMS-BAY, METR-LA, and
   Weather protocols, not universal robustness across datasets or horizons.
6. The same-machine clean-interpreter result remains supplementary only. The
   independent-host PEMS03 replication closes the machine-diversity gate for
   that named validation coordinate, not the independent-data or test-governance
   gaps.

7. The PEMS03/25% TimeMixer precision expansion has a favorable mean but fails
   the registered five-seed noninferiority gate. It is therefore a strong
   baseline boundary, not evidence of stable superiority.

8. Strict PEMS03 H192/25% replication completed on verified independent host
   `zhong-Precision-Tower-7810` (RTX 3060 Ti, Torch 2.6/CUDA 12.6, Mamba 2.2.4):
   Channel-Patch improves forecast-only by `-0.406%`, 3/3, upper `0.99662`.
   It is validation-only and does not cure the test-governance limitation.

The new PEMS03 result expands low-label traffic-network coverage, but does not
close any of these gates: it is a registered validation readout and includes
neither an untouched test evaluation nor a six-condition robustness extension.

## Defensible Current Claim

On the named three compatible targets under chronological low-label splits,
the frozen channel-agnostic Mamba-1 encoder pretrained with train-only ACF
forecast-contrastive relations improves over matched predictive-pretraining
controls, remains favorable under declared validation stresses, and outperforms
the locked official PatchTST and post-release official iTransformer baselines
in the recorded test readouts. This claim excludes Electricity, Traffic,
universal SOTA, and universal latency superiority.
