# Strict Representation Transfer Readout

## Protocol

The deployed forecaster retains one MambaTSv3 scan. Source-domain pretraining
uses train-prefix-only signed forecast-level ACF InfoNCE and next-patch
prediction. During low-label adaptation, all transferred shared-core weights
(`patch_embedding`, patch positions, Mamba encoder, final normalization, and
representation projections) stay frozen. Only the target forecast head,
NLinear primary head, and residual scale are trainable. AutoCon loss and its
representation return are absent from deployment inference.

Every result below is validation-only and comes from fresh active/null paired
runs. No row in these matrices accesses a target test partition.

## Positive Matrices

| Target | Horizon | Labels | Seeds | Active vs exact-null validation MSE | All seeds active wins |
| --- | ---: | ---: | --- | ---: | ---: |
| PEMS-BAY | 192 | 10% | 5800--5802 | -1.356% | 3/3 |
| PEMS-BAY | 192 | 25% | 5810--5812 | -1.736% | 3/3 |
| METR-LA | 192 | 10% | 5820--5822 | -0.377% | 3/3 |
| Weather | 192 | 10% | 5830--5832 | -0.317% | 3/3 |

The PEMS-BAY 10% checkpoint matrix also retained active wins for all three
seeds under late validation shift, 10% and 30% missing channels, and 0.05 and
0.10 signal-relative noise. Mean effects range from -0.957% to -1.355%.

Inference from the locked PEMS-BAY active checkpoint is prediction-only: one
Mamba scan, no returned AutoCon representation, no relation loss, 0.321ms
median latency at batch size four, and 18.49MB peak allocated memory.

## Registered Post-Validation Test Read

A later registered read-only checkpoint evaluation produced the following
protocol-specific test results. Checkpoints had already been locked by their
validation runs; the reader performed no training, selection, or mutation.

| Target | Horizon | Labels | Active vs exact-null test MSE | Active wins |
| --- | ---: | ---: | ---: | ---: |
| PEMS-BAY | 192 | 10% | -1.196% | 3/3 |
| PEMS-BAY | 192 | 25% | -1.519% | 3/3 |
| METR-LA | 192 | 10% | -0.286% | 3/3 |
| Weather | 192 | 10% | -0.233% | 3/3 |

These values are not described as globally pristine held-out evidence because
the shared workspace had accessed the public dataset test partitions under
other historical protocols. Their valid scope is a transparent, registered,
read-only post-validation comparison of the locked strict-transfer
checkpoints.

## Strong Baseline Boundary

On a same-seed PEMS-BAY H192 25% validation comparison (seeds 5810--5812),
the pinned official PatchTST implementation achieved mean MSE 2.34379 while
the strict frozen-core active model achieved 2.43341. The active model was
therefore worse by +3.824% in all three seeds. This is a retained external
baseline failure: the current strict transfer method has a reproducible causal
gain versus exact-null, but it is not yet competitive with PatchTST at this
coordinate and does not meet a credible SCI-Q1 accuracy bar.

A target-only zero-gated channel adapter and a shared nonlinear temporal MLP
decoder were both explored on validation only. They preserved the active/null
gain but did not close the PatchTST gap. No test read was opened for either
adapter probe.

## Scope And Negative Boundary

These results support the specific claim that train-prefix ACF pretraining can
improve frozen shared-core transfer on compatible traffic-network source/target
pools under low label budgets. They do not support a universal claim.

Electricity and Traffic remain retained negative domains for the current
channelwise forecast-InfoNCE recipe. Two source/pairing mechanisms and a
pooled-patch alternative did not pass their validation gates; the pooled patch
view also exhibited representation collapse until an anti-collapse regularizer
was added, and that regularizer corrected representation spread without a
Traffic forecast gain. See
`mambatsv3_acf_infonce_highdim_mechanism_negative_readout_20261003.md` and
`mambatsv3_pooled_patch_variance_probe_20261003.md`.

## Remaining Evidence Before Any Test Or Submission Claim

1. Register a separate held-out test protocol before accessing the new strict
   transfer checkpoints' test rows.
2. Repeat the strict frozen-core protocol on at least one additional compatible
   target pool and on an independently provisioned machine.
3. Compare against locked external baselines under the same strict low-label
   split, then complete human claim review and venue-specific checks.
