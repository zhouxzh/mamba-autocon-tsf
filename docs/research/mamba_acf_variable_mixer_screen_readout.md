# MambaACF Variable-Mixer Screen Readout

> Exploratory paired screen only. Each dataset has n=3 seeds; this readout is not confirmatory evidence and must not be merged into the paper's main comparison table.

## Contract and provenance

- Phase: `variable_mixer_screen`; validated jobs: `12/12`.
- Pairing: same dataset, feature setting, horizon, and seed; delta is `MambaACFVariable - MambaACF`.
- Metric: scaled test MSE/MAE from `stream_metrics.json`; forecast windows are not inferential replicates.
- Validation: `valid=true`; provenance ID `c60871ef5382a0f5`.
- Plan SHA-256: `1d9fd21d50e4f01c0ef19ddbfac4de15d1193e5186f96b99be9e7db462cc9f3b`; manifest SHA-256: `d7b78fdbc745b20cce3bdecdc9a7324d6a7bfa705f8a240a1c04ecd0b402c33f`.
- Frozen source-tree SHA-256: `c8af1371adf97c0c70f059763b75da8075502f49b8efd2a4fdfb5b0a416cc16c`; all 12 jobs had matching before/after hashes.
- The registered screen used `mamba_causal_embedding=false`, `mamba_acf_channel_order=input`, and no channel permutation. It therefore does not test strict causal embedding or order invariance; a follow-up must compare input order with a fixed strength order and controlled random permutations.

### Replayability boundary

No `acf_prior.npz` sidecar was emitted by this historical snapshot, and the run configs therefore have no persisted-prior identity fields. This screen is nevertheless forecast-path independent of that sidecar: both the plain and variable arms set `mamba_use_acf_memory=false` and `mamba_acf_relation_loss=false`. The contrast remains a frozen historical-snapshot result and has not been rerun under the current sidecar-enabled runner; this caveat must not be transferred to screens that enable static ACF memory or relation supervision.

## Per-seed paired metrics

| Dataset | Seed | Plain MSE | Variable MSE | Delta MSE | Plain MAE | Variable MAE | Delta MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.392341 | 0.387909 | -0.004433 | 0.413267 | 0.413996 | 0.000729 |
| ETTh1 | 2022 | 0.386744 | 0.384194 | -0.002550 | 0.409355 | 0.407511 | -0.001844 |
| ETTh1 | 2023 | 0.389907 | 0.385979 | -0.003928 | 0.412321 | 0.410734 | -0.001587 |
| traffic | 2021 | 0.574395 | 0.554267 | -0.020128 | 0.391007 | 0.361928 | -0.029079 |
| traffic | 2022 | 0.569977 | 0.555834 | -0.014143 | 0.390509 | 0.361712 | -0.028797 |
| traffic | 2023 | 0.566952 | 0.558587 | -0.008365 | 0.386598 | 0.360849 | -0.025749 |

## Paired summaries

Negative delta is favorable. SD is the sample standard deviation across seed-level deltas.

| Contrast | n | Plain MSE mean | Variable MSE mean | Delta MSE mean +/- SD | MSE wins | MSE relative change | Plain MAE mean | Variable MAE mean | Delta MAE mean +/- SD | MAE wins | MAE relative change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 3 | 0.389664 | 0.386027 | -0.003637 +/- 0.000975 | 3/3 | -0.933% | 0.411648 | 0.410747 | -0.000901 +/- 0.001417 | 2/3 | -0.219% |
| traffic | 3 | 0.570441 | 0.556229 | -0.014212 +/- 0.005882 | 3/3 | -2.491% | 0.389371 | 0.361496 | -0.027875 +/- 0.001846 | 3/3 | -7.159% |
| pooled descriptive (not independent) | 6 | 0.480053 | 0.471128 | -0.008924 +/- 0.006912 | 6/6 | -1.859% | 0.400510 | 0.386122 | -0.014388 +/- 0.014847 | 5/6 | -3.592% |

## Runtime observation

| Contrast | Plain mean seconds | Variable mean seconds | Overhead |
|---|---:|---:|---:|
| ETTh1 | 19.973 | 26.182 | 31.08% |
| traffic | 54.854 | 62.583 | 14.09% |

## Interpretation boundary

- ETTh1 and Traffic each have only three paired seeds. Means and SDs describe this screen; they do not establish a stable gain or generalization.
- The pooled six-pair row is a descriptive cross-dataset summary, not six independent replications of one data-generating process.
- The variable arm adds a shared cross-variable token branch to the same executable `MambaACF`; it does not change the ACF memory or relation-loss settings in this screen.
- Because the branch consumes the CSV channel order, the observed gain may depend on that ordering. No random-permutation or order-robustness result is available yet.
- A future confirmatory run must freeze the arm from validation evidence, pre-register the matrix, and replicate it under an independent evaluation protocol.
