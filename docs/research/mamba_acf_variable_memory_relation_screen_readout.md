# MambaACF Variable Memory + Relation Screen Readout

> Exploratory paired screen only. Each dataset has n=3 seeds; no result here is confirmatory evidence or a basis for publication claims.

## Contract and validation

- Plan: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/docs/research/mamba_acf_variable_memory_relation_screen_plan.json`; SHA-256 `5d3da536cbe964cc4590a803e0a5b77d2845b71591c54bc070e5d5e13bee9953`.
- Screen root: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/mamba_acf_variable_memory_relation_screen_20260901`; successful jobs: `18/18`.
- Pairing unit: same dataset, feature setting, horizon, and seed. Delta is treatment minus reference; negative is favorable.
- Metric source: `stream_metrics.json` scaled MSE/MAE. Forecast windows are not treated as independent observations.

### Replayability boundary

This screen was produced before the current `acf_prior.npz` sidecar contract was added. No sidecar is present under the 18 job directories, and their configs do not bind `mamba_acf_prior_artifact` or its SHA-256. The six `variable_plain` runs have memory and relation disabled and are not dependent on this sidecar. The six `variable_memory` and six `variable_memory_relation` runs fitted the train-only prior in process; their checkpoints cannot be replayed from checkpoint plus config alone by the current runner, which requires the persisted prior for static memory or relation supervision. Recomputing those 12 runs requires the original train split and matching immutable source snapshot and should be called a historical re-run, not a sidecar-backed standalone replay.

| Phase | Jobs | Manifest SHA-256 | Provenance ID | Source-tree SHA-256 |
|---|---:|---|---|---|
| `variable_plain_h96_screen` | 6/6 | `aae7aa3a79c3e2a06971c5f8db0f21384ff1577aaf47144ebc2a1b4ab4e552c9` | `9d2fa8b115631e18` | `5c250399d1e9c409a2906248fe84477cd7243250d2ff61fc9310eb8cf5be8be6` |
| `variable_memory_h96_screen` | 6/6 | `e04fc3d926b182ce757c3bfa08ffbd9ab00a35f8f3bef8001b29d3f37e3ddf5b` | `3fb116f04b365754` | `5c250399d1e9c409a2906248fe84477cd7243250d2ff61fc9310eb8cf5be8be6` |
| `variable_memory_relation_h96_screen` | 6/6 | `9c5ece2739528b43ab2400f117972141266bbbcb579db564fe0c07d87cf8e66b` | `77b680d7108144e3` | `5c250399d1e9c409a2906248fe84477cd7243250d2ff61fc9310eb8cf5be8be6` |

## Per-seed paired metrics

| Dataset | Seed | Plain MSE | Memory MSE | Memory-Plain | Memory+Relation MSE | Relation-Memory | Plain MAE | Memory MAE | Memory-Plain | Memory+Relation MAE | Relation-Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.387909 | 0.386251 | -0.001657 | 0.383971 | -0.002280 | 0.413996 | 0.411236 | -0.002760 | 0.410254 | -0.000982 |
| ETTh1 | 2022 | 0.384194 | 0.384213 | 0.000019 | 0.381790 | -0.002423 | 0.407511 | 0.405672 | -0.001839 | 0.404356 | -0.001316 |
| ETTh1 | 2023 | 0.385979 | 0.387093 | 0.001114 | 0.382507 | -0.004586 | 0.410734 | 0.409091 | -0.001643 | 0.402113 | -0.006979 |
| traffic | 2021 | 0.554268 | 0.542044 | -0.012224 | 0.538694 | -0.003350 | 0.361928 | 0.358757 | -0.003171 | 0.353619 | -0.005138 |
| traffic | 2022 | 0.555834 | 0.544567 | -0.011268 | 0.541552 | -0.003015 | 0.361712 | 0.358296 | -0.003415 | 0.352434 | -0.005862 |
| traffic | 2023 | 0.558587 | 0.543999 | -0.014587 | 0.539678 | -0.004322 | 0.360852 | 0.357006 | -0.003845 | 0.353958 | -0.003048 |

## Paired summaries

SD is the sample standard deviation across seed-level deltas. Paired t values and p-values are descriptive only (n=3 per dataset).

| Dataset | Contrast | Metric | n | Reference mean | Treatment mean | Delta mean +/- SD | Wins | Relative change | t | two-sided p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | variable memory minus variable plain | MSE | 3 | 0.386027 | 0.385852 | -0.000175 +/- 0.001396 | 1/3 | -0.045% | -0.2168 | 0.8484 |
| ETTh1 | variable memory minus variable plain | MAE | 3 | 0.410747 | 0.408666 | -0.002081 +/- 0.000597 | 3/3 | -0.507% | -6.0396 | 0.0263 |
| ETTh1 | variable memory + relation minus variable memory | MSE | 3 | 0.385852 | 0.382756 | -0.003096 +/- 0.001292 | 3/3 | -0.802% | -4.1520 | 0.0534 |
| ETTh1 | variable memory + relation minus variable memory | MAE | 3 | 0.408666 | 0.405574 | -0.003092 +/- 0.003370 | 3/3 | -0.757% | -1.5891 | 0.2530 |
| traffic | variable memory minus variable plain | MSE | 3 | 0.556230 | 0.543537 | -0.012693 +/- 0.001709 | 3/3 | -2.282% | -12.8664 | 0.0060 |
| traffic | variable memory minus variable plain | MAE | 3 | 0.361497 | 0.358020 | -0.003477 +/- 0.000341 | 3/3 | -0.962% | -17.6474 | 0.0032 |
| traffic | variable memory + relation minus variable memory | MSE | 3 | 0.543537 | 0.539974 | -0.003562 +/- 0.000679 | 3/3 | -0.655% | -9.0913 | 0.0119 |
| traffic | variable memory + relation minus variable memory | MAE | 3 | 0.358020 | 0.353337 | -0.004683 +/- 0.001461 | 3/3 | -1.308% | -5.5504 | 0.0310 |
| pooled descriptive (not independent) | variable memory minus variable plain | MSE | 6 | 0.471128 | 0.464695 | -0.006434 +/- 0.006997 | 4/6 | -1.366% | -2.2523 | 0.0741 |
| pooled descriptive (not independent) | variable memory minus variable plain | MAE | 6 | 0.386122 | 0.383343 | -0.002779 +/- 0.000880 | 6/6 | -0.720% | -7.7372 | 0.0006 |
| pooled descriptive (not independent) | variable memory + relation minus variable memory | MSE | 6 | 0.464695 | 0.461365 | -0.003329 +/- 0.000957 | 6/6 | -0.716% | -8.5174 | 0.0004 |
| pooled descriptive (not independent) | variable memory + relation minus variable memory | MAE | 6 | 0.383343 | 0.379456 | -0.003887 +/- 0.002481 | 6/6 | -1.014% | -3.8377 | 0.0122 |

## Relation objective coverage

The relation arm was required to use objective v3 and strictly positive global coverage series.

| Dataset | Seed | Objective | Valid min | Candidate min | Candidate-anchor min | Positive-count min | Effective lags |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 2021 | 3 | 0.029527 | 0.163162 | 0.993986 | 0.915330 | 96, 120 |
| ETTh1 | 2022 | 3 | 0.029603 | 0.163474 | 0.994340 | 0.917689 | 96, 120 |
| ETTh1 | 2023 | 3 | 0.029595 | 0.162964 | 0.994222 | 0.917453 | 96, 120 |
| traffic | 2021 | 3 | 0.029755 | 0.140081 | 0.995631 | 0.922407 | 96, 120 |
| traffic | 2022 | 3 | 0.029643 | 0.139676 | 0.996420 | 0.918924 | 96, 120 |
| traffic | 2023 | 3 | 0.029743 | 0.139544 | 0.995937 | 0.922020 | 96, 120 |

## Interpretation boundary

- Three seeds per dataset provide a mechanism-screening readout only. The exact paired Wilcoxon minimum p-value is not used to inflate evidence; only descriptive paired t statistics are shown here.
- The pooled row is a cross-dataset descriptive summary, not six independent replications of one data-generating process.
- The variable branch, train-only ACF memory, and relation objective are jointly confounded in the second contrast; this screen cannot identify a causal contribution of each component.
- A separate, frozen confirmation matrix with validation-only selection and an independent evaluation protocol is required before any generalization or publication claim.
