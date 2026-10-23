# MambaACF Low-Rank Head Screen

Status: **exploratory; not confirmatory**.

The two arms share the MambaACF variable-token AutoCon model, raw train-only ACF sidecar, memory branch, global hard relation target, sampler, data, seeds, and eight-epoch budget. They differ only in whether both temporal forecast projections are full-rank or rank-16 factorized. This is provenance-snapshot exploratory evidence, not a current-tree retraining or standalone replay.

Negative accuracy deltas favor rank-16. Negative elapsed-time deltas mean the factorized arm finished sooner. The paired sample is six dataset/seed coordinates and is descriptive only; all p-values below are unadjusted two-sided paired t-tests.

## Paired Readout

| Contrast (rank16 - full) | n | Mean delta | Relative delta | Wins lower | Two-sided p |
|---|---:|---:|---:|---:|---:|
| MSE | 6 | 0.00506838 | 1.095% | 0 | 0.02188 |
| MAE | 6 | 0.000758388 | 0.199% | 3 | 0.4601 |
| Elapsed seconds | 6 | -0.4715 | -0.977% | 3 | 0.6128 |

## Dataset Splits

| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative | Elapsed delta | Elapsed relative |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 3 | 0.00173678 | 0.455% | 0.00186116 | 0.460% | -1.502 | -5.832% |
| traffic | 3 | 0.00839998 | 1.544% | -0.000344386 | -0.097% | 0.559 | 0.790% |

## Head Contract

The analyzer checks the learned checkpoint tensors, not only the config flag. The reference contains two 96x96 projections; the treatment contains two 96x16x96 factorized projections. AutoCon receives the same representation tensor before these forecast heads.

| Arm | Mean head parameter count | Factorization |
|---|---:|---|
| full | 18624.0 | no |
| rank16 | 6336.0 | yes |

## Training Run Length

Early stopping makes elapsed time partly a training-budget outcome rather than a pure forward-speed measure.

| Arm | Total epochs | Epochs ran (min-max) |
|---|---:|---:|
| full | 41 | 5--8 |
| rank16 | 39 | 5--8 |

## Relation Diagnostics

| Arm | AutoCon loss mean | Global valid fraction | Active-anchor fraction | Positive count / anchor | Sign accuracy |
|---|---:|---:|---:|---:|---:|
| full | 0.625291 | 2.977% | 91.447% | 0.923 | 91.243% |
| rank16 | 0.600345 | 2.976% | 91.432% | 0.923 | 91.281% |

## Boundaries

- This H96, three-seed screen does not establish a universal accuracy or speed advantage.
- Elapsed time is a short-budget training readout and is not an end-to-end throughput benchmark with repeated synchronized resource measurements.
- Non-inferiority, longer horizons, parameter-matched external baselines, and independent seeds require a separate registered experiment.
