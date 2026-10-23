# MambaACF ACF Representation Fusion Screen

Status: **exploratory; not confirmatory**.

Both arms use the same raw-source train-only sidecar, MambaACF variable-token AutoCon model, relation target, sampler, seeds, and budget. The only registered change is whether the ReZero ACF context adapter is enabled in the representation path.

| Contrast (fusion on - off) | n | Mean delta | Relative delta | Wins lower | Two-sided p |
|---|---:|---:|---:|---:|---:|
| MSE | 6 | -8.78016e-05 | -0.019% | 5 | 0.786 |
| MAE | 6 | -0.000109689 | -0.029% | 5 | 0.3046 |

Negative deltas favor fusion. The result is descriptive and does not establish a general accuracy or speed claim.

| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 3 | -0.00014253 | -0.037% | -0.000161956 | -0.040% |
| traffic | 3 | -3.30731e-05 | -0.006% | -5.74216e-05 | -0.016% |

## Boundaries

- This screen tests representation-path injection only; the forecast branch is unchanged by the ReZero adapter at initialization.
- Raw ACF makes the adapter's signal transform strictly causal at the source level. Residual/trend exploratory screens do not have that guarantee.
- It does not test longer horizons, parameter-matched PatchTST baselines, multi-lag balanced sampling, or end-to-end runtime.
