# MambaACF Horizon Amplitude Calibration Screen

Status: **exploratory; not confirmatory**.

The off/on arms use the same MambaACFAutoCon model, hard global signed ACF relation objective, raw train-only sidecar, data, seeds, and eight-epoch budget. They differ only in the rank-2 horizon amplitude adapter.

## Paired Readout

| Contrast (on - off) | n | Mean delta | Relative delta | Wins lower | Two-sided p |
|---|---:|---:|---:|---:|---:|
| scaled MSE | 6 | -0.000653053 | -0.136% | 5 | 0.0515 |
| scaled MAE | 6 | -0.00025762 | -0.064% | 5 | 0.0168 |
| elapsed seconds | 6 | -2.86017 | -6.291% | 3 | 0.4403 |

Negative accuracy deltas favor calibration-on. The six paired coordinates are descriptive only.

## Dataset Splits

| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 3 | -0.00010432 | -0.027% | -0.000141812 | -0.034% |
| traffic | 3 | -0.00120179 | -0.213% | -0.000373428 | -0.097% |

## Adapter Diagnostics

The checkpoint contract expects six trainable coefficients (`K=3`, rank `2`) on the treatment arm and no calibration tensors on the reference arm. Runtime amplitude logs are null for off and finite for on.

| Arm | Effective memory RMS | Runtime gate mean | Amplitude mean | Amplitude min | Amplitude max |
|---|---:|---:|---:|---:|---:|
| off | 0.160928 | 0.271782 | n/a | n/a | n/a |
| on | 0.171736 | 0.241419 | 1.16672 | 1.02309 | 1.34047 |

## Boundaries

- The adapter is a bounded functional correction to the existing memory path; its amplitude is partly confounded with the existing horizon-specific ACF gate.
- The paired sample is dataset x seed (`n=6`); p-values are unadjusted descriptive summaries and cannot select a method.
- This screen does not test longer horizons, dynamic ACF, alternative relation targets, or an independent replication.
