# MambaACF Relation Null Control

Status: **exploratory; not confirmatory**.

The two arms use the same variable-token MambaACF AutoCon model, train-only ACF sidecars, relation target, lag-pair sampler, data, seeds, and budget. They differ only in relation-loss weight (`lambda=0.0` versus `0.03`).

## Paired Readout

| Contrast (0.03 - 0.0) | n | Mean delta | Relative delta | Wins lower | Two-sided p |
|---|---:|---:|---:|---:|---:|
| MSE | 6 | -0.00340365 | -0.732% | 4 | 0.04544 |
| MAE | 6 | -0.00313963 | -0.821% | 4 | 0.09801 |

Negative deltas favor the non-zero relation loss. The result is descriptive and does not establish a general accuracy or speed claim.

## Dataset Splits

| Dataset | n | MSE delta | MSE relative | MAE delta | MAE relative |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 3 | -0.000796626 | -0.208% | -7.01057e-05 | -0.017% |
| traffic | 3 | -0.00601068 | -1.101% | -0.00620915 | -1.727% |

## Objective Diagnostics

The active-anchor fraction and relation-loss scale must be reported because global inactive anchors are currently averaged as zeros. These diagnostics are not evidence of representation quality by themselves.

| Arm | AutoCon loss mean | Global valid fraction | Active-anchor fraction | Positive count / anchor | High-ACF candidates | High-ACF conditional excluded | Sign accuracy | Sign hinge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lambda=0.0 | 1.27986 | 2.977% | 91.447% | 0.923 | 125.9 | 76.063% | 91.610% | 0.0000 |
| lambda=0.03 | 0.625948 | 2.977% | 91.447% | 0.923 | 125.9 | 76.063% | 91.247% | 0.0011 |

## Boundaries

- This control isolates the relation-loss weight from the AutoCon representation head and sampler only for this registered H96 matrix.
- It does not test raw-source strict causality, longer horizons, alternative target modes, per-channel relation modeling, or runtime advantage.
- The current residual ACF transform uses a centered window smoother; it is not an end-to-end strict-causal transform.
