# MambaACF Relation-Target Screen

Status: **exploratory; not confirmatory**.

Both arms use the current sidecar-backed raw train-only prior, variable-token MambaACF AutoCon, global relation loss, the same lag-pair sampler, full-rank head, H96 matrix, lambda 0.03, and eight-epoch budget. The only registered intervention is hard selected-peak support versus soft strong-ACF support at `abs(ACF)>0.5`; selected-lag boost is zero. High-ACF telemetry uses an independent `abs(ACF)>0.1` threshold.

| Contrast (soft05 - hard) | n | Mean delta | Relative delta | 95% CI | Wins lower | Two-sided p |
|---|---:|---:|---:|---|---:|---:|
| MSE | 6 | 0.000687238 | 0.146% | [-0.00010581, 0.00148029] | 2 | 0.1502 |
| MAE | 6 | 0.000613408 | 0.160% | [-8.15307e-05, 0.00130835] | 3 | 0.1442 |
| Elapsed seconds | 6 | -0.315667 | -0.659% | [-1.39149, 0.760161] | 2 | 0.5901 |

## Paired Seeds

| Dataset | Seed | MSE delta | MAE delta | Hard MSE | Soft05 MSE |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | -7.74294e-05 | -0.000161662 | 0.384214 | 0.384137 |
| ETTh1 | 2022 | -0.000447578 | -5.57646e-05 | 0.384084 | 0.383636 |
| ETTh1 | 2023 | 8.57218e-06 | -0.000191028 | 0.384998 | 0.385007 |
| traffic | 2021 | 0.00150706 | 0.000853678 | 0.553604 | 0.555111 |
| traffic | 2022 | 0.00114193 | 0.0016418 | 0.554996 | 0.556138 |
| traffic | 2023 | 0.00199088 | 0.00159342 | 0.552961 | 0.554952 |

## Target-Support Diagnostics

| Arm | AutoCon loss | Active anchors | Positive count / anchor | High-ACF candidates | High-ACF conditional excluded | Total epochs | Epochs (min-max) |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard | 0.626559 | 91.447% | 0.923 | 142.9 | 78.468% | 41 | 5--8 |
| soft05 | 0.729192 | 94.819% | 1.470 | 142.9 | 66.620% | 41 | 5--8 |

## Dataset-Level Gates

| Dataset | MSE delta | Relative MSE | MSE wins | MAE relative delta |
|---|---:|---:|---:|---:|
| ETTh1 | -0.000172145 | -0.045% | 2/3 | -0.033% |
| traffic | 0.00154662 | 0.279% | 0/3 | 0.380% |

## Registered Decision

- Stop rule triggered: **yes**.
- Stop reason: traffic: soft05 has 0/3 lower-MSE seeds (<=1/3).
- Expansion eligible: **no**.
- Expansion gate: ETTh1: mean MSE improvement is 0.045% (<0.5%).
- Expansion gate: traffic: mean MSE improvement is -0.279% (<0.5%).
- Expansion gate: traffic: fewer than 2/3 lower-MSE seeds.
- Expansion gate: diagnostic high-ACF exclusion reduction is 15.099% (<50%).

## Boundaries

- The independent diagnostic threshold prevents `soft_multi_peak` from making its own exclusion rate tautologically zero.
- The soft arm changes target support and relation-loss density together; it is a mechanism/sensitivity screen, not a pure optimization or false-negative proof.
- Pairing is at dataset x seed (`n=6`); p-values are unadjusted descriptive tests and cannot select a model.
- No memory branch, longer horizon, cross-channel relation reduction, or runtime claim is tested.
