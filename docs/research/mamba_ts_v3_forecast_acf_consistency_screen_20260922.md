# MambaTS v3 Forecast ACF Consistency Screen (2026-09-22)

## Question

Can a parameter-free, training-only loss that matches the predicted temporal
ACF to a signed ACF prior fitted on the training split improve MambaTS v3 with
AutoCon? This is deliberately different from the historical representation
relation loss: gradients flow directly through the normalized forecast time
series, while inference has no extra module, parameter, or branch.

## Protocol and integrity

- Four datasets (ETTh1, ETTm1, Electricity, Traffic), H96, three seeds,
  MambaTS v3 AutoCon, and an exact active/null architecture match.
- Both arms use `AutoCon_lambda=0.003`; the active arm additionally uses the
  direct forecast ACF weight `0.01`, two training-prior selected lags, and
  minimum prior magnitude `0.10`.
- 24/24 validation-only jobs completed. Test execution and test metrics were
  disabled.
- An initial concurrent-run attempt produced duplicate manifest rows and
  overwritten checkpoint hashes. It is retained outside this result as an
  invalid audit artifact and is not read here. The clean rerun used a separate
  output root and one runner. Both phase validators report 12 unique jobs,
  matching artifact hashes, and valid provenance.

Plan: [`mamba_ts_v3_forecast_acf_consistency_screen_20260922.json`](./mamba_ts_v3_forecast_acf_consistency_screen_20260922.json)

Readout: [`readout.md`](../../artifacts/exploratory/mamba_ts_v3_forecast_acf_consistency_screen_clean_rerun_20260922/readout/readout.md)

## Results

| Dataset | Null MSE | Active MSE | Active - null | Wins / 3 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.710068 | 0.709845 | `-0.031%` | 2 |
| ETTm1 | 0.395674 | 0.395353 | `-0.082%` | 3 |
| Electricity | 0.159689 | 0.159770 | `+0.051%` | 1 |
| Traffic | 0.371510 | 0.371618 | `+0.029%` | 1 |

Negative changes favor the active objective. Effects are very small, and the
two high-dimensional datasets do not improve consistently. The candidate
therefore fails the precondition for a broader horizon or baseline comparison.

## Decision

Keep the implementation as a reproducible negative ablation; do not tune its
weight, read test metrics, or place it on the paper mainline. It establishes
that merely making the ACF auxiliary loss forecast-connected is insufficient.
Future candidates must couple a train-only lag prior to a selective forecast
mechanism rather than globally regularizing the entire predicted trajectory.

