# MambaACF Forecast Relation-View Screen

> Exploratory, validation-only, non-confirmatory readout. The four arms are paired by dataset, H96, and seed.

- Completed rows: `24/24`; paired coordinates: `6/6`.
- The primary metric is the best pure-forecast `val_loss` (reported as validation MSE) from `loss_logs.json`; no test metrics are read.

## Per-coordinate estimands

| Dataset | Seed | Encoder null | Encoder active | Encoder active-null | Forecast null | Forecast active | Forecast active-null | Difference-in-differences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.668214 | 0.667091 | -0.001122 | 0.668113 | 0.670240 | 0.002126 | 0.003249 |
| ETTh1 | 2022 | 0.662119 | 0.663207 | 0.001088 | 0.662426 | 0.657991 | -0.004434 | -0.005522 |
| ETTh1 | 2023 | 0.679508 | 0.677982 | -0.001526 | 0.679897 | 0.673181 | -0.006716 | -0.005190 |
| traffic | 2021 | 0.406797 | 0.404058 | -0.002739 | 0.408665 | 0.410797 | 0.002131 | 0.004871 |
| traffic | 2022 | 0.412167 | 0.406548 | -0.005619 | 0.414253 | 0.415339 | 0.001086 | 0.006705 |
| traffic | 2023 | 0.409630 | 0.407039 | -0.002591 | 0.411926 | 0.419066 | 0.007140 | 0.009731 |

Negative active-null deltas favor adding the relation objective within that representation view. A negative difference-in-differences favors the forecast route relative to the encoder route; all values are descriptive.

## Paired summaries

| Estimand | n | Mean delta | SD | Wins lower | Two-sided p |
|---|---:|---:|---:|---:|---:|
| Encoder active - null | 6 | -0.002085 | 0.002212 | 5 | 0.0691 |
| Forecast active - null | 6 | 0.000222 | 0.005014 | 2 | 0.9178 |
| Forecast contrast - encoder contrast | 6 | 0.002307 | 0.006315 | 2 | 0.4119 |

## Relation diagnostics

| Arm | n | AutoCon loss | Candidate count | Contrastive-active anchors | Degenerate positives | Sign accuracy |
|---|---:|---:|---:|---:|---:|---:|
| encoder_null | 6 | 1.284135 | 4.783034 | 0.828414 | 0.085895 | 0.915835 |
| encoder_active | 6 | 0.621267 | 4.775897 | 0.828503 | 0.085710 | 0.910978 |
| forecast_null | 6 | 1.303225 | 4.783034 | 0.828414 | 0.085895 | 0.765613 |
| forecast_active | 6 | 1.182582 | 4.772023 | 0.828781 | 0.085572 | 0.789221 |

## Boundaries

- This readout uses validation trajectories only; it does not produce or inspect test metrics.
- The forecast relation view is a window-level cross-channel objective because the current global loss averages the forecast steps; it is not a stepwise temporal relation objective.
- Zero-lambda arms still calculate the relation path for telemetry and are not an auxiliary-path throughput control.
- A complete screen remains exploratory and cannot select a route or support a publication claim without a separately reviewed frozen replication.
