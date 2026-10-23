# MambaACF Relation-Coverage Screen

Status: **exploratory; not confirmatory**.

All arms use raw train-only ACF sidecars, the variable-token MambaACF AutoCon alias, memory, global hard-selected relation supervision, full-rank heads, the same H96 matrix, and lambda 0.03. They differ only in realized lag-pair policy or global active-anchor normalization. The readout is provenance-snapshot exploratory evidence, not a current-tree retraining or confirmatory result.

| Contrast | Metric | n | Mean delta | Relative delta | Wins lower | Two-sided p |
|---|---|---:|---:|---:|---:|---:|
| balanced - reference | MSE | 6 | 0.00110852 | 0.239% | 2 | 0.3574 |
| balanced - reference | MAE | 6 | 0.000948173 | 0.249% | 2 | 0.6469 |
| balanced - reference | Elapsed seconds | 6 | -0.1915 | -0.397% | 3 | 0.8519 |
| active norm - reference | MSE | 6 | 6.16116e-05 | 0.013% | 2 | 0.2501 |
| active norm - reference | MAE | 6 | 6.72492e-05 | 0.018% | 2 | 0.1587 |
| active norm - reference | Elapsed seconds | 6 | 1.08383 | 2.249% | 2 | 0.5597 |

## Realized Lag Exposure

Counts are accepted disjoint pairs aggregated over epochs; exposure is each lag's share. Balance span is max exposure minus min exposure for the run, where lower is more even.

| Arm | Dataset | Seed | Lag | Pair count | Exposure | Aggregate span | Mean epoch span |
|---|---|---:|---:|---:|---:|---:|---:|
| active_norm | ETTh1 | 2021 | 96 | 13641 | 50.475% | 0.951% | 1.002% |
| active_norm | ETTh1 | 2021 | 120 | 13384 | 49.525% | 0.951% | 1.002% |
| active_norm | ETTh1 | 2022 | 96 | 9651 | 49.938% | 0.124% | 0.994% |
| active_norm | ETTh1 | 2022 | 120 | 9675 | 50.062% | 0.124% | 0.994% |
| active_norm | ETTh1 | 2023 | 96 | 9644 | 49.848% | 0.305% | 1.266% |
| active_norm | ETTh1 | 2023 | 120 | 9703 | 50.152% | 0.305% | 1.266% |
| active_norm | traffic | 2021 | 96 | 19144 | 50.440% | 0.880% | 1.716% |
| active_norm | traffic | 2021 | 120 | 18810 | 49.560% | 0.880% | 1.716% |
| active_norm | traffic | 2022 | 96 | 18975 | 50.069% | 0.137% | 1.050% |
| active_norm | traffic | 2022 | 120 | 18923 | 49.931% | 0.137% | 1.050% |
| active_norm | traffic | 2023 | 96 | 19026 | 50.165% | 0.330% | 1.057% |
| active_norm | traffic | 2023 | 120 | 18901 | 49.835% | 0.330% | 1.057% |
| balanced | ETTh1 | 2021 | 96 | 9643 | 49.933% | 0.135% | 0.207% |
| balanced | ETTh1 | 2021 | 120 | 9669 | 50.067% | 0.135% | 0.207% |
| balanced | ETTh1 | 2022 | 96 | 11601 | 50.013% | 0.026% | 0.060% |
| balanced | ETTh1 | 2022 | 120 | 11595 | 49.987% | 0.026% | 0.060% |
| balanced | ETTh1 | 2023 | 96 | 11596 | 49.938% | 0.125% | 0.159% |
| balanced | ETTh1 | 2023 | 120 | 11625 | 50.062% | 0.125% | 0.159% |
| balanced | traffic | 2021 | 96 | 18930 | 50.001% | 0.003% | 0.134% |
| balanced | traffic | 2021 | 120 | 18929 | 49.999% | 0.003% | 0.134% |
| balanced | traffic | 2022 | 96 | 18956 | 50.000% | 0.000% | 0.137% |
| balanced | traffic | 2022 | 120 | 18956 | 50.000% | 0.000% | 0.137% |
| balanced | traffic | 2023 | 96 | 18961 | 50.005% | 0.011% | 0.179% |
| balanced | traffic | 2023 | 120 | 18957 | 49.995% | 0.011% | 0.179% |
| reference | ETTh1 | 2021 | 96 | 13641 | 50.475% | 0.951% | 1.002% |
| reference | ETTh1 | 2021 | 120 | 13384 | 49.525% | 0.951% | 1.002% |
| reference | ETTh1 | 2022 | 96 | 9651 | 49.938% | 0.124% | 0.994% |
| reference | ETTh1 | 2022 | 120 | 9675 | 50.062% | 0.124% | 0.994% |
| reference | ETTh1 | 2023 | 96 | 9644 | 49.848% | 0.305% | 1.266% |
| reference | ETTh1 | 2023 | 120 | 9703 | 50.152% | 0.305% | 1.266% |
| reference | traffic | 2021 | 96 | 19144 | 50.440% | 0.880% | 1.716% |
| reference | traffic | 2021 | 120 | 18810 | 49.560% | 0.880% | 1.716% |
| reference | traffic | 2022 | 96 | 18975 | 50.069% | 0.137% | 1.050% |
| reference | traffic | 2022 | 120 | 18923 | 49.931% | 0.137% | 1.050% |
| reference | traffic | 2023 | 96 | 19026 | 50.165% | 0.330% | 1.057% |
| reference | traffic | 2023 | 120 | 18901 | 49.835% | 0.330% | 1.057% |

## Relation Scale

| Arm | AutoCon loss mean | Global active-anchor fraction | Total epochs | Epochs ran (min-max) |
|---|---:|---:|---:|---:|
| reference | 0.625292 | 91.447% | 41 | 5--8 |
| balanced | 0.62708 | 91.423% | 41 | 5--8 |
| active_norm | 0.626307 | 91.447% | 41 | 5--8 |

## Boundaries

- Balanced pairing is a least-exposed-lag greedy policy over feasible disjoint pairs, not a strict stratified sampler.
- The six paired coordinates and eight-epoch budget support descriptive diagnostics only; p-values are unadjusted two-sided paired t-tests, not confirmatory evidence.
- Elapsed time includes the full training job and should not be read as a hardware-independent speed claim.
