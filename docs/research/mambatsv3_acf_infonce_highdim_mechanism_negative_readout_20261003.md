# High-Dimensional Mechanism Screen Readout

## Scope

This record retains two validation-only tests of plausible explanations for
negative transfer on Electricity and Traffic at H96 with 25% target labels.
Neither screen opened target test rows. Each active arm was compared with an
exact-null pretraining arm under the same chronological split, initialization,
source budget, prediction-head transfer, and target fine-tuning recipe.

## Strict Positive-Label Coverage

The original `channel_coverage` sampling policy did not restrict every
channelwise InfoNCE numerator to the same support-qualified lags. The first
candidate made that restriction explicit.

| Target | Active validation MSE | Exact-null MSE | Active vs null | Seed wins |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 8.08752 | 7.91613 | +2.165% | 0/3 |
| Traffic | 0.81877 | 0.81732 | +0.178% | 0/3 |

Audit: `artifacts/exploratory/mambatsv3_acf_infonce_coverage_label_candidate_20261003/audit.json`.

## Source-Compatible Single-Source Pretraining

The 25%-label target train-prefix rankings had one dominant source: Traffic
for Electricity (0.986; second source 0.200) and Electricity for Traffic
(0.940; second source 0.059). The second candidate retained only that source
and used 192 windows, equal to the previous 64 windows across three sources
per epoch. Both active and null used this same source and budget.

| Target | Active validation MSE | Exact-null MSE | Active vs null | Seed wins |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 7.93774 | 7.80711 | +1.673% | 0/3 |
| Traffic | 0.82100 | 0.81917 | +0.223% | 0/3 |

Audit: `artifacts/exploratory/mambatsv3_acf_infonce_similarity_source_candidate_20261003/audit.json`.

## Interpretation Boundary

The two failures rule out neither all cross-domain pretraining nor all
autocorrelation-based representations. They do show that the current signed
forecast-level channelwise InfoNCE objective is not suitable as a universal
high-dimensional transfer treatment, even after removing rare-positive labels
and strongly mismatched source domains. The held-out PEMS-BAY, METR-LA, and
Weather confirmations therefore remain conditional results, not evidence for a
universal SCI-Q1-level claim. No further held-out tests may be run from either
screen.

All 36 guarded stages across the two screens completed without a thermal or fan
trigger. The highest recorded GPU temperature was 55C and the highest recorded
fan speed was 32%, below the 78C and 85% limits.
