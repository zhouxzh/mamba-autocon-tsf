# Channel-Patch Q1 Go/No-Go Audit

## Decision

**No-go for a Q1 submission based on the current frozen one-scan MambaTSv3
Channel-Patch program.** The mechanism is real and reproducible on several
low-label traffic-network transfers, but the evidence does not support the
cross-domain, strong-baseline, deployment, and test-governance claims needed
for a high-level journal manuscript.

## Requirement Audit

| Required property | Current evidence | Decision |
| --- | --- | --- |
| One-scan deployed inference without AutoCon loss | PEMS03 deployment audit records one scan and no relation loss | Proven for the audited fixed-shape path |
| Channel-count-independent shared patch encoder | Shared scan, source-excluded PEMS03 replication, and independent-machine replication | Proven as an implementation and traffic-transfer property |
| Train-only ACF self-supervised representation mechanism | Positive source representation geometry and low-label PEMS/METR/PEMS-BAY results | Proven, but domain-bounded |
| 10% and 25% low-label noninferiority | Strong on several traffic targets; Weather 25% calibration is only -0.057%, 2/3 | Not universal |
| Drift, missing-sensor, and noise robustness | PEMS and calibrated Weather validation stress matrices are favorable | Validation-only and not cross-domain universal |
| Strong external-baseline competitiveness | PEMS03 passes matched PatchTST/iTransformer/TimeMixer comparisons; calibrated Weather fails TimeMixer | Not universal |
| High-dimensional Electricity/Traffic transfer | Ordinary and all tested variants fail against forecast calibration or PatchTST | Directly contradicted |
| Clean held-out governance | Historical public-test access caveat and Weather test reversal remain | Not sufficient |
| External latency advantage | Fixed-shape CUDA Graph helps one PEMS batch shape, but no external universal advantage | Not sufficient |

## High-Dimensional Falsification Set

The current Traffic d32/24 bridge has now tested the following changes under
validation-only, target-prefix-isolated, frozen-core protocols. None crossed
the registered `-0.25%` expansion threshold.

| Mechanism | Result | Interpretation |
| --- | ---: | --- |
| Per-channel full trajectory Channel-Patch | +0.320% vs forecast | Base high-dimensional mechanism fails |
| ACF-strength channel aggregation | +0.023% vs uniform Channel-Patch | Reliability weighting does not help |
| Source-only Channel-Patch | -0.043% vs source-and-target | Target loss placement is not the main cause |
| Final causal patch similarity | -0.014% vs full trajectory | Historical state aggregation is not the main cause |
| Four ACF-grouped patch states | +0.806% vs forecast | Heterogeneous-channel grouping fails |
| ACF descriptor in patch encoder | +0.135% vs forecast | Reduces Channel-Patch regression but remains unfavorable |
| Target-tuned descriptor scalar | +1.345% vs forecast | Target scalar adaptation does not repair descriptor transfer |

All artifacts retain train/validation separation, no test metrics, guarded GPU
events, and source/configuration provenance. The machine-readable aggregate is
`artifacts/paper_final/channel_patch_mechanism_evidence_20261006/audit.json`.

## Consequence

The defensible manuscript claim is currently narrower: train-only ACF
Channel-Patch representation pretraining can improve low-label transfer and
validation robustness for selected traffic-network domains under an audited
one-scan deployment path. It cannot be represented as a general high-dimensional
time-series solution or a Q1-ready cross-domain method.

Further work should not tune another loss coefficient, patch similarity, or
small target gate under this same protocol. A new Q1 attempt requires a new
research scope or a materially different forecasting architecture and a fresh
pre-registered evaluation design; those choices exceed a local continuation of
the frozen MambaTSv3 Channel-Patch mechanism.
