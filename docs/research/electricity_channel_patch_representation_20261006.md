# Electricity Channel-Patch Representation Boundary

Status: independently replicated validation-only mechanism result with a
same-seed external-baseline failure. No test rows were read.

## Protocol

This extension starts from the only positive high-dimensional frozen-transfer
protocol available before the experiment: Traffic source next-patch and ACF
pretraining at relation fraction `0.1`, plus the earliest Electricity 25%
label prefix as target calibration at relation fraction `0.3`. The target
Mamba core remains frozen. The new candidate adds only the training-only
channelwise patch-trajectory ACF InfoNCE term at fraction `0.05`; it keeps one
Mamba-1 scan and makes no inference change.

The independent replication uses Electricity H96 fresh seeds `6033--6035`.
Each channel-patch training epoch had active relation anchors equal to `1.0`.

## Causal Result

| Model | Validation MSE | Channel-patch vs forecast active | Wins | One-SE ratio upper |
| --- | ---: | ---: | ---: | ---: |
| Forecast-level strong active | 8.201674 | -- | -- | -- |
| Channel-patch active | 8.148099 | -0.653% | 3/3 | 0.99481 |

The representation term therefore gives a repeatable improvement over the
already positive strong-calibration ACF transfer baseline. This is stronger
than earlier high-dimensional channel-summary, pooling, target-decoder, and
direct representation-fusion branches, which did not improve their matched
plain active configurations.

## External Boundary

The same three fresh seeds were independently trained with pinned official
PatchTST commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`, matching H96,
25% chronological labels, `d_model=24`, `d_ff=48`, patch `24/8`, five epochs,
and the 64-window target budget.

| Candidate MSE | Official PatchTST MSE | Candidate vs PatchTST | Candidate wins | One-SE ratio upper |
| ---: | ---: | ---: | ---: | ---: |
| 8.148099 | 7.503567 | +8.590% | 0/3 | 1.08883 |

Thus the channel-patch objective is a valid high-dimensional causal
improvement, but it does not close the external accuracy gap and cannot
support a high-dimensional SOTA or Q1-level generalization claim. No
robustness extension or held-out test reader is authorized from this failed
external gate.

Artifacts:

- `artifacts/exploratory/electricity_channel_patch_strong_calibration_probe_20261006/audit.json`
- `artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/audit.json`
- `artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/external_patchtst_audit.json`
- `artifacts/exploratory/official_patchtst_electricity_channel_patch_replication_20261006/plan.json`
