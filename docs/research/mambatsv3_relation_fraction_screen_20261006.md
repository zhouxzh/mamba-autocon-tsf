# AutoCon Relation-Fraction Screen

Status: completed validation-only mechanism screen. No target test rows were
read. The experiment asks whether the current source-only ACF pretraining
benefit is being suppressed merely by a too-small relation-loss fraction.

## Contract

Mamba-1, MambaTSv3 v3, 24/8 patches, channel-agnostic shared encoder, exact
10% target prefix, frozen target core, next-patch source prediction and
train-only forecast-contrastive ACF pairs were held fixed. Only the fraction
of predictive pretraining loss allocated to the ACF relation objective changed:
`0.05`, `0.10`, `0.20`, and `0.30`. Every active arm has an exact `lambda=0`
null at the same seed (`5980`) and target coordinate.

| Fraction | PEMS-BAY MSE | METR-LA MSE | Both beat null | Relative to 0.10 mean |
| ---: | ---: | ---: | --- | ---: |
| 0.05 | 2.784279 | 1.590410 | yes | +0.022% |
| 0.10 | 2.782655 | 1.591089 | yes | 0.000% |
| 0.20 | 2.785675 | 1.591222 | yes | +0.072% |
| 0.30 | 2.784128 | 1.591004 | yes | +0.032% |

The exact nulls were `2.872610` (PEMS-BAY) and `1.599454` (METR-LA). Thus all
four active fractions were directionally better than their null on this
screen, but none improved the existing `0.10` setting on both targets. The
screen therefore has no replication-eligible alternative fraction.

## Decision

Keep `relation_loss_fraction=0.10` as the frozen protocol value. Do not
continue increasing the relation weight or promote a fraction selected from
this single validation seed. This closes the simple-loss-weight explanation
for the remaining high-dimensional gap; subsequent work must improve ACF pair
quality, cross-domain normalization, or target adaptation, with a new
pre-registered mechanism and an exact null.

The experiment used only train-prefix source data and target validation for
selection. All 20 training/adaptation stages were serialized through
`run_gpu_guarded.py`; no guard triggered. Peak telemetry was 55C/32% fan on
PEMS-BAY and 52C/0% fan on METR-LA.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_relation_fraction_screen_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_relation_fraction_screen_20261006/audit.json`
- `scripts/run_mambatsv3_mamba1_relation_fraction_screen.py`
- `scripts/audit_mambatsv3_mamba1_relation_fraction_screen.py`
