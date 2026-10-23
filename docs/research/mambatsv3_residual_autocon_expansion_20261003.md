# MambaTSv3 Residual AutoCon Expansion: 2026-10-03

## Purpose

This release tests the frozen residual-pretraining recipe on two additional
traffic domains. The target-side selector uses only the 10% few-shot prefix;
the source selector uses only source training prefixes and is bound by SHA-256
manifests. Main validation and test rows are not available during selection.

The deployed model is:

```text
shared channel-agnostic MambaTSv3 pretraining
  -> next-patch last-value residual target
  -> forecast-affine ACF relation objective (calibrated to 10% of predictive loss)
  -> exact P*d -> H96 head transfer
  -> NLinear primary + one-scan Mamba residual correction
```

AutoCon is training-only. Prediction-only inference calls `return_repr=False`,
does not evaluate a relation loss, and retains one Mamba scan.

## 10% Validation

Three fresh paired seeds were used per target. The target primary and sampler
were selected before the main validation block: NLinear and uniform sampling
for both targets.

| Target | NLinear mean | Hybrid mean | Relative change | Wins |
| --- | ---: | ---: | ---: | ---: |
| METR-LA | 1.483978 | 1.421429 | -4.215% | 3/3 |
| PEMS-BAY | 2.923039 | 2.632410 | -9.943% | 3/3 |

## 25% Validation

The source checkpoints and recipe were frozen; only the target few-shot
fraction changed to 25%.

| Target | NLinear mean | Hybrid mean | Relative change | Wins |
| --- | ---: | ---: | ---: | ---: |
| METR-LA | 1.094365 | 1.064213 | -2.755% | 3/3 |
| PEMS-BAY | 2.506467 | 2.238789 | -10.679% | 3/3 |

## Held-Out Test

The held-out release was opened only after the frozen selectors and validation
checkpoints were complete. The control, Hybrid, iTransformer, and PatchTST
were trained under the same low-data budget and evaluated after loading their
best validation state.

| Target | Control | Hybrid | vs control | vs iTransformer | vs PatchTST | Wins vs control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| METR-LA | 2.385117 | 2.294146 | -3.814% | -10.596% | -5.964% | 3/3 |
| PEMS-BAY | 2.755797 | 2.506659 | -9.041% | -9.312% | -3.889% | 3/3 |

The strict machine-readable audit is
`artifacts/paper_final/mambatsv3_residual_autocon_expansion_confirmatory_20261003/audit.json`.
The generated readout is in the same directory.

## Robustness

Each held-out coordinate was evaluated on closed validation windows under
late-block shift, 10% and 30% channel masking, and noise scales 0.05 and
0.10. Hybrid beat the paired NLinear control in all 12 target-condition
aggregates, with 3/3 paired wins for every condition. Mean relative changes
were:

| Target | In distribution | Late shift | Missing 10% | Missing 30% | Noise 0.05 | Noise 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| METR-LA | -4.215% | -4.099% | -3.738% | -2.824% | -4.216% | -4.217% |
| PEMS-BAY | -9.943% | -9.436% | -8.965% | -7.044% | -9.942% | -9.928% |

## Inference Boundary

Prediction-only benchmarks use batch size 8, 25 warm-up iterations, and 100
timed iterations on the RTX 5090 D. METR-LA is 0.328 ms median and PEMS-BAY
is 0.382 ms median. Both records report `mamba_scan_count=1`,
`autocon_representation_returned=false`, and `relation_loss_included=false`.
These are fixed-shape forward latency measurements, not end-to-end training
throughput claims.

## ACF Active/Null Mechanism Control

The expansion also includes an exact source-side mechanism control. The
prediction target, source domains, source initialization, seed, target
fine-tuning, and model are held fixed; only the source forecast-affine ACF
term changes from the calibrated active value to zero. This is validation-only
and does not reopen held-out test metrics.

| Target | ACF active mean | ACF null mean | Active vs null | Active wins |
| --- | ---: | ---: | ---: | ---: |
| METR-LA | 1.421429 | 1.417438 | +0.282% | 0/3 |
| PEMS-BAY | 2.632410 | 2.605412 | +1.036% | 0/3 |

This rejects the current forecast-affine ACF loss as the causal explanation
for the expansion gains. It is retained as a negative, reproducible ablation;
the positive residual-transfer result must be described as predictive
pretraining plus exact-head transfer, not as a demonstrated AutoCon benefit.

## Interpretation and Remaining Gaps

The expansion provides strong evidence for a low-data, cross-domain predictive
residual-transfer effect: the gain persists at 10% and 25%, across four
traffic/weather targets, across three seeds, and against two nonlinear
baselines on the new held-out domains. It does not establish a positive AutoCon
mechanism or universal superiority. The current release still uses H96, a
small source pretraining budget, and a linear primary model chosen from a
limited candidate set. Before an SCI Q1 submission, the work needs a redesigned
ACF contrastive objective that beats its exact null, horizon transfer (at least
H192/H336), a full-data reference, and an agreed external baseline
implementation or published-result comparison.

## Artifacts

- Validation: `artifacts/paper_final/mambatsv3_residual_autocon_expansion_validation_20261003/`
- 25% validation: `artifacts/paper_final/mambatsv3_residual_autocon_expansion_25_validation_20261003/`
- Held-out release: `artifacts/paper_final/mambatsv3_residual_autocon_expansion_confirmatory_20261003/`
- Active/null validation: `artifacts/paper_final/mambatsv3_residual_autocon_expansion_null_validation_20261003/`
- Audit script: `scripts/audit_mambatsv3_residual_expansion_release.py`
