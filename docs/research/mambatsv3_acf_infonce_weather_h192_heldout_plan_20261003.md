# Frozen Weather H192 Held-Out Plan: 2026-10-03

This release reads the chronological Weather test partition once for the
validation-frozen H192 checkpoints from seeds 5450--5452. Active signed
InfoNCE Hybrid, NLinear, PatchTST, iTransformer, and TimesNet all use 10%
target data, five target epochs, and 64 training windows.

The source selector is the pre-existing few-shot-only
METR-LA/PEMS-BAY/Electricity artifact. Validation artifacts have
`test_metrics_produced=false`; test uses the read-only evaluator, whose output
must record `training_performed=false`.

| Validation comparison | Active relative change | Wins |
| --- | ---: | ---: |
| NLinear | -2.449% | 3/3 |
| PatchTST | -4.478% | 3/3 |
| iTransformer | -14.968% | 3/3 |
| TimesNet | -9.282% | 3/3 |

The plan fixes models, source pool, seeds, budget, and horizon. No test result
may cause retraining, source reselection, or changed hyperparameters.
