# Frozen H192/H336 Held-Out Plan: 2026-10-03

## Scope

This release opens the PEMS-BAY chronological test partition exactly once for
the validation-frozen 10% low-data H192 and H336 checkpoints. It evaluates
existing model states only; it does not retrain, tune, select, or alter a
checkpoint after seeing test data.

## Frozen Coordinates

| Horizon | Seeds | Hybrid checkpoint root | Primary root | Strong-baseline root |
| --- | --- | --- | --- | --- |
| H192 | 5410, 5411, 5412 | `mambatsv3_acf_infonce_h192_screen_20261003` | `mambatsv3_acf_infonce_horizon_primary_validation_20261003` | `mambatsv3_acf_infonce_horizon_strong_baselines_20261003` |
| H336 | 5413, 5414, 5415 | `mambatsv3_acf_infonce_h336_screen_20261003` | `mambatsv3_acf_infonce_horizon_primary_validation_20261003` | `mambatsv3_acf_infonce_horizon_strong_baselines_20261003` |

Each coordinate evaluates the active signed-InfoNCE Hybrid, NLinear, PatchTST,
and iTransformer. NLinear control checkpoints are stored by the common Hybrid
trainer with `linear_only=true`; test evaluation restores them through the
same Hybrid wrapper, which executes only the NLinear branch. The Hybrid's
source head was pretrained at the same horizon and transferred with
`prediction_head_transfer_mode=exact`.

## Preconditions

- Source selection uses the pre-existing PEMS-BAY few-shot-only artifact.
- H192/H336 active/null validation gave `-1.395%/-1.956%`, both 3/3 wins.
- Hybrid beats NLinear, PatchTST, and iTransformer in both horizons, 3/3.
- Every stored validation artifact reports `test_metrics_produced=false`.
- Test evaluation uses `scripts/evaluate_lowdata_checkpoint_test.py`; each
  invocation is GPU-guarded at 78C/85% fan and reports
  `training_performed=false`.

## Readout Rules

Report mean test MSE, relative difference, paired wins, and the one-standard-
error upper ratio against NLinear. Retain all six seeds and both horizons even
if any coordinate is negative. This plan does not authorize changing sources,
loss weights, horizons, seeds, or baseline budget after the test readout.
