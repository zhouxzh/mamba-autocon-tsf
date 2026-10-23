# MambaTS v5 signed-ACF-anchor PEMS screen

## Scope

This is a validation-only exploratory screen for `MambaTSv5AutoCon`. The v5 model keeps the v4 patch-token/shared-scan/context/AutoCon path and adds one train-only, causal, signed ACF anchor. For forecast step `h` and channel `c`, the anchor selects the largest-absolute train-split ACF lag whose observed index is available, reads only that normalized historical value, multiplies by the learned ACF sign, and applies one scalar zero-initialized ReZero gate. `MambaTSv4AutoCon` is the exact architectural control.

The screen contains 16 exact coordinates: PEMS03/04/07/08, horizons 12 and 96, seeds 2022 and 2023. Both arms use the same data fingerprints, four epochs, AutoCon lambda `0.001`, batch size, learning rate, and split policy. Test evaluation was disabled.

Plan: [`mamba_pems_ts_v5_anchor_screen_20260910.json`](mamba_pems_ts_v5_anchor_screen_20260910.json)

Artifact root: `artifacts/exploratory/mamba_pems_ts_v5_anchor_screen_20260910`

## Results

| Horizon | Coordinates | v5 wins vs v4 | Mean relative validation MSE change | Median relative change | Mean v5 best validation MSE | Mean abs. anchor gate | Nonzero anchor gates |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 8 | 8/8 | -0.0080% | -0.0049% | 0.080002 | 0.001569 | 8/8 |
| 96 | 8 | 4/8 | +0.0017% | +0.0053% | 0.255209 | 0.016138 | 8/8 |

Across all 16 coordinates, v5 wins 12/16 and has mean relative change `-0.0032%`. The absolute differences are very small: mean paired MSE delta is `-5.72e-6` at H12 and `+1.73e-6` at H96. The H12 win count therefore does not represent a practically meaningful accuracy gain.

The complete paired table is [`mambatsv4autocon_deltas.csv`](../../artifacts/exploratory/mamba_pems_ts_v5_anchor_screen_20260910/readout/mambatsv4autocon_deltas.csv); the model-level summary is [`summary.csv`](../../artifacts/exploratory/mamba_pems_ts_v5_anchor_screen_20260910/readout/summary.csv). The generated readout is [`readout.md`](../../artifacts/exploratory/mamba_pems_ts_v5_anchor_screen_20260910/readout/readout.md).

## Runtime and activation checks

- All 16 manifest records completed successfully.
- The source snapshot command explicitly used `MambaTSv5AutoCon` and `mamba_ts_variant=v5`.
- The screen used `mamba_use_acf_memory=true` and `AutoCon_acf_source=raw`; the prior is reconstructed from the training split only.
- `test_metrics_produced` is false for the screen; no test result was used for selection.
- Mean elapsed time was `18.199s` per job for v5 versus `18.089s` for the paired v4 screen, approximately `+0.61%`.
- The memory gate was nonzero in all rows. The signed-anchor gate was nonzero in all rows, but its mean absolute value was small, especially for H12. The representation-fusion gate was not enabled in this plan.

## Interpretation

This screen does not support a claim that the signed ACF anchor improves the model. It is a useful negative/neutral result: the causal lookup is executable and train-only, but under the current four-epoch budget and fixed lag bank it contributes almost no validation accuracy and adds a small runtime cost. The v5 anchor should not be promoted to the main paper model on this evidence.

The result also separates two questions that should not be conflated:

1. A train-only ACF signal can be implemented without target leakage.
2. This particular strongest-lag signed residual is not yet an effective accuracy mechanism for MambaTS on these PEMS coordinates.

## Manual review checklist

- [ ] Confirm the four PEMS SHA-256 fingerprints against `docs/research/pems_dataset_audit_20260909.md`.
- [ ] Confirm that `set_ts_acf_prior` receives only the training portion and that invalid early forecast steps are masked.
- [ ] Confirm paired rows use identical `(dataset, horizon, seed)` coordinates.
- [ ] Confirm this screen remains validation-only and is excluded from a final publication table unless replicated with a pre-registered full-budget protocol.
- [ ] Treat v5 as a neutral ablation/control, not as the claimed contribution.

