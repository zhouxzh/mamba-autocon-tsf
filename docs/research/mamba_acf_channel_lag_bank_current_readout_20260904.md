# Current-Tree Per-Channel ACF Lag-Bank Readout

**Status:** exploratory validation-only evidence. Test metrics were skipped; this is not a confirmatory result.

## Paired protocol

The dense and top-2 arms use the current variable-token `MambaACF` model, ETTh1 and Traffic, H96, seeds 2021--2023, `MeanStd` normalization, raw train-only ACF priors, the same optimizer and eight-epoch budget. The only registered factor is `mamba_acf_channel_lag_bank`: the treatment keeps the two strongest legal lag slots independently for each channel. The support mask is detached and adds no trainable parameters.

| Dataset | seed | dense bank | per-channel top-2 | relative change |
|---|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.740612 | 0.740586 | -0.003% |
| ETTh1 | 2022 | 0.718732 | 0.717995 | -0.103% |
| ETTh1 | 2023 | 0.712276 | 0.711739 | -0.075% |
| Traffic | 2021 | 0.381542 | 0.382954 | +0.370% |
| Traffic | 2022 | 0.379587 | 0.378477 | -0.292% |
| Traffic | 2023 | 0.381815 | 0.381087 | -0.191% |

The mean relative change is ETTh1 `-0.060%` (3/3 wins), Traffic `-0.037%` (2/3 wins), and pooled `-0.052%` (4/6 wins). This is a negligible validation difference, with one clear Traffic regression, and does not justify a test-enabled expansion by itself. Full artifacts are in [`mamba_acf_channel_lag_bank_current_pilot_20260904`](../../artifacts/exploratory/mamba_acf_channel_lag_bank_current_pilot_20260904/); the frozen plan is [`mamba_acf_channel_lag_bank_current_pilot_plan_20260904.json`](mamba_acf_channel_lag_bank_current_pilot_plan_20260904.json).

## What the mask actually did

The train-peaks-available resolver produced ETTh1 bank `[48, 68, 72, 89, 96, 120, 144]` and Traffic bank `[24, 48, 72, 96, 120, 144, 168]`. Legal forecast-step fractions were respectively `[.50, .708, .75, .927, 1, .75, .50]` and `[.25, .50, .75, 1, .75, .50, .25]`.

Top-2 support retained exactly `2/7 = 28.6%` of channel-lag entries. On Traffic, selected-slot counts were `[0, 0, 91, 831, 767, 34, 1]`; on ETTh1 they were `[0, 0, 3, 3, 7, 1, 0]`. Thus the treatment was a genuine sparse, channel-specific support restriction, not a temperature-only change.

## Decision

Stop this top-2 screen as a standalone accuracy direction. It is computationally cheap and may remain a useful diagnostic or deployment option, but the current validation evidence does not show a meaningful gain over the dense legal bank. A future replication should only be justified if it also measures test-set paired metrics and effective-memory telemetry, or if a different forecast-level coefficient/anchor mechanism is pre-registered.

