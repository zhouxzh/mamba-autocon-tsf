# MambaACF Channel-State Mixer: Three-Dataset H96 Confirmation

## Status

Stopped as a universal Mamba time-series forecasting candidate. The rank-4 channel-state mixer has reproducible high-dimensional validation evidence, but fails the pre-registered external-baseline gate on ETTh1. No held-out test metrics were generated or read for this branch.

## Frozen Protocol

Three independent 27-job, validation-only factorials used Electricity (321 variables), Traffic (862 variables), and ETTh1 (7 variables). Each coordinate contains DLinear, NLinear, PatchTST, TimesNet, and iTransformer at three fresh seeds, plus a `MambaACFAutoCon` 2x2 factorial:

- rank-4 `LowRankGlobalChannelStateMixer` off/on;
- AutoCon relation lambda `0/.001`.

All candidate arms retain one shared temporal Mamba scan, train-only raw-ACF forecast memory, MeanStd normalization, a prediction-connected AutoCon representation, `L=96`, `H=96`, eight epochs, and `skip_test=true`. The mixer conditions the existing shared hidden summary through a zero-initialized low-rank FiLM residual; it does not add a Mamba scan or a dense channel-by-channel matrix. Strict readers verify complete coverage, per-phase snapshot stability, finite validation losses, ACF sidecars, no test artifacts, and nonzero trained ReZero scales for all mixer-on checkpoints.

## Registered Result

| Dataset | Primary mixer-on active MSE | Mixer active effect | Paired wins | Strongest external baseline | Candidate vs baseline | Branch gate |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Electricity | 0.155544 | -1.347% | 2/3 | TimesNet (0.169221) | -8.082% | pass |
| Traffic | 0.396263 | -0.145% | 2/3 | iTransformer (0.413117) | -4.080% | pass |
| ETTh1 | 0.717523 | -0.125% | 2/3 | iTransformer (0.686635) | +4.498% | fail |

The mixer effect remains directionally positive against its exact active control on every dataset, but its magnitude is small on ETTh1 and Traffic. More importantly, ETTh1 misses the external-baseline requirement by 4.498%. The architecture cannot be described as a universal or cross-domain non-inferior model.

Across Electricity, Traffic, and ETTh1, active AutoCon relation relative to the exact mixer-on null is `+0.012%`, `+0.013%`, and `+0.007%`, respectively. All three effects meet only the registered 1% non-inferiority bound; none improves the mean, and paired relation wins are `0/3`, `1/3`, and `0/3`. AutoCon remains an implemented prediction-connected representation-learning component and a required ablation, but there is no evidence that the current relation loss supplies an independent forecasting gain.

## Resource Boundary

The rank-4 mixer adds 453 trainable parameters in all shapes. Exact-shape B32 prediction-only median latency changes from `.883` to `.860 ms` on Electricity, `.825` to `.873 ms` on ETTh1, and `.922` to `.973 ms` on Traffic. These are small overhead checks only. This branch is neither faster than its own mixer-off control on all datasets nor a speed winner against DLinear/NLinear/iTransformer, so it cannot satisfy the requested efficiency claim.

## Decision

Stop rank changes, lambda tuning, multi-horizon expansion, and held-out-test release for this channel-state mixer as a universal main model. Retain Electricity/Traffic as explicitly high-dimensional, validation-only mechanism evidence. The next paper candidate must separately demonstrate both external-baseline non-inferiority and a named speed advantage on a pre-registered multi-dataset family.

Artifacts: [Electricity plan](mamba_acf_channel_mixer_electricity_h96_20260927.json), [ETTh1 plan](mamba_acf_channel_mixer_etth1_h96_20260927.json), [Traffic plan](mamba_acf_channel_mixer_traffic_h96_20260927.json); [Electricity reader](mamba_acf_channel_mixer_electricity_h96_20260927_readout/readout.md), [ETTh1 reader](mamba_acf_channel_mixer_etth1_h96_20260927_readout/readout.md), and [Traffic reader](mamba_acf_channel_mixer_traffic_h96_20260927_readout/readout.md).
