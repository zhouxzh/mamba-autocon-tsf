# Current-Tree ACF Memory and Temperature Readout

**Status:** exploratory validation-only evidence. Both screens skipped the test split; neither supports a paper claim.

## Memory on/off

The current variable-token `MambaACF` implementation was run on ETTh1 and Traffic at H96 with seeds 2021--2023, `MeanStd` normalization, raw train-only ACF priors, eight training epochs, and identical optimizer/split settings. The only registered factor was `mamba_use_acf_memory`.

| Dataset | seed | memory off | memory on | relative change (on-off) |
|---|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.742545 | 0.740612 | -0.260% |
| ETTh1 | 2022 | 0.720758 | 0.718732 | -0.281% |
| ETTh1 | 2023 | 0.713358 | 0.712276 | -0.152% |
| Traffic | 2021 | 0.380777 | 0.381542 | +0.201% |
| Traffic | 2022 | 0.383423 | 0.379581 | -1.002% |
| Traffic | 2023 | 0.385059 | 0.381811 | -0.843% |

The paired mean change is ETTh1 `-0.232%` (3/3 wins), Traffic `-0.550%` (2/3 wins), and pooled `-0.342%` (5/6 wins). These are best validation losses, not test MSE; the paired sample is too small for a confirmatory inference. The complete 12-job artifacts are in [`mamba_acf_memory_current_pilot_20260904`](../../artifacts/exploratory/mamba_acf_memory_current_pilot_20260904/), with the plan in [`mamba_acf_memory_current_pilot_plan_20260904.json`](mamba_acf_memory_current_pilot_plan_20260904.json).

The resolved train-peaks-available compact banks are dataset-specific: ETTh1 resolves to `[48, 68, 72, 89, 96, 120, 144]`, while Traffic resolves to `[24, 48, 72, 96, 120, 144, 168]`. They must not be pooled with the earlier three-slot `[72, 96, 120]` historical screen.

## Temperature 4 versus 1

Using the same current-tree variable-token MambaACF protocol and the same two datasets/seeds, the only factor was the static-memory softmax temperature.

| Dataset | paired mean relative change (T=1 versus T=4) | wins | mean elapsed T=4 -> T=1 |
|---|---:|---:|---:|
| ETTh1 | -0.057% | 3/3 | 20.585 s -> 20.639 s |
| Traffic | -0.045% | 2/3 | 56.761 s -> 57.755 s |
| pooled | -0.053% | 4/6 | -- |

At T=4, the mean lag-weight entropy is approximately `1.612` for ETTh1 and `1.386` for Traffic; at T=1 it is `1.577` and `1.376`, respectively. Thus T=1 sharpens the distribution only modestly under the legal-step mask. The lag banks and train-only priors are unchanged between temperature arms.

The temperature screen is therefore stopped. A small numerical improvement with no consistent Traffic effect and a small runtime increase is not enough to justify further temperature tuning. The next targeted factor is per-channel top-k legal lag support, which changes which ACF values enter the forecast rather than only rescaling an almost shared mixture.

## Interpretation boundary

The current memory path is causal in its historical index lookup and uses a detached train-only prior, but it is still a residual expert gated by a learned hidden-state gate. The results show that the branch affects validation forecasts; they do not show that ACF identifies the channels or horizons that benefit. In particular, no telemetry was collected in this plain validation run (`stream_metrics=false`), so effective-memory RMS and gate strength must be checked in a later test-enabled replication.

