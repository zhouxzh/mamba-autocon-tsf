# MambaTS v4 AutoCon representation-selected ACF memory (2026-09-22)

## Candidate

The v4 static ACF memory is driven by an immutable train-only prior. The new
selector reads the same AutoCon temporal representation and produces a shared
horizon-wise gate that makes a centered correction to the memory residual. It
has no channel-dependent parameters, starts with a zero ReZero scale, and is
inactive in the plain v4 control.

## Validation protocol

- ETTh1, ETTm1, Electricity, Traffic; H96; seeds 2021/2022/2023.
- 24/24 jobs, exact v4 static-memory control versus selector, validation-only.
- `AutoCon_lambda=0.001`, `MeanStd`, raw train-only ACF, lags `[24, 48]`, one
  MambaTS temporal scale, eight epochs, test execution disabled.
- Both phase validators report 12 unique jobs, valid provenance, and matching
  artifact hashes.

Plan: [`mamba_ts_v4_acf_memory_selector_screen_20260922.json`](./mamba_ts_v4_acf_memory_selector_screen_20260922.json)

Readout: [`readout.md`](../../artifacts/exploratory/mamba_ts_v4_acf_memory_selector_screen_20260922/readout/readout.md)

## Results

| Dataset | Static memory MSE | Selector MSE | Relative change | Wins / 3 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.708864 | 0.705486 | `-0.475%` | 2 |
| ETTm1 | 0.396016 | 0.392187 | `-0.943%` | 2 |
| Electricity | 0.159907 | 0.157705 | `-1.376%` | 3 |
| Traffic | 0.370867 | 0.368736 | `-0.572%` | 2 |

This is the first current-tree ACF/AutoCon candidate with a consistent
four-dataset paired improvement over its exact memory control. It remains
validation evidence, not a test or publication result.

## Baseline and efficiency boundary

Against the older six-model validation screen, the selector is still behind
ETTh1 PatchTST (selector `0.705486` versus PatchTST `0.685596`), while it is
descriptively better than the old ETTm1 PatchTST, Electricity TimesNet, and
Traffic PatchTST/TimesNet cells. Those baseline artifacts use a different
source snapshot and cannot be merged into a paper table; a new same-snapshot
baseline screen is required.

On Traffic/C862/H96/B8, RTX 5090 D, 50 warm-up and 200 timed device-resident
forwards, prediction-only latency is:

| Model | Median latency | Parameters | Peak increment |
| --- | ---: | ---: | ---: |
| v4 static memory | `0.7168 ms` | 104,124 | 152.6 MB |
| v4 selector | `0.7822 ms` | 110,365 | 153.9 MB |
| PatchTST | `0.8873 ms` | 46,080 | 95.2 MB |
| TimesNet | `2.8072 ms` | 1,302,046 | 17.9 MB |

The selector is about `9.1%` slower than static v4 but remains faster than the
two strong baselines in this one shape. This is inference latency only; it does
not establish end-to-end training speed.

Raw resources: [`selector resources`](../../artifacts/exploratory/mamba_ts_v4_acf_memory_selector_prediction_resources_traffic_20260922/model_resources.json) and [`baseline resources`](../../artifacts/exploratory/mamba_ts_v4_selector_baseline_resources_traffic_20260922/model_resources.json).

## Decision

Promote this selector to a frozen multi-horizon, same-snapshot baseline
confirmation candidate. Do not claim publication readiness yet: ETTh1 still
does not meet the strongest-baseline accuracy bar, and the current resource
comparison covers one horizon and one hardware/input shape.

