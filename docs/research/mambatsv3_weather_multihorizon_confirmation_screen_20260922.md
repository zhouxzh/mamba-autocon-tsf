# MambaTS v3 Weather Multi-Horizon Confirmation Screen (2026-09-22)

## Protocol

This validation-only Weather screen contains 63 jobs: MambaTS v3 plain,
MambaTS v3 AutoCon with an exact `lambda=0` null, active AutoCon with
`lambda=0.001`, and PatchTST/TimesNet/iTransformer/Mamba-2 baselines. Each arm
uses seeds 2021/2022/2023, MeanStd normalization, sequence length 96,
`d_model=32`, and eight training epochs. H96, H336, and H720 are included; test
evaluation was disabled for every job.

The plan and machine-readable readout are
[`mambatsv3_weather_multihorizon_confirmation_screen_20260922.json`](./mambatsv3_weather_multihorizon_confirmation_screen_20260922.json)
and
[`summary.csv`](../../artifacts/exploratory/mambatsv3_weather_multihorizon_confirmation_screen_20260922/readout/summary.csv).
All four phase validators report `valid=true`, with 63/63 unique jobs and no
failed jobs. This is an exploratory short-budget validation screen, not a
test-set result.

## Validation Results

Values are paired relative MSE changes; negative values favor the first model.
`wins` counts seed-level wins out of three.

| Horizon | Active AutoCon vs null | Null vs plain | Active vs PatchTST | Active vs TimesNet | Active vs iTransformer | Active vs Mamba-2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | `-0.358%` (3/3) | `+0.128%` (1/3) | `-10.010%` (3/3) | `-3.153%` (3/3) | `-7.933%` (3/3) | `-6.616%` (3/3) |
| 336 | `+0.066%` (1/3) | `-0.107%` (1/3) | `-6.572%` (3/3) | `-7.653%` (3/3) | `-6.382%` (3/3) | `-8.392%` (3/3) |
| 720 | `+0.289%` (0/3) | `+0.333%` (2/3) | `-5.095%` (3/3) | `-2.702%` (2/3) | `-5.401%` (3/3) | `-7.603%` (3/3) |

The MambaTS v3 forecast backbone is strong on this Weather screen and beats
all listed baselines at all three horizons under the short validation budget.
However, AutoCon improves its exact null only at H96 and is slightly worse at
H336/H720. The screen therefore supports a Weather candidate signal, not a
claim that contrastive representation learning improves long-horizon accuracy.

## H720 Prediction-Only Resources

These measurements use the same Weather shape (C=21, sequence 96, horizon 720,
batch 32), FP32 device-resident inputs, 50 warm-up iterations and 200 timed
forwards. AutoCon relation loss is not part of inference.

| Model | Trainable parameters | Median latency | p95 latency | Peak increment |
| --- | ---: | ---: | ---: | ---: |
| MambaTS v3 architecture | 342,853 | `0.272 ms` | `0.282 ms` | `15.93 MB` |
| PatchTST | 286,320 | `0.281 ms` | `0.290 ms` | `9.11 MB` |
| TimesNet | 1,254,117 | `8.436 ms` | `8.762 ms` | `100.28 MB` |
| iTransformer | 35,536 | `0.151 ms` | `0.158 ms` | `5.63 MB` |
| Mamba-2 | 153,249 trainable | `0.697 ms` | `0.708 ms` | `24.50 MB` |

The candidate is slightly faster than PatchTST and substantially faster than
TimesNet/Mamba-2, but slower than iTransformer and uses more parameters than
PatchTST. These are isolated forward measurements, not training throughput,
data-loading cost, or end-to-end time-to-accuracy.

## Decision

Keep MambaTS v3 as the current Weather candidate for a test-isolated
confirmation phase. Do not claim a general AutoCon gain until a longer-budget,
multi-seed test release compares active and null models. The current evidence
justifies the next confirmation experiment, but not a publication-ready
accuracy table or a universal efficiency claim.
