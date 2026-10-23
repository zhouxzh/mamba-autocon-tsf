# PEMS NLinear Baseline Screen (2026-09-10)

## Purpose

This screen adds a reproducible NLinear reference to the PEMS03/04/07/08
Mamba/AutoCon exploratory track. It is a baseline-completion experiment, not a
method-selection experiment: all 16 jobs use the same `seq_len=96`, horizons
`12/96`, seeds `2022/2023`, four PEMS datasets, four training epochs, and
`AutoCon=false`.

The frozen plan is [`mamba_pems_nlinear_baseline_screen_20260910.json`](mamba_pems_nlinear_baseline_screen_20260910.json).
The artifact root is
`artifacts/exploratory/mamba_pems_nlinear_baseline_screen_20260910/`; its
manifest and phase validation record are the provenance boundary.

## Completed readout

All 16 registered jobs finished successfully. The primary metric is the best
pure forecasting validation loss from each `loss_logs.json`. The run was
registered with `skip_test=true`, so this screen has no test MSE/MAE and cannot
support a test-set ranking.

| Horizon | Rows | Mean best validation MSE | SD | Median MSE | Mean seconds |
|---:|---:|---:|---:|---:|---:|
| 12 | 8 | 0.124768 | 0.013771 | 0.127475 | 8.83 |
| 96 | 8 | 1.139899 | 0.088759 | 1.138302 | 10.54 |

The machine-readable rows and summaries are in
[`readout/validation_rows.csv`](../../artifacts/exploratory/mamba_pems_nlinear_baseline_screen_20260910/readout/validation_rows.csv)
and [`readout/summary.csv`](../../artifacts/exploratory/mamba_pems_nlinear_baseline_screen_20260910/readout/summary.csv).
The generated narrative readout is
[`readout/readout.md`](../../artifacts/exploratory/mamba_pems_nlinear_baseline_screen_20260910/readout/readout.md).

## Exact-coordinate comparison (seed 2022)

The older PEMS strong-baseline screen has only seed 2022, so the comparison is
restricted to that seed rather than mixing coordinates. Negative relative
change means NLinear has lower validation MSE.

| Reference | H12 mean change | H12 wins | H96 mean change | H96 wins |
|---|---:|---:|---:|---:|
| DLinear | +0.036% | 1/4 | +0.331% | 0/4 |
| Mamba2 | +46.719% | 0/4 | +325.041% | 0/4 |
| MambaTSv3 | +57.327% | 0/4 | +382.068% | 0/4 |
| PatchTST | +12.279% | 0/4 | +61.064% | 0/4 |
| TimesNet | +28.572% | 1/4 | +420.270% | 0/4 |

The exact paired CSVs are in
`artifacts/exploratory/mamba_pems_nlinear_baseline_screen_20260910/readout_seed2022/`.

## Interpretation for review

1. NLinear is now a complete two-seed PEMS screening baseline, but it is not a
   competitive long-horizon model in this configuration. It is roughly tied
   with DLinear and much worse than the neural baselines at H96.
2. This result does not establish a general ordering. It is validation-only,
   uses two seeds, and compares to an older seed-2022 screen with its own
   frozen training protocol.
3. The result does not justify removing NLinear. Keeping it in the final
   baseline table is important because it is a standard normalized-linear
   control and exposes the large H96 gap.

## Manual-review checklist

- [ ] Confirm that the PEMS target/channel convention matches the Mamba and
  TimesNet screens.
- [ ] Decide whether the final paper matrix needs NLinear on the same
  five-seed, three-horizon plan as the other plain baselines.
- [ ] Do not cite these validation values as final test results.
- [ ] Do not compare elapsed seconds across screens as an end-to-end speed
  claim; the screen is a small exploratory run with a different artifact root.
