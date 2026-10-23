# MambaTS v3 versus strong PEMS baselines (2026-09-09)

## Scope

This validation-only exploratory screen compares plain `DLinear`, `PatchTST`,
`TimesNet`, `Mamba2`, and `MambaTSv3` on PEMS03/04/07/08 at horizons 12 and
96.  All rows use the same `MeanStd` normalization, data split, optimizer,
seed `2022`, and four-epoch budget.  Test windows were skipped.

The screen asks whether the v3 signal is limited to the Mamba family or also
survives against common forecasting baselines.  It is not confirmatory evidence:
one seed and a short budget cannot establish test generalization or statistical
significance.

## Reproduction and integrity

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_training_plan.py \
  --plan docs/research/mamba_pems_ts_v3_strong_baseline_screen_20260909.json \
  --validate_only

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v3_strong_baseline_screen_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_ts_v3_strong_baseline_screen_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_ts_v3_strong_baseline_screen_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_strong_baseline_screen_20260909/readout
```

Plan validation expanded to **40 jobs** and the frozen launcher completed
**40/40** successfully.  The phase record keeps
`test_selection_allowed=false`:
`artifacts/exploratory/mamba_pems_ts_v3_strong_baseline_screen_20260909/pems_mamba_ts_v3_strong_baselines_h12_h96/phase_validation.json`.

## Exact paired validation summary

The primary metric is minimum validation scaled MSE from `loss_logs.json`.
The generated paired table is
`artifacts/exploratory/mamba_pems_ts_v3_strong_baseline_screen_20260909/readout/paired_deltas.csv`.
Negative relative change means lower MSE for v3.

| comparison | better / total | mean relative MSE change | median | range |
|---|---:|---:|---:|---:|
| `MambaTSv3 - DLinear` | 8 / 8 | -57.04% | -54.77% | -29.06% to -82.89% |
| `MambaTSv3 - PatchTST` | 8 / 8 | -46.76% | -42.58% | -22.54% to -70.96% |
| `MambaTSv3 - TimesNet` | 5 / 8 | -5.94% | -3.51% | -31.47% to +16.48% |
| `MambaTSv3 - Mamba2` | 8 / 8 | -9.12% | -8.01% | -2.31% to -17.79% |

The coordinate-level validation winner was:

| dataset | horizon 12 | horizon 96 |
|---|---|---|
| PEMS03 | MambaTSv3 (`0.070167`) | TimesNet (`0.162275`) |
| PEMS04 | MambaTSv3 (`0.088279`) | TimesNet (`0.195812`) |
| PEMS07 | MambaTSv3 (`0.065697`) | MambaTSv3 (`0.215844`) |
| PEMS08 | MambaTSv3 (`0.095954`) | TimesNet (`0.375467`) |

Thus v3 is the best row on all four short-horizon coordinates and on PEMS07
H=96, while TimesNet is better on the other three long-horizon coordinates.
This is a mixed but useful result: v3 is a credible short-horizon candidate and
competitive at long horizon, not a universal winner.

A fresh-seed TimesNet replication is recorded in
[`mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md`](mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md).
With seed `2023`, v3 improves on `6/8` coordinates: all four H=12 cells and
PEMS07/08 at H=96, while TimesNet remains better on PEMS03/04 at H=96.  Across
the two exploratory seeds, v3 wins `11/16` paired cells against TimesNet; the
combined H=96 direction is only `3/8` wins, so the long-horizon limitation is
not treated as noise.

## Coarse elapsed-time signal

Mean job elapsed times in this screen were approximately `6.20 s` for DLinear,
`18.76 s` for PatchTST, `25.03 s` for TimesNet, `22.54 s` for Mamba2, and
`12.64 s` for MambaTS v3.  These are wall-clock training-job readings from one
environment and budget, not a controlled throughput, memory, or end-to-end
speed benchmark.

## Decision and next gate

Carry plain MambaTS v3 forward as a candidate for short-horizon traffic
forecasting and as a long-horizon competitor.  Keep TimesNet as a mandatory
long-horizon comparator and do not write a blanket accuracy claim.  The next
confirmation should use fresh seeds beyond 2022/2023, a longer budget, the same
baseline set, locked model selection before opening the test split, and
uncertainty intervals reported per dataset and horizon.
