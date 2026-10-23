# MambaTS v3 and AutoCon PEMS screen (2026-09-09)

## Scope

This validation-only exploratory screen compares plain `Mamba2`, `MambaTSv3`,
`Mamba2AutoCon`, and `MambaTSv3AutoCon` on PEMS03/04/07/08 at horizons 12 and
96.  The dataset arrays, `MeanStd` windows, optimizer, seed `2022`, and
four-epoch budget are held fixed.  Test windows were skipped and no result in
this screen was used to select a test model.

The purpose is to separate the effect of the v3 architecture from the effect
of adding the existing AutoCon objective.  It is not a confirmation experiment:
there is one seed, a short budget, and the primary metric is minimum validation
MSE.

## Reproduction and integrity

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v3_autocon_screen_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_ts_v3_autocon_screen_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_ts_v3_autocon_screen_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_autocon_screen_20260909/readout
```

All **32/32** jobs completed successfully.  The frozen phase record keeps
`test_selection_allowed=false`:
`artifacts/exploratory/mamba_pems_ts_v3_autocon_screen_20260909/pems_mamba_ts_v3_autocon_h12_h96/phase_validation.json`.

## Exact paired validation result

The primary metric is the minimum validation scaled MSE in each
`loss_logs.json`.  Deltas are paired by dataset, horizon, and seed; a negative
relative change means the candidate has lower validation MSE.

| comparison | better / total | mean relative MSE change | median | range |
|---|---:|---:|---:|---:|
| `MambaTSv3 - Mamba2` | 8 / 8 | -9.12% | -8.01% | -2.31% to -17.79% |
| `MambaTSv3AutoCon - MambaTSv3` | 1 / 8 | +1.64% | +1.36% | -0.54% to +4.78% |
| `MambaTSv3AutoCon - Mamba2AutoCon` | 8 / 8 | -7.14% | -6.97% | -0.56% to -13.95% |

The first contrast reproduces the plain v3 direction on this additional seed.
The second contrast does **not** support a general accuracy gain from adding
AutoCon to v3: AutoCon improves only PEMS08 horizon 96 and is worse on the
other seven coordinates.  The third contrast shows that v3 remains better than
the AutoCon-regularized Mamba-2 reference, but this does not isolate a positive
AutoCon contribution because both models differ in architecture.

The complete coordinate-level values are retained in:

`artifacts/exploratory/mamba_pems_ts_v3_autocon_screen_20260909/readout/paired_deltas.csv`

## Coarse elapsed-time signal

Mean job elapsed times from the same run were approximately `12.69 s` for
`MambaTSv3`, `22.83 s` for `Mamba2`, `17.08 s` for `MambaTSv3AutoCon`, and
`25.43 s` for `Mamba2AutoCon`.  These are wall-clock training-job readings
under one environment and budget, not a controlled throughput or memory
benchmark.

## Decision and next gate

This screen supports carrying **plain MambaTS v3** forward as the primary
candidate.  It does not justify claiming that the current AutoCon loss improves
v3.  The next confirmatory design should therefore keep the plain-v3 baseline,
include the AutoCon ablation as a secondary analysis, use multiple fresh seeds
and a longer budget, lock model selection before opening the test window, and
report matched inference resources and uncertainty intervals.
