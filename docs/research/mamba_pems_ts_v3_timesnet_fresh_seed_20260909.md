# MambaTS v3 versus TimesNet fresh-seed PEMS replication (2026-09-09)

## Scope

This validation-only exploratory replication compares plain `MambaTSv3` and
`TimesNet` on PEMS03/04/07/08 at horizons 12 and 96 with fresh seed `2023`.
The split, `MeanStd` normalization, optimizer, model width, and four-epoch
budget match the seed-2022 strong-baseline screen.  Test windows were skipped.

## Reproduction and integrity

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_training_plan.py \
  --plan docs/research/mamba_pems_ts_v3_timesnet_fresh_seed_20260909.json \
  --validate_only

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v3_timesnet_fresh_seed_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_ts_v3_timesnet_fresh_seed_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_ts_v3_timesnet_fresh_seed_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_timesnet_fresh_seed_20260909/readout
```

The plan expanded to **16 jobs** and all **16/16** completed successfully.  The
phase record keeps `test_selection_allowed=false`:
`artifacts/exploratory/mamba_pems_ts_v3_timesnet_fresh_seed_20260909/pems_mamba_ts_v3_timesnet_fresh_seed_h12_h96/phase_validation.json`.

## Fresh-seed paired result

The primary metric is minimum validation scaled MSE.  Candidate-minus-reference
deltas are in
`artifacts/exploratory/mamba_pems_ts_v3_timesnet_fresh_seed_20260909/readout/paired_deltas.csv`.

| dataset | H=12 relative change | H=96 relative change |
|---|---:|---:|
| PEMS03 | -9.18% | +17.59% |
| PEMS04 | -10.79% | +10.16% |
| PEMS07 | -30.70% | -9.14% |
| PEMS08 | -32.06% | -0.67% |

`MambaTSv3` improves on **6/8** fresh-seed coordinates: **4/4** at H=12 and
2/4 at H=96.  The result reinforces a horizon- and dataset-dependent claim:
v3 is a strong short-horizon candidate, while TimesNet remains a credible
long-horizon comparator on PEMS03/04.

Combining this seed with the independent seed-2022 strong-baseline screen gives
11/16 v3 wins against TimesNet, mean relative change `-7.02%`; by horizon, the
combined direction is 8/8 wins at H=12 and 3/8 at H=96.  This pooled descriptive
view is not a statistical confirmation because it still uses a four-epoch
validation screen and no locked test release.

## Decision

Keep TimesNet as a mandatory long-horizon baseline and avoid a headline claim of
universal v3 superiority.  A paper-level run must pre-register the same
comparison, use a longer budget and at least five seeds, report per-cell
uncertainty, and open the test split only after model selection is frozen.

