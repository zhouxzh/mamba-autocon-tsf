# MambaTS v3 fresh-seed PEMS screen (2026-09-09)

## Scope

This validation-only exploratory screen repeats the plain Mamba-2 versus
MambaTS v3 comparison on fresh seeds `2022` and `2023`.  The four PEMS arrays,
horizons 12/96, `MeanStd` windows, optimizer, and four-epoch budget are held
fixed.  Test windows were skipped.

The screen is still not a confirmation experiment: it has two seeds but only
one short training budget and uses validation MSE for direction finding.

## Reproduction and integrity

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v3_fresh_seeds_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_ts_v3_fresh_seeds_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_ts_v3_fresh_seeds_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_fresh_seeds_20260909/readout
```

All **32/32** jobs completed successfully.  The phase validation record has
16 exact paired rows, two seeds per dataset/horizon coordinate, and keeps
`test_selection_allowed=false`:
`artifacts/exploratory/mamba_pems_ts_v3_fresh_seeds_20260909/pems_mamba_ts_v3_fresh_h12_h96/phase_validation.json`.

## Exact paired validation result

| comparison | better / total | mean relative MSE change | range |
|---|---:|---:|---:|
| `MambaTSv3 - Mamba2` | 16 / 16 | -8.22% | -1.14% to -17.79% |

The advantage was present at both horizons and on all four PEMS arrays.  The
smallest observed margin was PEMS04 horizon 12/seed 2023 (`-2.21%` in the
machine-readable paired table; PEMS04 horizon 96/seed 2023 was `-1.14%`).
This is evidence of seed stability within this screen, not a claim of test-set
generalization or statistical significance.

## Coarse elapsed-time result

Mean job elapsed time across the 16 rows was approximately `12.67 s` for
MambaTS v3 and `22.45 s` for Mamba-2.  These are wall-clock training-job
readings, not a controlled throughput benchmark; they support only a
directional efficiency observation under this exact environment and budget.

## Decision

The v3 plain candidate passes the fresh-seed diagnostic gate: it improves the
validation MSE on every exact coordinate while preserving the shared-scan
runtime direction.  Before any paper claim, the AutoCon pair must be evaluated
on the same coordinates, then promising settings must be repeated with a
longer budget, additional seeds, a locked test release, and matched resource
protocols.

