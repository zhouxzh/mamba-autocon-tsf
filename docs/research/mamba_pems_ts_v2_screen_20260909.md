# MambaTS v2 PEMS screen (2026-09-09)

## Scope

This validation-only exploratory screen compares the unchanged MambaTS v1
baseline, the MambaTS v2 candidate, Mamba-2, and their matched AutoCon arms on
PEMS03/04/07/08 at horizons 12 and 96.  Every coordinate uses the same
`MeanStd` windows, optimizer, four-epoch budget, and seed `2021`.

The test split was skipped.  There is one seed per coordinate, so this is a
diagnostic result and cannot support a final benchmark ranking, significance
claim, or generalization claim.

## Reproduction and integrity

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v2_screen_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_ts_v2_screen_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_ts_v2_screen_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_ts_v2_screen_20260909/readout
```

The frozen launcher completed **48/48** jobs.  The phase validation record is
marked valid, with 48 unique successful manifest rows and
`test_selection_allowed=false`:
`artifacts/exploratory/mamba_pems_ts_v2_screen_20260909/pems_mamba_ts_v2_h12_h96/phase_validation.json`.

## Validation results

| comparison | better / total | mean relative MSE change |
|---|---:|---:|
| `MambaTSv2 - MambaTS` | 6 / 8 | -4.38% |
| `MambaTSv2 - Mamba2` | 0 / 8 | +35.16% |
| `MambaTSv2AutoCon - MambaTSv2` | 4 / 8 | -0.19% |
| `MambaTSv2AutoCon - Mamba2AutoCon` | 0 / 8 | +34.48% |

The v2 improvement over v1 was concentrated in short-horizon and PEMS04
coordinates.  At horizon 96, v2 remained worse than v1 on PEMS07 and PEMS08.
AutoCon stayed close to the corresponding plain v2 arm and did not provide a
stable improvement pattern.

## Resource implication

Using the manifest elapsed time as a coarse training-job indicator, the mean
times over eight coordinates were approximately:

| model | mean elapsed seconds |
|---|---:|
| `MambaTS` | 12.12 |
| `MambaTSv2` | 39.23 |
| `MambaTSAutoCon` | 16.48 |
| `MambaTSv2AutoCon` | 45.63 |

This is not a throughput benchmark, but it is sufficient to reject the claim
that independent per-variable scans are automatically faster.  The v2 path
launches many short Mamba sequences (`batch x channels`), which is especially
expensive for PEMS07 (`883` channels).

## Decision

v2 is a useful diagnostic candidate because it closes part of the v1 gap while
preserving the old model as a negative control.  It is not a paper-ready
model: it remains behind Mamba-2 on every coordinate and has a substantial
high-dimensional runtime penalty.  The next candidate keeps the efficient
shared scan, adds an explicitly low-rank cross-variable context, and uses a
per-channel persistence gate; it must beat v1 and reduce the Mamba-2 gap before
any fresh-seed or test-release experiment is justified.

