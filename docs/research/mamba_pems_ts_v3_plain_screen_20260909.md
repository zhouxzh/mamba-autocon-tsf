# MambaTS v3 plain PEMS screen (2026-09-09)

## Scope

This validation-only exploratory screen compares the unchanged MambaTS v1
baseline, the MambaTS v2 independent-scan candidate, the shared MambaTS v3
candidate, and Mamba-2 on PEMS03/04/07/08 at horizons 12 and 96.  Every row
uses the same `MeanStd` windows, optimizer, four-epoch budget, and seed
`2021`.  Test windows were skipped.

The screen is diagnostic rather than confirmatory: one seed is not enough for
an inferential claim, and validation MSE cannot establish test generalization.

## Reproduction and integrity

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_ts_v3_plain_screen_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_ts_v3_plain_screen_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_ts_v3_plain_screen_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_ts_v3_plain_screen_20260909/readout
```

The frozen launcher completed **32/32** jobs with zero failures.  The phase
record reports 32 unique successful rows and keeps
`test_selection_allowed=false`:
`artifacts/exploratory/mamba_pems_ts_v3_plain_screen_20260909/pems_mamba_ts_v3_plain_h12_h96/phase_validation.json`.

## Exact paired validation signal

The primary metric is the minimum validation scaled MSE in each checkpoint's
`loss_logs.json`.  Candidate-minus-reference deltas are paired by dataset,
horizon, and seed.

| comparison | better / total | mean relative MSE change |
|---|---:|---:|
| `MambaTSv3 - MambaTS` | 8 / 8 | -32.57% |
| `MambaTSv3 - Mamba2` | 8 / 8 | -8.12% |

The v3 best validation MSE values were:

| dataset | horizon 12 | horizon 96 |
|---|---:|---:|
| PEMS03 | 0.068228 | 0.184816 |
| PEMS04 | 0.088454 | 0.231037 |
| PEMS07 | 0.065185 | 0.213343 |
| PEMS08 | 0.095146 | 0.369543 |

These numbers are encouraging but remain a one-seed exploratory result.  The
next gate is a fresh-seed paired screen, followed by a separately planned
AutoCon ablation; no test result may be used to choose the architecture.

## Coarse elapsed-time signal

Mean elapsed time over the eight coordinates was approximately:

| model | mean seconds |
|---|---:|
| `MambaTS` | 12.11 |
| `MambaTSv2` | 39.20 |
| `MambaTSv3` | 12.71 |
| `Mamba2` | 22.79 |

This is training-job wall time, not a controlled throughput benchmark.  It
does show that v3 preserves the shared-scan efficiency that v2 lost when it
expanded the batch by channel.

## Architecture interpretation

v3 retains the efficient channel-major shared scan from v1.  It adds a
zero-initialized low-rank context adapter derived only from the observed
normalized window and a zero-initialized per-channel persistence gate.  The
zero initialization provides a clear ablation path: early behavior is close to
the original scan, while optimization can learn cross-variable context and
last-value persistence when the validation objective supports them.

## Decision

v3 is the first local candidate that passes this exploratory accuracy-and-time
screen against Mamba-2.  It is not yet paper-ready.  Fresh seeds, AutoCon
pairing, longer training, test release after selection, matched inference
resource measurements, and a pre-specified statistical analysis are still
required.

