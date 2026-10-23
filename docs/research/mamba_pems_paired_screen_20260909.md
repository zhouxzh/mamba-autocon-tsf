# MambaTS paired PEMS screen (2026-09-09)

## Purpose

This was a validation-only exploratory screen to diagnose the local
MambaTS-inspired implementation.  It used the same PEMS windows,
`MeanStd` normalization, optimizer budget, and seed for each exact
`dataset x horizon` coordinate.

The screen is not a confirmation experiment: test windows were not read,
there was one seed per coordinate, no graph adjacency was used, and the
training budget was four epochs.  The rows must therefore not be reported as
a final benchmark ranking or as evidence of statistical significance.

## Reproduction

```bash
/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/run_frozen_exploratory_plan.py \
  --plan docs/research/mamba_pems_paired_screen_20260909.json \
  --artifact-root artifacts/exploratory/mamba_pems_paired_screen_20260909

/home/zhong/anaconda3/envs/mamba/bin/python \
  scripts/summarize_validation_pilot.py \
  --results-root artifacts/exploratory/mamba_pems_paired_screen_20260909 \
  --output-dir artifacts/exploratory/mamba_pems_paired_screen_20260909/readout
```

The launcher completed **32/32** jobs without a failure.  The machine-readable
rows and exact paired deltas are in
`artifacts/exploratory/mamba_pems_paired_screen_20260909/readout/`.

## Main validation signal

The primary value is the minimum validation scaled MSE in each
`loss_logs.json`.  Candidate-minus-reference deltas are paired on the same
dataset, horizon, and seed.

| comparison | coordinates better / total | observed direction |
|---|---:|---|
| `MambaTS - Mamba2` | 0 / 8 | MambaTS was worse on every coordinate |
| `MambaTSAutoCon - MambaTS` | 4 / 8 | AutoCon effect was small and mixed |
| `MambaTSAutoCon - Mamba2AutoCon` | 0 / 8 | MambaTSAutoCon was worse on every coordinate |

Relative MSE degradation of plain `MambaTS` versus `Mamba2` was:

| dataset | horizon 12 | horizon 96 |
|---|---:|---:|
| PEMS03 | +24.60% | +72.44% |
| PEMS04 | +27.15% | +104.49% |
| PEMS07 | +12.30% | +58.99% |
| PEMS08 | +10.06% | +27.20% |

For `MambaTSAutoCon - MambaTS`, the relative changes were `-0.47%` and
`+1.14%` on PEMS03, `-0.32%` and `-1.11%` on PEMS04, `+1.27%` and `+0.81%`
on PEMS07, and `+0.90%` and `-1.41%` on PEMS08 (horizons 12 and 96,
respectively).  This is not a stable improvement pattern.

The elapsed-time direction was favorable for MambaTS in this small screen,
but that does not establish end-to-end speed superiority: the implementation
was still slower than DLinear in the separate resource smoke and the timing
does not include a matched confirmation protocol.

## Interpretation and next gate

The result points to an architectural failure mode rather than an AutoCon
hyperparameter conclusion.  The current v1 scan concatenates all channel
patches into one sequence, so the Mamba state crosses from the final patch of
one sensor into the first patch of the next sensor without an explicit sensor
identity.  It also has no direct persistence/trend path.  The next candidate
is therefore implemented as `MambaTSv2` while retaining v1 unchanged.  A v2
run must use the same exact-coordinate screen before any fresh-seed or test
release is considered.

