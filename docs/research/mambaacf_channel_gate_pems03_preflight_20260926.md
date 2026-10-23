# MambaACF low-rank channel gate: PEMS03 preflight (2026-09-26)

## Purpose

This was a causal preflight for the existing low-rank, window-conditioned
cross-variable gate in `MambaACF`. It was selected after the v8 causal spectral
residual failed on PEMS03. The comparison uses the same PEMS03 H96 snapshot,
seed `2181`, batch size `32`, three epochs, `d_model=32`, one Mamba scale, and
GPU `cuda:0`.

The gate run uses `MambaACFAutoCon` with the existing signed ACF relation loss,
`AutoCon_lambda=0.001`, `MeanStd` normalization, and rank-4 raw channel gate.
The null run uses the exact same AutoCon recipe with the gate disabled. A plain
non-AutoCon `MambaACF` run is retained only as a context reference; it is not a
causal comparator for the gate.

## Validation result

| configuration | parameters | best validation MSE | test |
|---|---:|---:|---|
| plain MambaACF (context only) | 76,346 | 0.139925 | not run |
| AutoCon relation null | 78,458 | 0.218195 | not run |
| AutoCon + rank-4 channel gate | 83,475 | 0.215591 | not run |

The gate delta against the exact AutoCon null is `-1.194%` in this one-seed,
three-epoch preflight. This is a small descriptive improvement, not evidence of
a stable gain. The gate adds 5,017 parameters (6.39%) and its comparison with
plain MambaACF is confounded by the AutoCon objective: the plain-context gap is
`+54.07%` and must not be attributed to the gate.

## Representation diagnostics

The active and null AutoCon logs are nearly indistinguishable:

- positive/contrastive anchors are degenerate at `100%`;
- contrastive anchors are `0%`;
- sign accuracy is approximately `48.2--48.9%`;
- ACF candidate coverage is only about `9%` of windows, despite about `94%`
  active anchors.

This means the current relation objective is not providing a useful
contrastive signal in this setting. The small gate delta therefore cannot be
interpreted as successful ACF-guided representation learning.

## Decision

Stop expanding this gate as a paper candidate from this preflight. Do not
release PEMS test evaluation or tune rank/lambda based on this result. A future
continuation would first need a repaired non-degenerate positive-pair policy,
then a preregistered multi-dataset, multi-seed screen against the exact null.

Artifacts:

- `artifacts/exploratory/mambaacf_gate_pilot_20260926/plain.log`
- `artifacts/exploratory/mambaacf_gate_pilot_20260926/null.log`
- `artifacts/exploratory/mambaacf_gate_pilot_20260926/gate.log`

