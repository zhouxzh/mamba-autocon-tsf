# Variable-Mamba ACF Memory Selector: Traffic H336 Confirmation Audit (2026-09-26)

## Invalid confirmation

The six H336 jobs completed and the basic phase validator initially reported
complete artifacts, but the mechanism-validity audit rejects this confirmation.
The resolved train-only lag set was `[144, 168, 336]` for `seq_len=96`.
`NLinearVariableMambaAutoCon._acf_memory` only retrieves an observed suffix
when `lag <= seq_len`; consequently the ACF memory was identically zero and
the selector scale remained exactly zero in all three treatment checkpoints.

| seed | selector off | selector on | selector scale |
| ---: | ---: | ---: | ---: |
| 2061 | .389895 | .389895 | 0.0 |
| 2062 | .387382 | .387382 | 0.0 |
| 2063 | .390067 | .390067 | 0.0 |

The apparent `3/3` numerical improvements are below floating-point reporting
precision and are not mechanism evidence. This screen is therefore excluded
from selector confirmation and from any paper table.

## Corrective action

The runtime now raises when selector-enabled training resolves no selected
train-only lag at or below `seq_len`. Strict phase validation also rejects a
selector artifact whose persisted `mamba_acf_prior_lags` has no accessible lag.
The Traffic H96 screen remains the only valid selector screen because it
resolved `[72, 96, 120]` and the selector scales became nonzero.

No H336 test metric was generated or inspected. The H336 artifacts remain
under `artifacts/exploratory/nlinear_variable_mamba_acf_memory_selector_traffic_h336_confirmation_20260926/`
for audit only.
