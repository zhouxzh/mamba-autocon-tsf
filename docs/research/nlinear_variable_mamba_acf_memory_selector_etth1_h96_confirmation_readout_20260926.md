# Variable-Mamba ACF Memory Selector: ETTh1 H96 Confirmation (2026-09-26)

## Strict paired confirmation

This is a validation-only, independently seeded confirmation of the frozen
selector after the Traffic H96 screen. The selector-off and selector-on arms
share the same `NLinearVariableMambaAutoCon` backbone, train-only ACF prior,
prediction-connected AutoCon representation, lag-pair objective, split,
optimizer, budget, and data fingerprint. The only treatment difference is
`mamba_variable_acf_memory_selector`.

| seed | selector off | selector on | on change |
| ---: | ---: | ---: | ---: |
| 2064 | .652791 | .648336 | -0.6824% |
| 2065 | .653626 | .650265 | -0.5143% |
| 2066 | .660047 | .655406 | -0.7032% |
| mean | .655488 | .651336 | -0.6335% |

All six registered jobs completed. Both phases passed strict
`validate_training_phase.py` validation with `3/3` expected unique jobs and no
validator errors. Every job used `skip_test=true`; no test metrics were
generated or inspected.

The train-only ACF resolver selected `[72, 96, 120]` for all seeds, so the
selector had accessible memory inside the 96-step input window. The final
selector ReZero scales were `0.4141`, `-0.4224`, and `0.3974` for seeds
`2064/2065/2066`; the treatment was therefore active rather than a zero-scale
no-op. The paired result passes the frozen gate: lower mean validation loss,
`3/3` seed improvements, and nonzero treatment scales.

## Evidence boundary

This confirms a short-horizon validation signal on a second dataset. It does
not establish held-out test superiority, broad horizon transfer, external
baseline superiority, or inference/training speed. The earlier Traffic H96
resource preview measured the selector slower than its memory control, so no
speed claim is allowed. The next defensible step is a pre-registered test
release and a broader, same-source baseline matrix only after the independent
mechanism evidence is consolidated; post-hoc selector tuning remains closed.

Artifacts:

- `artifacts/exploratory/nlinear_variable_mamba_acf_memory_selector_etth1_h96_confirmation_20260926/off/`
- `artifacts/exploratory/nlinear_variable_mamba_acf_memory_selector_etth1_h96_confirmation_20260926/on/`
- `docs/research/nlinear_variable_mamba_acf_memory_selector_etth1_h96_confirmation_20260926.json`
