# Variable-Mamba ACF Memory Selector: Traffic H96 Readout (2026-09-26)

## Valid screen

This is a validation-only, paired mechanism screen. Both arms used the same
`NLinearVariableMambaAutoCon` backbone, train-only raw ACF prior,
prediction-connected AutoCon representation, lag-pair relation objective,
optimizer, split, epoch budget, and seeds. The only treatment difference was
`mamba_variable_acf_memory_selector`.

The selector uses a rank-2 horizon and rank-1 channel gate produced from the
same AutoCon representation that is connected to the forecast path. It adds
no Mamba scan. Its ReZero scale is zero at initialization.

| seed | selector off | selector on | on change |
| ---: | ---: | ---: | ---: |
| 2051 | .398185 | .392970 | -1.3097% |
| 2052 | .393656 | .389629 | -1.0230% |
| 2053 | .396369 | .391810 | -1.1502% |
| mean | .396070 | .391470 | -1.1615% |

All six registered jobs completed and both phases passed strict
`validate_training_phase.py` validation (`3/3` expected unique jobs,
`valid=true`, no validator errors). All jobs used `skip_test=true`; no test
metrics were generated or inspected.

The fitted train-only, availability-weighted ACF lags were `[72, 96, 120]`,
so at least one lag is available to the 96-step history. The final selector
scales for seeds `2051/2052/2053` were `-.8091/-.9001/-.7723`, respectively.
This confirms that the treatment did not remain a zero-initialized no-op.
The relation loss also retained positive global active-anchor coverage.

## Resource boundary

The corresponding explicit Traffic H96 resource preview (RTX 5090 D, FP32,
B8, device-resident inputs, 20 warm-up and 50 timed forwards) measured
`.36568 ms` for the selector and `.24714 ms` for the memory control. This is
a mechanism-cost preflight, not an end-to-end training-time measurement and
not a comparison with all external baselines. The selector therefore has an
observed forward cost and cannot currently support a speed-superiority claim.

## Earlier invalid attempts

Two earlier directories under the same experiment family are retained for
audit but excluded from this readout. The first declared `fixed` lag mode even
though the training path resolved train peaks, so strict phase validation
failed. The second corrected the declaration but selected `[168,336,504]`,
all longer than the `seq_len=96` history; its ACF memory was zero and selector
scales stayed at zero. A runtime/plan/validator guard now rejects selector
plans with `relation_min_lag >= seq_len`; the valid screen uses
`train_peaks_available` instead.

## Decision

The pre-registered initial gate is met: lower mean validation MSE, `3/3`
paired seed improvements, active selector scales, valid train-only sidecars,
and completed phase validation. This licenses exactly one independent
dataset/horizon confirmation with the selector architecture and all training
settings frozen. It does not license a test release, baseline-superiority
claim, speed claim, rank search, or a publication claim.

Artifacts:

- `artifacts/exploratory/nlinear_variable_mamba_acf_memory_selector_traffic_h96_screen_20260926_available_lags/off/`
- `artifacts/exploratory/nlinear_variable_mamba_acf_memory_selector_traffic_h96_screen_20260926_available_lags/on/`
- `artifacts/exploratory/nlinear_variable_mamba_acf_selector_preflight_20260926/`
