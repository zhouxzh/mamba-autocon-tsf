# Variable-Mamba ACF Selector: Electricity H96 Strong-Baseline Confirmation

**Status:** stopped after failing the registered external-baseline accuracy gate. No held-out test metric was produced or read.

This fresh-seed Electricity confirmation froze the selector that passed the Traffic and ETTh1 H96 screens. It retains one variable-axis Mamba-1 scan, NLinear forecast base, prediction-connected AutoCon representation, train-only ACF memory, and the zero-initialized rank-2 horizon plus rank-1 channel selector. All external baselines explicitly disable candidate-only relation options. The 21 validation jobs use seeds `2261--2263`, `seq_len=96`, `pred_len=96`, eight epochs, and `skip_test=true`.

## Registered Result

| Model or condition | Mean best validation MSE | Notes |
| --- | ---: | --- |
| Selector off | 0.185067 | Exact paired control |
| Selector on | 0.184745 | 3/3 paired seed wins; finite nonzero selector scales |
| TimesNet | 0.171796 | Strongest external baseline |
| iTransformer | 0.195824 | Same budget |
| PatchTST | 0.196422 | Same budget |
| DLinear | 0.200868 | Same budget |
| NLinear | 0.202235 | Same budget |

The selector itself is active and improves its exact control by `0.174%` at all three seeds, so the mechanism gate passes. However, selector-on is `7.537%` worse than TimesNet. The external-baseline accuracy gate therefore fails. The strict reader verified complete `(model, seed)` coverage, source-snapshot consistency, training completion, the absence of test artifacts, legal train-only lags, ACF sidecars, and finite nonzero selector scales.

## Decision

This result closes the current selector architecture for cross-dataset expansion, post-hoc selector tuning, long-context expansion, and held-out-test release. Traffic and ETTh1 remain valid local validation observations, but they cannot support a general accuracy or paper claim once the third, pre-registered high-dimensional dataset fails decisively. A later architecture must be evaluated as a new candidate with a new registered comparison protocol, not presented as a continuation of this selector's evidence.

Exact artifacts: [plan](nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_20260927.json), [launch authorization](nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_20260927_launch_sidecar.json), [strict readout](nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_readout_20260927/readout.md), [row-level validation data](nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_readout_20260927/validation_rows.csv), and [training artifacts](../../artifacts/exploratory/nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_20260927/).
