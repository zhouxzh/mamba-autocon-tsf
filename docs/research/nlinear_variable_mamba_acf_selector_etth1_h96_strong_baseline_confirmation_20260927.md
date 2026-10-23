# Variable-Mamba ACF Selector: ETTh1 H96 Strong-Baseline Confirmation

**Status:** passed the registered validation-only confirmation; no held-out test metric was produced or read.

This is the independent low-dimensional confirmation of the frozen selector that passed the Traffic/H96 strong-baseline screen. It uses fresh seeds `2251--2253`, the ETTh1 training split only for ACF estimation, and the same one-scan `NLinearVariableMambaAutoCon` architecture: a shared NLinear forecast base, variable-axis Mamba-1 scan, prediction-connected AutoCon representation, train-only ACF memory, and zero-initialized rank-2 horizon plus rank-1 channel selector. Selector on/off is the sole candidate graph difference.

All five external baselines explicitly have relation loss, relation lag mask, and relation minimum lag disabled. Candidate-only residual initialization is also phase-local, so baseline commands cannot inherit unsupported Mamba configuration. All 21 jobs completed under the same source snapshot `7755385b888f8c160bf8f1d5c83e6c81814af86c5b1e25dbf07d4fd4b89e28a4`.

## Registered Result

Every run uses ETTh1 multivariate data, `seq_len=96`, `pred_len=96`, eight epochs, the same split and optimizer contract, and `skip_test=true`.

| Model or condition | Mean best validation MSE | Notes |
| --- | ---: | --- |
| Selector off | 0.651147 | Exact paired control |
| Selector on | 0.647344 | 3/3 paired seed wins; finite nonzero selector scales |
| DLinear | 0.650119 | Strongest external baseline in this screen |
| iTransformer | See row-level readout | Same budget |
| PatchTST | See row-level readout | Same budget |
| TimesNet | See row-level readout | Same budget |
| NLinear | See row-level readout | Same budget |

Selector-on improves the exact control by `0.584%` and DLinear by `0.427%`. The strict reader verified full `(model, seed)` coverage, source-snapshot consistency, training completion, absence of test artifacts, ETTh1 legal train-only relation lags, ACF sidecars, and finite nonzero selector weights.

## Exact-Shape Forward Resources

An RTX 5090 D FP32 device-resident prediction-only measurement (`B=32`, 50 warm-up and 200 timed forwards) reports selector-on median latency `0.441 ms`; selector-off is `0.371 ms`. The selector is much faster than TimesNet (`2.954 ms`), but slower than PatchTST (`0.291 ms`), iTransformer (`0.166 ms`), DLinear (`0.068 ms`) and NLinear (`0.031 ms`). Therefore ETTh1 provides no selector speed-advantage claim. The measurement excludes data loading, ACF fitting, training and AutoCon relation-loss backward time.

## Boundary And Next Gate

Together with Traffic, this supports a two-dataset H96 validation signal for the frozen selector. It is still not a high-level-paper result: it has one prediction horizon, short common budget, and no held-out evaluation. The next experiment must either confirm a third dataset at H96 or use a registered long-context protocol that makes train-only ACF lags accessible. A held-out test release remains prohibited until that validation gate passes.

Exact artifacts: [plan](nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_20260927.json), [launch authorization](nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_20260927_launch_sidecar.json), [strict readout](nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_readout_20260927/readout.md), [row-level validation data](nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_readout_20260927/validation_rows.csv), [resource records](../../artifacts/exploratory/nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_20260927/resources/model_resources.json), and [training artifacts](../../artifacts/exploratory/nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_20260927/).
