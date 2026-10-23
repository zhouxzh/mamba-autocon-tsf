# Variable-Mamba ACF Selector: Traffic H96 Strong-Baseline Screen

**Status:** passed the registered validation-only screen; no held-out test metric was produced or read.

The first strong-baseline launch on 2026-09-27 is retained as an invalid predecessor: all external-baseline commands inherited a nonzero relation-lag option while relation loss was disabled, so they exited in argument validation. Its candidate rows are not used here. The replacement plan uses fresh seeds `2241--2243`, explicitly disables all candidate-only relation options for external baselines, and adds a plan-runner guard against this configuration error.

The frozen candidate is `NLinearVariableMambaAutoCon`: a shared NLinear forecast base, one Mamba-1 scan over variable tokens, a prediction-connected AutoCon representation, train-only ACF memory, and a zero-initialized rank-2 horizon plus rank-1 channel selector. The selector is the only graph difference between selector-off and selector-on; it adds no Mamba scan.

## Registered Result

All 21 validation jobs completed successfully under source snapshot `7755385b888f8c160bf8f1d5c83e6c81814af86c5b1e25dbf07d4fd4b89e28a4`. Every job used Traffic multivariate data, `seq_len=96`, `pred_len=96`, eight epochs, the same split and optimizer contract, and `skip_test=true`.

| Model or condition | Mean best validation MSE | Notes |
| --- | ---: | --- |
| Selector off | 0.397010 | Exact paired control |
| Selector on | 0.392141 | 3/3 paired seed wins; finite nonzero selector scales |
| iTransformer | 0.399676 | Strongest external baseline in this screen |
| PatchTST | 0.407419 | Same budget |
| TimesNet | 0.434185 | Same budget |
| NLinear | 0.506532 | Same budget |
| DLinear | 0.512167 | Same budget |

Selector-on improved the exact control by `1.226%` and the strongest observed external baseline (iTransformer) by `1.885%`. The strict reader also verified complete `(model, seed)` coverage, stable per-phase source snapshots, training completion markers, the absence of test artifacts, legal train-only relation lags, ACF sidecars, and active selector weights.

## Boundary And Next Gate

This is a positive screening result, not a paper result: it covers one high-dimensional dataset, one horizon, three seeds, and a short common budget. It supports a separately registered cross-dataset or multi-horizon validation confirmation with the architecture and hyperparameters frozen. It does not support a held-out-test release, a universal baseline claim, or an AutoCon relation-loss attribution claim.

Exact artifacts: [plan](nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_20260927.json), [launch authorization](nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_20260927_launch_sidecar.json), [strict readout](nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_readout_20260927/readout.md), [row-level validation data](nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_readout_20260927/validation_rows.csv), and [training artifacts](../../artifacts/exploratory/nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_20260927/).
