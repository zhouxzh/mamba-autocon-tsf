# Current-Tree Direct ACF Anchor Fresh-Seed Replication

**Status:** exploratory held-out replication. The test split was read once after validation-based best-epoch restoration; no test metric selected an arm or hyperparameter.

## Paired scaled MSE

| Dataset | seed | off | anchor | relative change | wins |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 2024 | 0.404108 | 0.402495 | -0.399% | yes |
| ETTh1 | 2025 | 0.401606 | 0.398462 | -0.783% | yes |
| ETTh1 | 2026 | 0.406747 | 0.404272 | -0.609% | yes |
| ETTh1 | 2027 | 0.400808 | 0.399341 | -0.366% | yes |
| ETTh1 | 2028 | 0.406238 | 0.403571 | -0.657% | yes |

**ETTh1:** ratio -0.563%, pairwise mean -0.563%, 5/5 wins, paired t p=0.002059, exact one-sided Wilcoxon p=0.03125.

| traffic | 2024 | 0.534524 | 0.526584 | -1.485% | yes |
| traffic | 2025 | 0.542452 | 0.532303 | -1.871% | yes |
| traffic | 2026 | 0.506739 | 0.509312 | 0.508% | no |
| traffic | 2027 | 0.560310 | 0.554945 | -0.957% | yes |
| traffic | 2028 | 0.528496 | 0.523250 | -0.993% | yes |

**traffic:** ratio -0.978%, pairwise mean -0.960%, 4/5 wins, paired t p=0.07195, exact one-sided Wilcoxon p=0.0625.


**Pooled descriptive summary:** ratio -0.799%, pairwise mean -0.761%, 9/10 wins; paired-delta 95% CI [-0.006321, -0.001178]. Pooling different datasets is descriptive, not a universal treatment effect.

## Other metrics and diagnostics

| Metric | ETTh1 ratio | Traffic ratio | pooled ratio |
|---|---:|---:|---:|
| scaled MAE | -0.535% | -1.034% | -0.767% |
| original MSE | -0.835% | -0.849% | -0.835% |
| original MAE | -0.635% | -1.036% | -0.639% |

ETTh1 anchor diagnostics (mean across five test traversals): scale `0.097079`, anchor RMS `0.999292`, valid fraction `1.000000`, strength `0.740250`, lag mean `69.429`.

traffic anchor diagnostics (mean across five test traversals): scale `0.114784`, anchor RMS `1.020599`, valid fraction `1.000000`, strength `0.710588`, lag mean `96.209`.

## Interpretation

The pre-registered descriptive gate passes in both datasets: ETTh1 is 5/5 and Traffic is 4/5, with mean MSE ratios below -0.5% and no paired harm above 1%. This is evidence that the direct anchor is worth a controlled mechanism study, not evidence of universal accuracy or publication readiness.

The branch is active in every treatment checkpoint and the diagnostic support is valid. The ACF sidecar and all non-anchor configuration fields are paired-identical. The anchor itself is a single global scalar over detached per-channel lag selections, so these results do not establish learned channel/horizon coefficients or interaction with AutoCon/relation learning.

Source snapshot: `2837199515e0ee7f...`; plan hash: `d82779df614a32be`.

Evidence boundary: Fresh-seed exploratory held-out replication. Test windows are not inferential replicates; no test metric selected an arm or hyperparameter.
