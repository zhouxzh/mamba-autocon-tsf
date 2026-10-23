# ACF Source-Selection Ablation: 2026-10-03

## Frozen Contract

PEMS-BAY H192 target adaptation uses the same NLinear primary, H192 exact
head, active signed forecast-level InfoNCE, three fresh seeds, and identical
target budget. Only the source pretraining pool changes:

- `selected`: ACF top-3 `PEMS07/PEMS08/PEMS04`, similarities `.880/.860/.858`.
- `counterfactual`: heterogeneous `Traffic/Weather/Electricity`, similarities
  `.056/.610/.192` in the earlier few-shot-only ranking pool.

No target validation or test row was used to form either pool. Both arms use
the current pair-preserving default (`legacy`) source-window contract.

## Validation Result

| Source pool | Mean H192 validation MSE | Relative to counterfactual | Wins |
| --- | ---: | ---: | ---: |
| ACF selected top-3 | 2.780616 | +2.032% | 0/3 |
| Heterogeneous counterfactual | 2.725234 | - | - |

This is a negative mechanism result for ACF-signature source selection. It
does not invalidate the main PEMS-BAY/METR-LA/Weather held-out releases, which
use their own frozen source-selection artifacts; it invalidates only the claim
that the current ACF similarity ranking is itself a reliable transfer rule.

## Consequence

Do not tune source count, source combinations, or ranking metrics against the
target validation/test blocks. The paper should describe source selection as a
train-prefix heuristic, report this negative sensitivity, and avoid attributing
the forecasting gains to source ranking. The reproducible positive claim is
the shared residual pretraining plus signed InfoNCE mechanism under the frozen
release source contracts.

Artifacts: `artifacts/exploratory/mambatsv3_acf_source_selection_ablation_h192_20261003/`.
