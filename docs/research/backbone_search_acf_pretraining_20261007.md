# Backbone Search for ACF Pretraining

This search begins after the user authorized backbone changes. It keeps the
research target intact: train-only ACF representation pretraining, low-label
transfer, strict validation/test separation, and inference without an AutoCon
loss. A persistent hypothesis tree records development probes and fresh-seed
gates separately:
`artifacts/research_state/backbone_search_20261007/.arbor/`.

## Evaluators

The development evaluator is Traffic H96 with 25% chronological target labels,
using validation-only paired MSE. A result can move forward only through a
fresh-seed gate; no held-out test split is opened for search.

## Completed Directions

| Direction | Development evidence | Decision |
| --- | --- | --- |
| Shared channel-independent PatchTST encoder + Channel-Patch SSL | `1.293474` vs forecast `1.287876`, `+0.435%` | Pruned |
| V3 post-scan channel-count-independent FiLM + Channel-Patch | Development seed `-0.413%`, but fresh seeds `+0.339%`, 0/3 vs forecast | Pruned; development gain did not replicate |
| Variable-token Mamba with representation ACF `.05` | Strong forecast base (`0.781710`), ACF `0.781503`, `-0.026%` | ACF gain below gate |
| Variable-token Mamba representation ACF `.10` | `0.782637` vs `.05` `0.782514`, `+0.016%` | Pruned |
| Variable-token Mamba direct representation fusion | `0.781781` vs fusion forecast `0.781720`, `+0.008%` | Pruned |
| Peak-pair CPC with high-ACF negative masking | `0.782264` vs signed relation `0.782556`, `-0.037%`; active anchors 3%--11% | Weak direction only; below gate |
| Multi-positive high-ACF CPC | `0.784187` vs selected CPC `0.784206`, `-0.002%`; coverage increased to 27%--39% | Pruned; broader positives did not change validation accuracy |
| Per-variable ACF token supervision | `0.784886` vs averaged global ACF `0.784012`, `+0.111%`; coverage 17%--24% | Pruned; preserving channel peaks is unfavorable |
| Two-source pretraining, Electricity + Weather | Fresh Traffic seeds `6110--6112`: `0.769143` vs update-matched Electricity replay `0.772766`, `-0.469%`, 3/3 wins, paired 95% ratio upper `0.99905` | Replicated development signal; must isolate the ACF-loss interaction |
| Source diversity x ACF-objective 2x2 | Fresh seeds `6113--6115`: diverse forecast `0.767455`; diverse ACF `0.767660`, `+0.027%`, 0/3 wins, ratio upper `1.00042`; replay ACF effect `-0.063%` | ACF attribution fails; multi-source benefit is a forecast-pretraining effect, not evidence for this ACF loss |
| Task-aligned affine ACF coefficients | Traffic residual-transfer seed `6120`: `0.795736` vs raw-signal coefficients `0.795702`, `+0.004%` | Pruned; correcting a measured METR-LA ACF/task mismatch does not improve transfer |
| Task-aligned signed InfoNCE coefficients | Weather H192 residual-transfer seed `6121`: `4.516603` vs raw-peak signs `4.515470`, `+0.025%` | Pruned; corrected residual signs do not strengthen the weak Weather mechanism signal |
| Official Mamba-3 resource preflight | PEMS-BAY-shaped block: `0.312 ms` vs Mamba-1 `0.154 ms`, `2.03x`; memory `1.40x` | Stopped before training; fails the registered `1.5x` latency gate |

The post-scan FiLM's initial development result was invalidated before
interpretation because the mixer parameter prefix was omitted from the
transfer whitelist. It was corrected and rerun in a separate artifact root;
only the corrected fresh-seed replication appears in this table.

## Current Constraint

Changing the encoder can improve the high-dimensional low-label forecast base,
as demonstrated by the variable-token Mamba result. However, the current
train-only ACF InfoNCE supervision remains nearly neutral or unfavorable even
when it is routed directly into prediction. Future work must change the
supervision problem itself, such as the positive/negative relation semantics
or the forecasting task used for representation pretraining, rather than
adjusting its scalar weight or adding another routing gate.

The source-diversity direction may be retained as a general forecasting
pretraining baseline, but must not be reported as validation of the current
ACF self-supervision mechanism. Any further ACF work requires a new objective
that changes the forecast task or the positive-pair construction, followed by
this same factorial attribution test. It does not open the held-out test
split.

The short-horizon transfer search is therefore closed for the current ACF
statistic family. The evidence-bearing path is the separately audited signed
forecast InfoNCE protocol on ACF-compatible long-horizon traffic and sensor
coordinates; work should focus on its reproducibility package, explicit scope
boundaries, and submission materials rather than treating the H96 probes as
positive evidence.

No result here is a held-out test result, a broad cross-domain claim, or a
publication-readiness determination.
