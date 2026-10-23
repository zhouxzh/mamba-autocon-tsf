# Final Evidence Boundary (2026-10-01)

_Human-review summary after CPU-4 high-dimensional validation, LatentTemporalMamba screening, and thermal-protected GraphWaveNet stop._

---

## 📈 Results usable in the main paper

The audited main table contains 66 held-out candidate-baseline cells across four complete releases; 65 satisfy their registered non-inferiority condition. The remaining cell is the required Traffic/iTransformer failure.

| Evidence | Status | Permitted use |
| --- | --- | --- |
| PEMS03/04/07/08 H12 held-out | Complete, `28/28` non-inferior cells | Main accuracy table, scoped to four traffic datasets and H12 |
| METR-LA/PEMS-BAY H12 fusion held-out | Complete, `12/12` non-inferior cells | Main table for non-graph traffic and prediction-connected AutoCon fusion |
| Weather H96/H336/H720 held-out | Complete, `12/12` non-inferior cells | Main table for Weather and registered horizons |
| Electricity/Traffic H96 CPU-4 held-out | Complete, `13/14` non-inferior cells | Main table with the Traffic/iTransformer failure cell retained |
| GraphWaveNet held-out extension | Stopped at thermal protection before comparator completion | Do not report candidate test metrics or partial comparison |
| LatentTemporalMamba H96 validation | Complete negative result, `0/3` active-vs-null wins on both datasets | Negative architecture/AutoCon ablation; no held-out release |
| Cross-domain AutoCon pretraining | Final physical-interval-conditioned predictive lag-pair Traffic 10% pilot remains `+0.644%` above random initialization | Negative pretraining mechanism; 898-unit matrix stopped |

## ⚠️ Claim limits

The strongest defensible paper claim is that the registered AutoCon-connected MambaTSv3 variants are competitive on the named held-out datasets and horizons, with comparator-specific fixed-shape forward advantages and explicit failures. The evidence does not support universal accuracy superiority, universal speed superiority, an independent causal gain from the AutoCon relation loss across domains, or a complete graph-model held-out ranking.

The CPU-4 Electricity/Traffic release is especially important for honest reporting: the candidate passes 13/14 registered cells but loses to Traffic/iTransformer by `2.154%`. The LatentTemporalMamba screen is also a required negative result: its representation-inclusive forward path is faster, but its active AutoCon accuracy is worse than the exact null by `10.186%` and `6.343%`. The new cross-domain pretraining line verified that random batches left the global relation objective inactive, while ACF lag-pair batching activated more than 94% of global relation anchors. Even after channel-count agnostic context and prediction-aligned next-patch pretraining, the Traffic 10% adaptation pilot remained worse than random initialization, so it does not support a low-data transfer claim.

## 🌡️ Resource boundary

The CPU-4 high-dimensional release peaked at `57C` and `35%` fan. The GraphWaveNet extension was proactively stopped at `75C` and `68%` fan, before the `78C/85%` hard guard, and the GPU returned to idle. No active training process remains.

## ✅ Audit state

The latest read-only submission audit reports `checks_passed_human_approval_required`; the four-release bundle audit passes and the complete regression suite is `842 passed, 45 warnings`. Human review remains required for venue choice, statistical interpretation, citation policy, and final manuscript approval.
