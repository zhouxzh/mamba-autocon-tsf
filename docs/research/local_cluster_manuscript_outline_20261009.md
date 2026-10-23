# Local-Cluster ACF-InfoNCE Manuscript Outline

_Internal evidence-bound writing outline. Not for submission and not a claim of
literature or policy verification._

## Working Scope

**Question.** Under low-label chronological transfer on the named periodic
sensor targets, does train-prefix signed forecast-level ACF InfoNCE improve a
matched Mamba forecaster while retaining a single-scan prediction path?

**Permitted conclusion.** Conditional low-label transfer support on the named
PEMS-BAY, METR-LA, and Weather coordinates only. Effects are reported by
domain, not pooled. [claim:C102] [evidence:E101] [evidence:E103]

**Not permitted.** Universal multivariate forecasting superiority, a universal
Mamba advantage, causal source-selection claims, first-open public-test
claims, or latency superiority over all named external baselines.

## Title And Abstract

### Title candidates for human selection

- Train-Prefix Autocorrelation Contrastive Pretraining for Low-Label Sensor Forecasting
- Signed Forecast-Level Autocorrelation Transfer with a Single-Scan Mamba Forecaster

The title must be checked against the final target venue and cannot imply a
universal method claim.

### Abstract fact inventory

Use only after the accountable author verifies the linked evidence:

- method and deployment path: [claim:C101] [evidence:E101]
- six held-out active/null coordinates: [claim:C102] [evidence:E101]
- independent-host three-domain direction, without pooling: [claim:C109] [evidence:E103] [evidence:E104]
- external-code comparisons must be labelled post-release: [claim:C106] [evidence:E101]
- retained failures and scope limits: [claim:C107] [evidence:E101]

Do not write numerical aggregate gains, p-values, or citations until those
facts have an author-verified source record.

## Introduction

1. State the low-label multivariate forecasting setting and the risk that
   generic representation pretraining may not preserve periodic forecast
   structure. No citation placeholder is filled here.
2. Motivate train-prefix autocorrelation only after authors verify background
   literature and add it to a venue-specific bibliography.
3. State the constrained contribution: an active/exact-null design that holds
   source pool, initialization, target protocol, and deployment path fixed.
   [claim:C101] [evidence:E101]
4. State the paper boundary before presenting results: named periodic sensor
   targets, not universal forecasting. [claim:C107] [evidence:E101]

## Methods

### Protocol

- Describe chronological source/target boundaries, train-prefix ACF fitting,
  source next-patch residual prediction, and active/exact-null separation.
  [claim:C101] [evidence:E101]
- Describe local contrastive clusters and channel-coverage pairing as recorded
  in the evidence package; do not add unverified implementation details.
  [claim:C101] [evidence:E101]
- State that the selected backbone is MambaTSv3 with the Mamba-1 mixer. Mamba-2
  is validation-only ablation material, not a released replacement. The
  provenance audit is [backbone selection audit](../../artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/backbone_selection_audit_20261009.json).

### Evaluation And Integrity

- Define seed as the replication unit and preserve all seed-level rows in
  supplements. Do not call overlapping windows independent samples.
- State that PatchTST extensions use pinned external code and a local
  chronological adapter, then label them post-release. [claim:C106] [evidence:E101]
- State that independent-host evidence is directional and non-pooled across
  target domains. [claim:C109] [evidence:E103] [evidence:E104]

## Results

### Primary Table

Use the six held-out active/exact-null rows exactly as bound in the internal
[results draft](mambatsv3_acf_infonce_local_cluster_results_draft_20261003.md).
[claim:C102] [evidence:E101]

### External And Independent Extensions

- Keep official PatchTST/iTransformer comparisons in a clearly marked
  post-release table. [claim:C106] [evidence:E101]
- Present independent-host active/null replications as three separate rows:
  PEMS-BAY, METR-LA, and Weather. [claim:C109] [evidence:E103]
- Present the independent-host PatchTST extension separately from both primary
  active/null effects and from a cross-domain pooled statistic. [claim:C106] [evidence:E101]

### Deployment And Robustness

- Report the one-scan/no-relation-loss deployment contract and the H336
  latency/memory boundary together. [claim:C105] [evidence:E101]
- Report validation-only robustness as post-confirmation support, not as
  additional confirmatory test evidence. [claim:C104] [evidence:E101]

## Discussion

1. Interpret the active/null contrast as evidence for the stated protocol on
   the named coordinates, not an automatic explanation for all Mamba models.
2. Retain Electricity, Traffic, Weather H192/25%, and PEMS03 gate failures.
   [claim:C107] [evidence:E101]
3. State that compatibility analysis is exploratory and cannot be converted
   into a deterministic source-selection rule. [claim:C108] [evidence:E102]
4. State latency candidly: the model has a clean single-scan path but does not
   establish universal speed superiority. [claim:C105] [evidence:E101]
5. State the historical public-test-access caveat and any final data/code
   access conditions only after authors verify them.

## Displays And Supplements

| Item | Status | Provenance |
| --- | --- | --- |
| Primary active/null result table | Draft from audited rows | [E101](mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md) |
| Independent replication table | Draft, non-pooled | [E103](mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md) |
| External-code comparison table | Supplement candidate, post-release label required | [E101](mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md) |
| Evidence overview figure | Generated internal figure; author review required | `artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/figures/local_cluster_evidence.pdf` |
| Stopped-backbone appendix table | Draft from final decision audit | `final_backbone_selection_20261009.md` |

## Required Human Inputs

- verified venue and current author instructions;
- author list, CRediT roles, funding, conflicts, ethics, data/code and AI-use
  declarations;
- author-verified literature sources and bibliographic metadata;
- completed nine-row claim/evidence worksheet;
- final decision on which post-release extensions belong in the main text versus
  supplementary material.
