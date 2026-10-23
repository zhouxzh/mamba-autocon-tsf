# Submission Handoff Checklist

_This is a human-review handoff, not a submission-ready declaration._

---

## Evidence package

| Item | Local evidence | Status |
| --- | --- | --- |
| Held-out main table | [66-cell table](../../artifacts/exploratory/paper_main_table_20261001/main_table.md) | Complete; 65/66 non-inferior cells |
| Release provenance | [Four-release audit](../../artifacts/exploratory/paper_release_bundle_audit_20261001/audit.md) | Complete; all four bundles pass |
| Failure boundaries | [Final evidence boundary](final_evidence_boundary_20261001.md) | Complete; Traffic/iTransformer, LatentTemporalMamba, GraphWaveNet thermal stop retained |
| Manuscript evidence links | [Internal draft](manuscript_draft_20260928.md) | Complete; 45/45 numeric lines bound |
| Claim and evidence map | [Claim registry](claim_evidence_registry_20261001.md) | Complete; 13 claims and 57 evidence records have local paths |
| Software regression | [Regression log](../../artifacts/exploratory/submission_readiness_audit_20261001/pytest_20261001.log) | Complete; 832 passed, 45 warnings |
| Citation syntax | [Submission audit](../../artifacts/exploratory/submission_readiness_audit_20261001/audit.json) | Complete with one citation warning |

## Author actions required

| Item | Required human action | Status |
| --- | --- | --- |
| Target venue | Select a journal/conference and verify current scope, template, anonymity, data/code, and AI-use policies | Unresolved |
| Author information | Confirm authorship, order, affiliations, CRediT roles, corresponding author, and acknowledgments | Unresolved |
| Declarations | Supply verified funding, conflicts, ethics, data/code availability, and AI-use disclosures | Unresolved |
| Statistical interpretation | Confirm seed-level uncertainty wording, non-inferiority margin rationale, and multiplicity treatment | Unresolved |
| Citation review | Resolve the AutoCon bibliography warning and verify every cited work's final metadata, license, and relevance | Unresolved |
| Displays | Produce venue-formatted tables/figures from the audited main table without omitting failure cells | Unresolved |
| Final approval | Read the complete manuscript and approve every claim, limitation, table, figure, and declaration | Unresolved |

## Prohibited shortcuts

- Do not report the stopped GraphWaveNet candidate test artifacts.
- Do not promote LatentTemporalMamba's lower forward latency into an accuracy claim.
- Do not remove the Traffic/iTransformer failure from the Electricity/Traffic table.
- Do not write universal accuracy, universal speed, or cross-domain causal AutoCon claims.
- Do not mark the manuscript submission-ready before the author actions above are complete.

## Current conclusion

The project has a reproducible, evidence-bound manuscript package for four named held-out releases. It is suitable for disciplined manuscript revision and venue selection, not for automatic submission or a guaranteed publication outcome.
