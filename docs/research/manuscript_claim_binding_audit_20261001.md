# Manuscript Claim-Binding Audit

This report is generated from `scripts/audit_manuscript_claim_bindings.py`. It is a conservative writing-integrity check, not a scientific correctness certificate. Human authors must still open every source and confirm each claim.

Current audit result: `42` manuscript lines contain numeric-looking claims; all `42` have an inline local evidence link or `C`/`E` identifier; `0` machine-detected numeric lines remain unresolved. The machine-readable audit is [`audit.json`](../../artifacts/exploratory/manuscript_claim_binding_audit_20261001/audit.json).

The manuscript remains an internal draft. Machine-detectable numeric prose is now annotated with claim IDs and evidence IDs, but submission still requires a human reconciliation pass: each verifier must open the artifacts, confirm the proposition, record a date, select a venue, and approve authorship and declarations.

## Priority Review Map

| Draft section | Claim registry | Local evidence IDs / sources | Required human check |
|---|---|---|---|
| Abstract, lines 11--14 | C001, C002, C003, C009 | E001--E005 | Confirm all held-out denominators, margins, fusion deltas, and scope wording |
| Data and protocols, lines 40--52 | C001--C003, C008 | E001--E005; release audits | Confirm seed counts, metric definition, and that latency wording remains prediction-only |
| Validation boundaries, lines 64--108 | C004--C006, C009 | E010--E052 | Confirm each validation-only label and retain every failure boundary |
| Graph baselines, lines 112--118 | C007, C008 | E053, E054 and graph readouts | Confirm local canonical-style wording, paired denominators, and fixed-shape resource scope |
| Limitations, lines 122--145 | C009, C010 | publication readiness audit; external registry | Confirm no unresolved gate is presented as completed |
| Evidence registry, lines 152 onward | all | E001--E054 | Open every linked artifact, record verifier and date, then resolve each numeric paragraph |

The audit tool may mark machine-readable bindings as `bound`, but it may never mark the manuscript submission-ready. An accountable human must still record the verifier, date, and final venue scope for every numeric manuscript claim.

The DCRNN numeric cross-check is [`dcrnn_evidence_consistency_audit_20261001`](../../artifacts/exploratory/dcrnn_evidence_consistency_audit_20261001/audit.json). It verifies that strict readout and resource CSV values match the numbers intentionally reported in the manuscript (paired deltas, wins, and fixed-shape latency), without requiring duplicated raw MSE tables in prose.

The end-to-end identifier and file-reachability check is [`claim_evidence_integrity_audit_20261001`](../../artifacts/exploratory/claim_evidence_integrity_audit_20261001/audit.json). It resolves all `10` used claim IDs and `54` used evidence IDs with no missing local evidence files. Like the other automated checks, it does not assess scientific correctness or replace a human source review.
