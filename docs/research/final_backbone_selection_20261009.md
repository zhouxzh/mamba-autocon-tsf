# Final Backbone Selection

## Decision

The paper-facing local-cluster ACF-InfoNCE line uses the channel-count-
agnostic, one-scan **Mamba-1** MambaTSv3 backbone. It is the only candidate
that currently has all of the following under one bounded evidence line:

- three-domain held-out active versus exact-null confirmations;
- independent-host three-domain active/null replication;
- an independent-host pinned official PatchTST extension on the same three
  target domains;
- prediction-only deployment with one Mamba scan, no returned contrastive
  representation, and no relation loss.

The machine-readable decision audit is
[`backbone_selection_audit_20261009.json`](../../artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/backbone_selection_audit_20261009.json).
It verifies evidence boundaries only; it does not certify scientific claims or
submission readiness.

## Evidence Summary

| Family | Status | Evidence-based reason |
| --- | --- | --- |
| MambaTSv3, Mamba-1 | Selected | Released low-label evidence line: local held-out confirmations, independent-host active/null replication, and independent official PatchTST extension all retain separate three-domain support. |
| Mamba-2 two-head shared scan | Ablation only | Validation-only results can be competitive, but no held-out release was opened. It is slower than the Mamba-1 release path and cannot silently replace it. |
| MambaTSv13 lightweight multi-scale | Stopped | The registered paired promotion screen did not clear its v3 comparison gate; larger validation matrices also fail cross-domain stopping rules. |
| MambaTSv14 local expert | Stopped before training | Resource preflight was slower and used more memory than the matched v3 control. |
| MambaTSv15/v16 calendar-conditioned variants | Stopped before training | Both exceeded their pre-registered latency limit in matched resource preflights. |
| Official Mamba-3 | Stopped before training | Official-block preflight measured a `2.03x` Mamba-1 latency ratio, above the `1.5x` gate. |

## Reporting Boundary

Mamba-2 may appear as a validation-only backbone ablation. The stopped families
may appear in an appendix or model-selection table with their negative/resource
boundaries. None may be reported as a released replacement for the selected
Mamba-1 local-cluster model.

This decision does not broaden the result beyond the named periodic sensor
targets, does not pool effects across domains, and does not remove the pending
author, venue, declaration, or historical test-access review requirements.
