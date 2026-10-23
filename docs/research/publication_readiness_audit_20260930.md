# Publication Readiness Audit (2026-09-30)

This audit evaluates the current evidence against the stated research objective. It is a review artifact, not a new training or test-release authorization.

## Requirement Matrix

| Requirement | Evidence | Status | Boundary |
| --- | --- | --- | --- |
| Add datasets | PEMS03/04/07/08, METR-LA, PEMS-BAY, Weather held-out; ETT, Electricity, Traffic, Exchange-Rate, ILI and Solar validation matrices | Partially satisfied | Results are not uniformly positive; ETT, Exchange-Rate, ILI, Solar and long-horizon coordinates contain explicit failures |
| Add strong model baselines | DLinear, NLinear, RLinear, PatchTST, TimesNet, TimeMixer, iTransformer, Mamba-1, Mamba-2, STGCN, GraphWaveNet, AGCRN, complete validation DCRNN | Partially satisfied | Complete graph held-out comparisons are still absent; all graph additions remain local validation evidence |
| Use published Mamba design signals | Local implementations cover shared-scan MambaTS, VPT/VAST-style context, TimeMachine-style routes, bidirectional and multi-scale probes, Mamba-1/Mamba-2 | Satisfied with scope | Local implementations are not bitwise external reproductions |
| Use AutoCon contrastive and representation learning | Train logs show finite nonzero relation loss; prediction-connected representation fusion appears in frozen traffic releases; active/null controls are available | Satisfied with limitations | Independent relation-loss accuracy gains are small or inconsistent across datasets |
| Match or exceed baselines | PEMS: 28/28 noninferior cells; METR-LA/PEMS-BAY: 12/12; Weather: 12/12; all three frozen releases pass the 1.02 gate | Satisfied only for named releases | This is not universal SOTA evidence; AGCRN validation shows a PEMS-BAY negative boundary |
| Improve computation speed | Candidate is faster than several named nonlinear baselines on fixed RTX 5090 D shapes; Solar versus TimeMixer is a strong validation-only example | Partially satisfied | Candidate is not uniformly faster than DLinear/NLinear/Mamba-1/iTransformer; fixed-shape forward timing is not training throughput |
| Support a high-level paper | Three audited held-out releases, 52-row table, source/plan hashes, negative boundaries and reproducibility tests | Draftable but not proven publishable | Venue fit, external reproduction, full graph held-out comparison, statistical review and author approval remain open |

## Frozen Held-out Evidence

- PEMS H12: `28/28` noninferior cells, `140` candidate paired wins.
- METR-LA/PEMS-BAY H12: `12/12` noninferior cells, `60` candidate paired wins.
- Weather H96/H336/H720: `12/12` noninferior cells, `60` candidate paired wins.
- The 2026-09-30 read-only release audit passes all three bundles; the regenerated 52-row CSV has the same SHA-256 as the previous main table.

## Important Negative Evidence

- AGCRN validation: candidate is approximately tied on METR-LA (`+0.103%`) but worse on PEMS-BAY (`+2.695%`).
- ETTh2 long-horizon TimeMixer comparison fails at H336 (`+4.157%`) and H720 (`+1.438%`), despite lower fixed-shape latency.
- Optimized DCRNN is now complete on two datasets and three paired seeds: candidate improves `5.287%` on METR-LA and `27.444%` on PEMS-BAY, with `3/3` wins in both cells. This remains validation-only and local-implementation evidence.
- A semantics-preserving cache reduces the local DCRNN fixed-shape forward preflight by about `11.9%`; this is a baseline implementation optimization, not evidence for the Mamba candidate.
- The completed DCRNN matrix reports candidate improvements of `5.287%` on METR-LA and `27.444%` on PEMS-BAY, with `3/3` paired wins in both cells; this remains validation-only evidence.
- Manifest-bound METR-LA fixed-shape resources are `.3252 ms` for the candidate versus `9.8622 ms` for optimized DCRNN; this is a comparator- and shape-bounded forward result, not universal speed evidence.
- Current-source high-dimensional H96 resource preflight reports the relation-auxiliary-only candidate at `.80166/.2.20709 ms` on Electricity/Traffic; it is faster than PatchTST/TimesNet but slower than the linear baselines, iTransformer and Mamba2 on both shapes. This reinforces a bounded nonlinear-comparator claim and does not authorize the high-dimensional held-out draft. See [`high-dimensional resource preflight`](../../artifacts/exploratory/current_source_highdim_relation_auxiliary_resource_preflight_20261001/readout.md).
- AutoCon relation-null controls on several datasets show near-zero or inconsistent standalone prediction gains.

## Publication Claims Allowed

The defensible claim is: **under the registered protocols, AutoCon-connected MambaTSv3 is a competitive and often efficient forecasting architecture on the named traffic and Weather held-out releases, with explicit dataset, horizon, comparator and hardware boundaries.**

The following claims are not supported: universal accuracy superiority, universal speed superiority, independent causal benefit of the relation loss across domains, bitwise reproduction of external Mamba repositories, or a complete graph-model held-out ranking.

## Remaining Gates

1. Human decision on whether to authorize the frozen GraphWaveNet/high-dimensional held-out plans.
2. Independent review of source licenses, citations, venue requirements and statistical reporting.
3. If DCRNN is retained in the paper, label it as a local validation-only graph baseline and report the optimized implementation boundary.
4. Do not open additional test runs from validation wins or tune architecture settings after inspecting held-out metrics.

## Verification

- Full regression suite after DCRNN completion, claim binding, numeric consistency audit, integrity audit, MambaTS representation-path optimization, Mamba-2 mixer contract, stride-24 default-path equivalence audit, and dynamic high-dimensional release-reader contracts: `828 passed, 45 warnings`.
- AGCRN validation reader: `12` rows, `2` paired cells, no test metrics.
- AGCRN resource attestation: manifest-bound, fixed-shape RTX 5090 D forward protocol.
- No new held-out metrics or prediction artifacts were generated in this audit.

## Citation Status

The local reference set [`mamba_autocon_references.bib`](references/mamba_autocon_references.bib) contains verified records for AutoCon, MambaTS, TimeMachine, ms-Mamba and the Mamba4Cast software repository. The local citation validator reports five valid entries, no errors and no duplicates. One non-blocking warning remains: the AutoCon ICLR entry does not contain a page field, so authors must resolve the exact venue-style record before submission. Source-by-source checks are documented in [`citation verification notes`](references/citation_verification_20260930.md).

The conservative manuscript claim-binding audit is [`manuscript_claim_binding_audit_20261001.md`](manuscript_claim_binding_audit_20261001.md). It intentionally leaves the draft in human-review status until numeric prose is reconciled line by line with claim IDs and evidence IDs.

The C/E integrity audit resolves the manuscript's used claim IDs and evidence IDs against the local registries and finds no missing local source files. This is a mechanical reachability check, not human verification that each source supports the related claim.

The consolidated local audit is [`submission_readiness_audit_20261001`](../../artifacts/exploratory/submission_readiness_audit_20261001/audit.json). It passes release provenance, DCRNN numerical consistency, C/E file reachability, manuscript binding, citation and completed regression checks, while deliberately reporting `checks_passed_human_approval_required` rather than submission readiness.
