# High-Dimensional Held-out Review

## Decision Requested

Review whether to authorize the exact high-dimensional held-out plan:

- Plan: [`mambatsv3_relation_auxiliary_highdim_test_release_draft_20260928.json`](mambatsv3_relation_auxiliary_highdim_test_release_draft_20260928.json)
- SHA-256: `6539139c30f9ceaba869e0d1afb8005623b653a86938273b697977a0e4d02614`
- Status: the stride-8 plan reviewed here remains `execution.authorized=false`; no stride-8 test task has started.
- Scope: Electricity and Traffic, multivariate H96, five fresh paired seeds `3311--3315`.
- Workload: 6 validation reproducibility jobs, 10 candidate held-out jobs, 70 held-out baseline jobs; total 86.

Approval authorizes only this exact file hash and only the registered tasks. It does not authorize architecture changes, tuning, additional datasets, retries, or any post-hoc modification to comparators.

## Frozen Candidate

`MambaTSv3AutoCon` uses a single Mamba-1 patch scan, `MeanStd` window normalization,
`d_model=32`, `d_ff=64`, `patch_len=stride=16`, and `AutoCon_lambda=.001`.
The train-only AutoCon contrastive representation objective is enabled; the prediction-connected
representation fusion flag is disabled. This choice was made before the held-out draft because the
exact high-dimensional mechanism control found fusion-on worse than fusion-off by `1.160%` on
Electricity and `1.081%` on Traffic, whereas active AutoCon relative to lambda-zero had small
validation gains of `0.376%` and `0.236%`.

## Validation Evidence

The candidate has a completed same-source, validation-only 48-row matrix against DLinear,
NLinear, RLinear, PatchTST, TimesNet, iTransformer, and Mamba2:

| Dataset | Candidate MSE | Strongest baseline | Change | Paired wins |
| --- | ---: | --- | ---: | ---: |
| Electricity H96 | `.158235` | TimesNet `.168558` | `-6.117%` | `3/3` vs every baseline |
| Traffic H96 | `.372924` | iTransformer `.401117` | `-7.029%` | `3/3` vs every baseline |

The train-only representation objective has a separate audit across 12 candidate runs: finite
nonzero AutoCon loss, fixed lambda, and exact recorded loss identity. See
[`training audit`](../../artifacts/exploratory/relation_auxiliary_autocon_training_audit_20260929/readout/readout.md).

## Efficiency Boundary

On RTX 5090 D, FP32, device-resident `B32,L96,H96` prediction-only inputs, the candidate has
median latency `.8011 ms` on Electricity and `2.2052 ms` on Traffic. It is faster than PatchTST
and TimesNet in those shapes, but slower than DLinear, NLinear, RLinear, iTransformer, and Mamba2.
The held-out plan therefore must not be used to claim universal efficiency, even if accuracy passes.

The independent Weather resource audit similarly finds the candidate faster than PatchTST, TimesNet,
and Mamba2 but slower than RLinear and iTransformer. ETTh1 remains an explicit accuracy failure
boundary, so it is excluded from this held-out proposal.

## Release Reader

[`analyze_mambatsv3_relation_auxiliary_highdim_test_release.py`](../../scripts/analyze_mambatsv3_relation_auxiliary_highdim_test_release.py)
will reject a release unless all phases have an enforced authorization binding, the approved plan
hash matches, source snapshots are identical, all 80 confirmation coordinates are present, candidate
and baseline settings remain isolated, and streaming metrics use the registered aggregation. It reports
one-sided paired 95% upper MSE ratios against the predeclared `1.02` non-inferiority margin.

## Remaining Risks

- Validation results may not reproduce on untouched test windows.
- Five seeds give limited uncertainty resolution.
- The candidate is not faster than the strongest linear or iTransformer comparators.
- The method is not universal: ETTh1 is a reproducible failure boundary.
- A passing release would support Electricity/Traffic H96 only, not causal superiority of AutoCon or all time-series domains.

## Candidate tradeoff note

An independently completed stride-24 relation-auxiliary matrix provides a faster alternative: it remains `3/3` against every registered comparator in both domains, but validation MSE is `3.494%/3.135%` higher than the stride-8 candidate while current-source fixed-shape latency is `51.529%/61.986%` lower. This alternative is not part of the present stride-8 draft. Its parallel plan is now authorized as [`stride-24 test plan`](mambatsv3_stride24_relation_auxiliary_highdim_test_release_draft_20261001.json), SHA-256 `49ff3b8ce004487a4e1b88c0463a1a06f39d23b71af034b4f321595a3250aa98`, through a hash-bound [authorization sidecar](mambatsv3_stride24_relation_auxiliary_highdim_test_release_20261001_launch_authorization.json) with GPU thermal/fan guardrails. See the [candidate tradeoff review](highdim_candidate_tradeoff_review_20261001.md).
