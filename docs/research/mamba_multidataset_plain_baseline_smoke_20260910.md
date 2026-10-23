# Multidataset plain-baseline execution smoke

## Scope

This is a bounded execution smoke derived from the registered six-model plain-baseline expansion. It uses a fresh output root and runs the first 24 deterministically ordered jobs from the 216-job plan. The intended full matrix is ETTh1, ETTm1, Electricity, and Traffic; M-feature input; horizons 96, 336, and 720; and seeds 2021, 2022, and 2023 for DLinear, NLinear, PatchTST, TimesNet, Mamba1, and Mamba2.

Plan: [`mamba_multidataset_plain_baseline_smoke_plan_20260910.json`](mamba_multidataset_plain_baseline_smoke_plan_20260910.json)

Artifact root: `artifacts/exploratory/mamba_multidataset_plain_baseline_smoke_20260910`

Audit: `artifacts/audit_multidataset_plain_baseline_smoke_20260910`

## Execution result

All 24/24 sampled jobs completed successfully. The sample contains:

| Model | Jobs |
| --- | ---: |
| DLinear | 9 |
| NLinear | 3 |
| PatchTST | 4 |
| TimesNet | 3 |
| Mamba1 | 4 |
| Mamba2 | 1 |

Dataset coverage in the sample is Traffic 7, ETTh1 6, Electricity 6, and ETTm1 5. Horizon coverage is H96 6, H336 7, and H720 11. Every job used `skip_test=true`; no test metric was produced or used for selection.

## Resource observation

The 24 jobs used a total of `1290.6s`, with mean `53.8s` and median `33.7s`. High-dimensional long-horizon jobs dominate the cost: observed examples include Electricity/TimesNet/H720 at `161.7s`, Traffic/PatchTST/H720 at `121.1s`, and Traffic/DLinear/H720 at `107.9s`. The sample is therefore evidence that the unified data/model path executes, not a reliable estimate for every coordinate.

The full 216-job plan is still unauthorized and remains a separate human-review decision. A rough serial estimate from this smoke is several hours; it should be launched only after a resource budget and stop policy are approved.

## Interpretation boundary

This smoke is not a baseline comparison and must not be pooled with the historical five-seed matrices. Because test evaluation is disabled and the sample is not balanced across coordinates, it cannot support accuracy ranking, non-inferiority, speed, or publication claims. Its purpose is to catch missing data contracts, model import failures, high-dimensional memory failures, and unexpectedly expensive jobs before a larger run.

The complete manifest is `plain_baselines/logs/job_manifest.jsonl`; the read-only audit correctly reports 24 config-only/validation-complete rows and zero publishable metric artifacts under the test-withheld policy.

## Manual review checklist

- [ ] Confirm the fresh smoke output root is separate from the registered 216-job expansion.
- [ ] Confirm all six model names and four dataset contracts before approving the full matrix.
- [ ] Decide whether the serial runtime is acceptable or whether the plan should be split into independently frozen phases.
- [ ] Keep test metrics withheld until the baseline matrix, selection rule, and resource protocol are frozen.

