# NLinear Baseline-Gap Screen Review

**Status:** `exploratory_nlinear_baseline_gap_screen_draft`; not authorized for training.

**Plan:** [`mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json`](./mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json)

## What This Closes

The current benchmark audit identifies `NLinear` as absent from the registered
model lists of the completed corrected and high-dimensional confirmatory
matrices. This is a design omission, not a failed NLinear job: both completed
matrices enumerate DLinear, PatchTST, TimesNet, AutoConNet, Mamba-1, and
Mamba-2 variants, but not NLinear.

There are four discoverable historical NLinear result records:

| Artifact | Dataset / horizon / seed | Budget | Provenance status | Reusable here? |
|---|---|---|---|---|
| `artifacts/paper_final/mamba_acf_v2_baseline_screen_M/...` | ETTh1, Weather, Exchange-Rate / H96 / 2021 | 3 epochs, batch 32 | registered exploratory plan and immutable source snapshot; no phase-validation record | No |
| `artifacts/extension_pilot_20260831/traffic_h96/nlinear/...` | Traffic / H96 / 2021 | 3 epochs, batch 16 | no registered plan, plan hash, or immutable source snapshot | No |

The v2 records are useful historical traces, including one ETTh1 point, but
cover neither Traffic nor the other two paired seeds. The Traffic pilot has no
ETTh1 counterpart and a different batch size. All four use a 3-epoch budget,
which differs from the current short screen (8 epochs, batch 32). Their scaled
MSE values must not be inserted into any current table.

## Existing Plans Assessed

| Plan | Contains NLinear? | Relevant coverage | Why it does not solve this exact gap |
|---|---:|---|---|
| `mamba_acf_fair_baseline_screen_plan.json` | No | ETTh1/Traffic, H96, seeds 2021--2023, 8 epochs | Its four plain baselines omit NLinear; their artifacts carry an earlier immutable source snapshot. |
| `mamba_acf_v2_benchmark_plan.json` | Yes | ETTh1/Weather/Exchange-Rate, H96, seed 2021, 3 epochs | Completed exploratory one-seed screen; it lacks Traffic and does not match the 8-epoch target protocol. |
| `mamba_acf_confirmation_plan.json` | Yes | ETTh1/ETTh2/Electricity/Traffic, 50 epochs | Draft with no completed jobs; the full matrix is 884 jobs, not a minimal gap screen. |
| `extension_baseline_plan.json` | Yes | Electricity/Traffic, H96/H336/H720, 5 seeds | Registered 180-job extension with no completed metrics; no ETTh1 coverage. |
| corrected/highdim historical plans | No | completed 5-seed matrices | NLinear is omitted in the phase model lists, so the audit correctly reports it missing despite 100% coverage of the reduced matrix. |

## Proposed Matrix

The plan contains exactly six test-evaluated jobs:

| Model | Datasets | Horizon | Seeds | Budget |
|---|---|---:|---|---|
| NLinear | ETTh1, Traffic | 96 | 2021, 2022, 2023 | 8 epochs, patience 3, batch 32 |

It fixes all data, generic forecasting hyperparameters, optimizer/scheduler,
streaming metric settings, and SHA-256 dataset identities used by the earlier
short fair screen. It uses a fresh root:
`artifacts/paper_final/mamba_acf_nlinear_baseline_gap_screen_20260902/nlinear_h96`.
No result directory has been created and this review does not authorize one.

## Provenance and Interpretation Rules

At launch, the runner will require an approved sidecar bound to the exact plan
hash and its declared human-review gates. It will then make an immutable source
snapshot, retain package/environment data, bind dataset hashes, and validate
the completed phase. A changed plan, source tree, dataset, or output root must
start a new phase rather than overwrite an existing one.

This narrow screen does **not** retroactively make the completed
`mamba_acf_fair_baseline_screen_20260901` matrix a valid five-model paired
comparison. Its four completed baselines have a different source snapshot and
plan hash. The new NLinear numbers may be reported only as a standalone,
descriptive record. A legitimate common-provenance five-model comparison
requires a separately reviewed fresh plan that reruns NLinear, DLinear,
PatchTST, Mamba1, and Mamba2 together.

Likewise, six short-budget jobs cannot repair the 50-epoch five-seed
corrected/highdim matrices or support a full baseline, architecture, speed, or
publication claim.

## Human Review Before Any Launch

- Confirm the narrow standalone result is worth the compute cost given it will
  not be merged with historical arms.
- Recheck the two CSV hashes and confirm that the output root is absent.
- Freeze this JSON, create an approved launch sidecar with evidence hashes for
  every prelaunch gate, and retain it in phase provenance.
- Do not use test MSE/MAE to select a model or trigger an unregistered follow-up
  configuration.

The runner's `--validate_only` check is required before authorization; a
successful validation only confirms plan expansion and static contracts, not
that training has been approved or performed.
