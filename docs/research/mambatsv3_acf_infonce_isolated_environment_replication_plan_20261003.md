# Isolated-Environment Replication Plan

## Scope

This document records two same-machine isolated-environment runs.  Neither is
described as an independent-machine replication and both remain separate from
the frozen three-seed release.

The first attempted direct replication used the historical union of all
channel-level ACF peaks when assembling small InfoNCE batches.  Its observed
active-positive fraction was only 0.06--0.12, versus about 0.93 in the frozen
release metadata.  It is therefore an invalid diagnostic run, receives no
fine-tuning or test evaluation, and cannot verify the release.

The second run is a new mechanism candidate: it derives source pair lags from
the same train-only channel ACF table but retains only lags supported by at
least 50% of source channels.  It uses a new paired seed and a validation-only
lifecycle.  This changes training sampling, so it is an optimization candidate
and not a direct reproduction of the frozen method.

## Fixed Protocol

| Item | Value |
| --- | --- |
| Target | PEMS-BAY, chronological 60/20/20 split |
| Label prefix | 10% of the target training partition |
| Source domains | PEMS07, PEMS08, PEMS04 |
| Source selection | Existing train-prefix ACF ranking; no test access |
| Input / prediction length | 96 / 192 |
| Shared backbone | Frozen one-scan channel-agnostic MambaTSv3 core |
| Source objective | Next-patch residual prediction; forecast-level signed ACF InfoNCE |
| Active condition | `mamba_lambda=0.1`, `relation_loss_fraction=0.1` |
| Exact-null control | Identical pipeline with `mamba_lambda=0` and `relation_loss_fraction=0` |
| Source / target epochs | 3 / 5 |
| Source / target maximum windows | 64 / 64 |
| Candidate paired seed | 5502 |
| Interpreter | `/home/zhong/anaconda3/envs/acf_infonce_repro_20261003/bin/python` |
| Resource guard | 4 CPU threads; GPU temperature <= 78C; fan <= 85% |

## Data and Source Integrity

The data and initialization hashes must equal the hashes recorded in
`artifacts/paper_final/mambatsv3_residual_autocon_expansion_selection_20261003/pems_bay_sources.json`
and `artifacts/exploratory/mambatsv3_acf_infonce_h192_screen_20261003/pems_bay_h192_s5410_active_pretrain.json`.
The run also records the current hashes of the pretraining, fine-tuning, model,
loss, sampler, and target-data files in `manifest.json`.

## Lifecycle and Decision Rule

1. Train active and exact-null source checkpoints with the identical
   `channel_coverage` source batch sampler, then fine-tune target
   checkpoints with no `--evaluate-test` flag.  The saved checkpoints contain
   validation MSE only.
2. Open the two validation readouts exactly once.  Run the read-only test
   evaluator for both checkpoints only when the active validation MSE is no
   larger than the null validation MSE.
3. If the validation rule fails, record a negative replication and do not read
   the test partition.  No hyperparameter, architecture, source-pool, or seed
   change follows from this single run.

## Interpretation Boundary

The invalid direct run tests neither execution nor the released effect.  The
candidate tests whether positive-support-aware sampling makes the intended
forecast InfoNCE mechanism active.  One fresh candidate seed cannot alter the
released effect estimate, establish a new SOTA claim, or be pooled into the
predeclared three-seed table.
