# Existing Anchor Checkpoint Test-Only Audit

**Status:** post-hoc held-out audit, not an independent replication. The 12
checkpoints were trained in the earlier validation-only pilot; no weights or
epochs were selected from these test metrics.

## Protocol and provenance

Each existing anchor-off/anchor-on checkpoint was copied to an independent
audit directory and evaluated once on the test split with `stream_metrics=true`,
`save_predictions=false`, and `stream_sample_windows=1`. The evaluation used a
fresh read-only source snapshot (`source_tree_sha256=4c9461...f7ff`, recorded in
the audit provenance) and copied train-only ACF sidecars. Because the original
training manifest was created with `provenance_id=dry-run` and did not record
the training source hash, this cannot establish exact training-time source
identity. It is a continuity check, not confirmatory evidence.

## Paired test metrics

| Dataset | seed | no anchor MSE | anchor MSE | relative change | no anchor MAE | anchor MAE | relative change |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.419727 | 0.413673 | -1.442% | 0.425608 | 0.421323 | -1.007% |
| ETTh1 | 2022 | 0.417995 | 0.415287 | -0.648% | 0.430052 | 0.427344 | -0.630% |
| ETTh1 | 2023 | 0.402350 | 0.401280 | -0.266% | 0.415626 | 0.414161 | -0.352% |
| Traffic | 2021 | 0.528908 | 0.523504 | -1.022% | 0.360084 | 0.357252 | -0.787% |
| Traffic | 2022 | 0.541417 | 0.535552 | -1.083% | 0.376472 | 0.371073 | -1.434% |
| Traffic | 2023 | 0.540751 | 0.531534 | -1.704% | 0.365540 | 0.365247 | -0.081% |

Anchor-on wins all 6 paired MSE coordinates. The ratio of pooled MSE means is
`-1.063%` (0.470138 versus 0.475191); the mean of the six pairwise relative
changes is `-1.028%`. By dataset, the MSE ratio is ETTh1 `-0.793%` and Traffic
`-1.272%`. A paired t-test over six coordinates gives
`p=0.0074` for the raw MSE differences and a one-sided exact Wilcoxon result is
`p=0.015625`; both are descriptive and do not repair the provenance limitation
or the fact that the weights came from the earlier pilot.

## Diagnostic limitation

The plain forecasting experiment did not persist model ACF diagnostics in this
post-hoc audit, so no test-time anchor RMS/valid-fraction series are available
here. The checkpoint state confirms that every anchor treatment has a nonzero
learned `acf_anchor_scale`, but that is an implementation check rather than a
mechanistic explanation of the test gain.

## Decision

The test direction is encouraging enough to justify the pre-registered fresh
seed replication in [`mamba_acf_anchor_current_test_replication_plan_20260904.json`](mamba_acf_anchor_current_test_replication_plan_20260904.json),
but it does not make the paper ready. The fresh matrix must use new seeds,
frozen source/data/environment hashes, validation-only best-epoch selection,
and one untouched test evaluation per arm. Even if reproduced, the result
supports only this plain MambaACF-variable anchor mechanism; it cannot be
extrapolated to AutoCon, soft-memory interaction, Mamba-1, other horizons, or
SOTA claims without separate registered experiments.
