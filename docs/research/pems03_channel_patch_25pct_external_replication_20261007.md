# PEMS03 Target-Excluded Channel-Patch 25% Replication

This registered validation-only H192 replication extends the strict PEMS03
external protocol from 10% to 25% labels. PEMS03 is excluded from all source
data and initialization checkpoints. Sources are PEMS04, PEMS08, and PEMS07;
initialization comes only from PEMS04 and PEMS08. The audit rechecks stored
commands for this exclusion and rejects any test-exposed artifact.

Three fresh seeds (`6060--6062`) compare matched null, forecast-only, and
channel-patch source pretraining after frozen-core target adaptation. The
locked official PatchTST adapter uses the same target, horizon, label budget,
and chronological validation boundary.

| Arm | Mean validation MSE | Channel-patch difference | Wins for channel-patch |
| --- | ---: | ---: | ---: |
| Null | 1.189381 | -0.771% | 3/3 |
| Forecast pretraining | 1.185612 | -0.455% | 3/3 |
| Channel-patch ACF SSL | 1.180216 | -- | -- |
| Official PatchTST | 1.376393 | -14.253% | 3/3 |

The one-standard-error ratio upper bounds are `0.99573` against forecast-only
pretraining and `0.86385` against PatchTST. Both registered noninferiority
checks pass. No test metric was generated. This adds external support at the
second label budget, but it does not remove the Weather frozen-test reversal,
historical test-access caveat, or high-dimensional limitations.

Pinned official iTransformer under the same split, label budget, target-window
budget, and seeds has mean validation MSE `1.428751`. Channel-Patch is lower
by `-17.395%`, winning all three pairs with one-SE upper ratio `0.83128`. This
is a post-registration validation-only external baseline extension.

Official TimeMixer is initially a high-variance boundary: the first five
paired seeds have favorable mean `-4.422%` but upper ratio `1.02625`. A capped,
pre-registered 15-seed follow-up matches both candidate and TimeMixer to batch
two, five epochs, 64 windows, and 160 optimizer updates. It yields candidate/
TimeMixer means `1.159693`/`1.199889` (`-3.350%`), 7/15 wins, and upper ratio
`0.99978`, passing noninferiority. This conclusion is limited to the matched
PEMS03 H192/25% training geometry.

## Validation Stresses And Deployment

Under the same six validation conditions as the 10% protocol, Channel-Patch
is noninferior to same-seed PatchTST in every cell, with 3/3 wins per
condition. Relative gains versus PatchTST are `-14.253%`, `-14.289%`,
`-13.133%`, `-10.750%`, `-14.206%`, and `-14.061%` in registered order. No
model was retrained and no test partition was opened.

The deployed Channel-Patch checkpoint returns no AutoCon representation and
computes no relation loss. The Mamba encoder is called exactly once for each
of 125 warm-up/measurement predictions. At batch four its median latency is
`0.34248 ms`, versus `0.25438 ms` for official PatchTST, so this supports a
clean one-scan inference claim only, not latency superiority.

Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/audit.json`

Robustness and inference audits:
`artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/robustness_validation/audit.json`
and
`artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/audit.json`

TimeMixer final capped audit:
`artifacts/exploratory/pems03_channel_patch_batch2_timemixer_final_expansion_20261007/audit.json`
