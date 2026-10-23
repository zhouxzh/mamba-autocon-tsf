# Channel-Patch SSL At 25% Labels

Status: fresh three-seed, validation-only current-geometry replication. No
test row was read.

The same Mamba-1 v3 `24/8` shared encoder and channel-patch ACF representation
objective used at 10% labels was rerun at 25% chronological labels. New PEMS
seeds are `6036--6038`; new METR-LA seeds are `6039--6041`. Each seed contains
next-patch null, forecast-level ACF InfoNCE, and forecast-plus-channel-patch
InfoNCE. The additional representation loss had active anchors in every source
epoch for all six candidate runs.

| Target | Null MSE | Forecast-only MSE | Channel-patch MSE | Channel-patch vs forecast | Wins | Upper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192 | 2.423843 | 2.356698 | 2.313817 | -1.820% | 3/3 | 0.98263 |
| METR-LA H192 | 1.193424 | 1.189608 | 1.178294 | -0.951% | 3/3 | 0.99093 |

Same-seed official PatchTST validation checkpoints were trained from pinned
commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9` under the identical 25%
chronological label budget.

| Target | Channel-patch vs official PatchTST | Wins | Upper |
| --- | ---: | ---: | ---: |
| PEMS-BAY H192 | -2.253% | 3/3 | 0.98460 |
| METR-LA H192 | -6.024% | 3/3 | 0.94209 |

The six read-only stresses also pass external noninferiority in every cell.
For PEMS-BAY, the PatchTST upper range is `0.98269--0.98911`; for METR-LA it
is `0.94096--0.95916`. All twelve cells have three candidate wins.

## Deployment Check

The PEMS-BAY channel-patch checkpoint is benchmarked directly at batch four,
50 warm-up calls and 200 CUDA-timed prediction calls. It uses exactly one
Mamba encoder block per forward (`250` total encoder calls including warm-up),
returns no AutoCon representation, and constructs no relation loss. Median
latency is `0.33880 ms`, versus `0.25507 ms` for the same-geometry official
PatchTST checkpoint. Thus the channel-patch training objective introduces no
inference scan or contrastive computation, but it does not establish an
external latency advantage.

Artifacts:

- `artifacts/exploratory/channel_patch_25pct_replication_20261006/audit.json`
- `artifacts/exploratory/channel_patch_25pct_replication_20261006/external_patchtst_25pct_audit.json`
- `artifacts/exploratory/channel_patch_25pct_replication_20261006/robustness_validation/audit.json`
- `artifacts/exploratory/channel_patch_25pct_replication_20261006/inference_benchmark/mamba.json`
- `artifacts/exploratory/channel_patch_25pct_replication_20261006/inference_benchmark/patchtst.json`
