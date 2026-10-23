# PEMS03 Fixed-Shape CUDA Graph Inference

The frozen PEMS03 H192/25% Channel-Patch deployment checkpoint was benchmarked
with a static CUDA Graph. The graph owns fixed-shape GPU input and output
buffers; each prediction first copies a new device-resident input into the
static input, then replays the graph. It does not modify weights, architecture,
Mamba scan count, AutoCon representation behavior, or relation loss.

| Device-side batch-four path | Median latency |
| --- | ---: |
| Eager Channel-Patch | 0.34248 ms |
| CUDA Graph Channel-Patch | 0.25824 ms |
| Official PatchTST eager | 0.25438 ms |

CUDA Graph replay is exactly equivalent to eager prediction (`max abs error =
0`), retains one Mamba encoder block, and includes no AutoCon representation
or relation loss. The fixed-shape device-only graph reduces latency by `24.60%`
relative to eager and is `+1.52%` relative to PatchTST.

This is a bounded serving optimization, not a universal speed claim: it omits
host-to-device transfer, output copying, dynamic batch/sequence/channel shapes,
and graph construction overhead.

## Batch Grid

The same checkpoint was measured at all registered fixed device-side batch
sizes, each with 25 warm-up and 100 timed predictions. The graph is exactly
equivalent to eager prediction in every cell.

| Batch | Channel-Patch CUDA Graph | Official PatchTST eager | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 0.14606 ms | 0.24506 ms | -40.40% |
| 4 | 0.25826 ms | 0.25710 ms | +0.45% |
| 8 | 0.40717 ms | 0.34355 ms | +18.52% |

The deployment claim is therefore restricted to low-latency batch-one service
on this GPU and fixed shape. It does not claim universal throughput or batch-8
superiority.

Batch-grid audit:
`artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/cuda_graph_batch_grid/audit.json`

Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/cuda_graph_audit.json`
