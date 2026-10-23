# MambaTS v3 AutoCon Channel Adapter: Frozen Validation Protocol

## Purpose

This protocol registers the first new architecture after the completed plain
Mamba matrix. It tests a low-rank, channel-preserving forecast residual built
from the existing MambaTS v3 patch states. The adapter performs no additional
Mamba scan: it averages the already encoded patch states for each channel,
projects them through a shared rank-8 bottleneck, and applies a zero-initialized
horizon gate. The same encoded states are exposed to the AutoCon representation
loss, so the experiment distinguishes a prediction-connected representation
from a forecast-only side branch.

The design is inspired by the variable-preserving principle in MambaTS and the
representation-sharing requirement of AutoCon. It is not a reimplementation of
ms-Mamba: the latter requires extra selective scans and an unfused Delta path,
which was rejected by the efficiency audit.

## Frozen comparison

- Datasets: PEMS03, PEMS04, PEMS07, PEMS08.
- Horizons: 12 and 96.
- Seeds: 2022 and 2023 for the exploratory screen.
- Arms: exact MambaTS v3 AutoCon null (`mamba_ts_channel_adapter=false`) and
  active adapter (`true`, rank 8), both with `AutoCon_lambda=0.001`.
- Controls: `seq_len=96`, `MeanStd`, shared optimizer, four-epoch direction-
  finding budget, `skip_test=true`, and the same source snapshot.
- Primary pairing: same dataset, horizon, and seed; minimum pure-forecast
  validation loss.
- Resource gate: active median prediction latency must be no more than 10%
  above the null on a fixed device-resident protocol.

## Selection rule

The adapter is retained only if its mean validation MSE is lower than the null
in at least two of the four PEMS datasets at both horizons, and it passes the
latency gate. Otherwise the branch is retained as a negative ablation and no
larger rank or horizon search is allowed. Test metrics remain withheld until a
separate confirmation matrix is frozen.

## Implementation evidence

The branch is default-off and contract-limited to `MambaTSv3AutoCon`; rank must
be positive. Its initialization is exactly nested in the null because the
horizon gate is zero. Contract tests cover output shape, zero-gated equality,
nonzero MSE gradient to the gate, runtime identity, plan validation, and resource
configuration parsing. The current implementation adds a shared rank-8 input
projection, a shared horizon projection, and `pred_len` gates per run; it does
not allocate a per-patch representation forecast tensor.

## Validation result

The frozen 32-job screen completed with `32/32` successful manifest rows. Both
phases share source snapshot
`f2a2ca1f2b3cc0ce7a6e7e33c42b6c9000b9d5be8be19b5578d69cdd1364ddab`, both are
bound to the same launch plan, and zero test artifacts were produced. The
strict machine-readable readout is in
[`readout.md`](../../artifacts/exploratory/mamba_ts_v3_channel_adapter_screen_20260923/readout/readout.md).

| Dataset | H12 change | H96 change | H96 adapter wins | Decision |
| --- | ---: | ---: | ---: | --- |
| PEMS03 | `-0.012%` | `-2.308%` | `2/2` | pass |
| PEMS04 | `-0.278%` | `-2.186%` | `2/2` | pass |
| PEMS07 | `-0.039%` | `-1.658%` | `2/2` | pass |
| PEMS08 | `+0.307%` | `+0.382%` | `1/2` | failure analysis |

The pre-registered screen rule is satisfied: at least two datasets improve at
both horizons, in fact three do. The adapter is therefore retained for a new
independent confirmation design, but this exploratory result is not a test
ranking or a publication claim. The PEMS08 regression must remain in the main
table and cannot be removed by selecting another seed or rank.

Mean job wall time increases by `0.83%--3.60%` across cells. The separate
device-resident resource preflight measured `+6.37%` median forward latency,
under the `10%` gate. Neither measurement is an end-to-end speed claim.

## Resource preflight

On the current `mamba` environment (`torch 2.8.0+cu128`, `mamba-ssm 2.3.1`),
RTX 5090 D, PEMS07-shaped synthetic input (`B=2,T=96,H=96,C=883`), 20 warm-up
and 100 timed device-resident prediction-only forwards gave:

| Arm | Parameters | Median latency | P95 latency |
| --- | ---: | ---: | ---: |
| v3 AutoCon null | 104,291 | 0.32765 ms | 0.33802 ms |
| v3 AutoCon channel adapter, rank 8 | 105,411 | 0.34851 ms | 0.36076 ms |

The adapter is `+6.37%` in median latency and therefore passes the registered
`<=10%` structural gate. This is a synthetic forward measurement only; it is
not a training-throughput, memory, or forecasting-accuracy claim. The raw
resource records are retained under `/tmp/mamba_ts_v3_adapter_resource_null`
and `/tmp/mamba_ts_v3_adapter_resource_on` for this working session and must be
regenerated under the frozen experiment provenance before any paper table.
