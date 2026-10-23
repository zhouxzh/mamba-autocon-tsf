# PEMS-BAY Local-Cluster Confirmation Readout

## Registered Result

The three seeds registered in
`mambatsv3_acf_infonce_local_cluster_pems_confirmation_20261003.md` all passed
the source-coverage and validation gates before test access. The development
seed 5503 is excluded from this table.

| Seed | Active validation | Null validation | Active test | Null test | Active vs null |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5510 | 2.757124 | 2.803061 | 2.630116 | 2.666715 | -1.372% |
| 5511 | 2.777384 | 2.827775 | 2.646697 | 2.687012 | -1.500% |
| 5512 | 2.751202 | 2.799341 | 2.628699 | 2.666089 | -1.402% |
| Mean | - | - | 2.635171 | 2.673272 | -1.425% |

The paired test-MSE difference is -0.038101, with a normal-approximation 95%
interval of [-0.040316, -0.035886]. Every active source run had an anchor
coverage of 1.0 in every pretraining epoch. The audit is
[`audit.json`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_confirmation_20261003/audit.json).

## Scope

This establishes a narrowly scoped positive result: with the registered
PEMS07/08/04 source pool, local identifiable ACF InfoNCE improves PEMS-BAY
H=192 forecasting at 10% target labels relative to exact-null predictive
pretraining. It does not establish a universal cross-domain claim, a full-data
claim, or superiority over external state-of-the-art systems.

## Deployment Check

The confirmation checkpoint remains prediction-only at deployment: one Mamba
scan, no returned AutoCon representation, and no relation loss. On the guarded
RTX 5090 D benchmark at batch size four, median forward latency is 0.316288 ms
with 18.49 MB peak allocated memory. This is a same-hardware benchmark, not a
universal hardware-performance claim.
