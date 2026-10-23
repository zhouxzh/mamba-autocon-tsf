# MambaTS v3 AutoCon Representation-Gated Linear Expert (2026-09-22)

## Protocol And Correction

The original paired screen froze exact MambaTS v3 AutoCon controls on
Exchange-Rate H96/H336 and ILI H24/H60 (three seeds per coordinate), with
`skip_test=true`. Its expert-on arm failed before validation because the
forecast runner passes `return_repr=false` outside training while the new
expert still required the AutoCon representation. The failure is retained in
the original artifact directory and is not counted as a result.

The implementation now treats the enabled expert as an internal representation
dependency while preserving the public AutoCon return contract. A focused
contract suite passes (`17 passed`). The expert-on retry used a new plan,
output root, provenance snapshot and launch binding; the original expert-off
control was not rerun or modified. Both arms pass strict phase validation
(`12/12` successful jobs each), retain the same data fingerprints and exact
dataset/horizon/seed coordinates, and produce no test metric artifacts.

The optional route is identity-preserving at initialization:

```text
prediction = mamba_prediction
           + tanh(alpha) * sigmoid(g(AutoCon_representation))
             * (shared_linear_prediction - mamba_prediction)
```

Here `alpha=0` initially, the linear head is shared across channels, and the
gate is produced from the same AutoCon representation consumed by the
contrastive loss. It adds negligible observed job-wall-time cost, but this is
not a hardware-independent throughput claim.

## Validation Readout

| Dataset | H | Expert-off MSE | Expert-on MSE | Change | Expert wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exchange-Rate | 96 | `0.276656` | `0.279362` | `+0.976%` | 0/3 |
| Exchange-Rate | 336 | `0.940137` | `0.955781` | `+1.697%` | 0/3 |
| ILI | 24 | `0.252610` | `0.249602` | `-1.190%` | 2/3 |
| ILI | 60 | `0.242389` | `0.247432` | `+2.096%` | 0/3 |

Negative change favors the expert. The strict machine-readable paired rows
and readout are in
[`artifacts/exploratory/mambatsv3_repr_linear_expert_retry_20260922/readout`](../../artifacts/exploratory/mambatsv3_repr_linear_expert_retry_20260922/readout/readout.md).

## Decision

The hypothesis fails its cross-coordinate stability criterion: it worsens both
Exchange coordinates and ILI H60, with no seed-level win on those three cells.
The isolated ILI H24 improvement is insufficient to select the architecture or
to justify a held-out test release. Keep the implementation as a documented
ablation switch, but exclude it from the candidate main model and do not tune
it further without a new, independently motivated hypothesis.
