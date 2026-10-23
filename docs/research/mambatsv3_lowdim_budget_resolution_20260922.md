# MambaTS v3 Low-Dimensional Budget Resolution (2026-09-22)

## Protocol

This validation-only experiment reran the exact MambaTS v3 AutoCon architecture
and DLinear/PatchTST under a 20-epoch maximum with patience 5, rather than the
earlier eight-epoch screening budget. It covers Exchange-Rate H96/H336 and ILI
H24/H60, three seeds per coordinate, 36 jobs total. Both phase validators pass
(`18/18` each), all jobs are successful, and test metrics are withheld.

## Result

| Dataset | H | Candidate vs DLinear | Candidate vs PatchTST |
| --- | ---: | ---: | ---: |
| Exchange-Rate | 96 | `+5.881%` | `+6.814%` |
| Exchange-Rate | 336 | `+7.462%` | `+8.878%` |
| ILI | 24 | `+8.388%` | `+10.898%` |
| ILI | 60 | `+9.804%` | `+9.911%` |

All four cells remain unfavorable for MambaTS v3 AutoCon after the longer
budget. The gap is not explained by early stopping at eight epochs, so simply
increasing the budget or adding Mamba depth is not a defensible optimization.

## Follow-up Decision

The proposed AutoCon-representation-gated shared linear expert has now been
tested under the exact paired coordinates. It improves only ILI H24 and
degrades the remaining three cells, so it does not close the
DLinear/PatchTST gap and is stopped as a negative ablation. See
[`mambatsv3_repr_linear_expert_retry_20260922.md`](mambatsv3_repr_linear_expert_retry_20260922.md).

No test release is justified for Exchange/ILI.
