# MambaACF Anchor x Soft-ACF-Memory Interaction Screen

> Exploratory validation-only 2x2 factorial screen. Test metrics are disabled; all contrasts are paired by dataset and seed.
> The immutable plan metadata remains a draft; the 24 artifacts were executed post-hoc under its frozen source snapshot, so this is not preregistered or confirmatory evidence.

Primary metric: best finite `val_loss` from `checkpoint/loss_logs.json`; lower is better.

## Simple effects

| Contrast | Dataset | n | mean MSE delta | relative ratio | wins |
|---|---|---:|---:|---:|---:|
| anchor_at_memory_off | pooled | 6 | -0.0022 | -0.398% | 5/6 |
| anchor_at_memory_off | ETTh1 | 3 | -0.0039 | -0.530% | 3/3 |
| anchor_at_memory_off | traffic | 3 | -0.0005 | -0.144% | 2/3 |
| anchor_at_memory_on | pooled | 6 | -0.0025 | -0.447% | 5/6 |
| anchor_at_memory_on | ETTh1 | 3 | -0.0034 | -0.464% | 3/3 |
| anchor_at_memory_on | traffic | 3 | -0.0016 | -0.414% | 2/3 |
| memory_at_anchor_off | pooled | 6 | -0.0008 | -0.148% | 5/6 |
| memory_at_anchor_off | ETTh1 | 3 | -0.0014 | -0.193% | 3/3 |
| memory_at_anchor_off | traffic | 3 | -0.0002 | -0.061% | 2/3 |
| memory_at_anchor_on | pooled | 6 | -0.0011 | -0.197% | 6/6 |
| memory_at_anchor_on | ETTh1 | 3 | -0.0009 | -0.127% | 3/3 |
| memory_at_anchor_on | traffic | 3 | -0.0013 | -0.331% | 3/3 |

## Interaction

| Dataset | n | difference-in-differences | 95% CI | wins (<0) | p (descriptive) |
|---|---:|---:|---:|---:|---:|
| pooled | 6 | -0.0003 | [-0.0014, 0.0009] | 2/6 | 0.5814 |
| ETTh1 | 3 | 0.0005 | [-0.0005, 0.0015] | 0/3 | 0.1740 |
| traffic | 3 | -0.0010 | [-0.0038, 0.0018] | 2/3 | 0.2555 |

## Decision boundary

- stop if combined arm worsens versus plain by >1% pooled or on all three seeds in either dataset
- Combined-vs-plain pooled ratio: `-0.594%`; pooled wins: `5/6`.
- Interaction coherent for follow-up: `False` (negative in both datasets: `False`; each dataset CI excludes zero: `False`).
- Action: **do not promote the combination: the no-harm screen passed, but the interaction is inconclusive or directionally inconsistent**

Negative interaction values mean the combined arm reduces validation error more than the sum of its two isolated effects. A positive value indicates interference or sub-additivity; it is not evidence that either mechanism is intrinsically invalid.

The sidecar, common configuration, and factor-key invariants are checked before this readout is written. Missing or incomplete artifacts return `NOT_RUN` (exit code 3).

Validation-only paired descriptive evidence; interaction and p-values are exploratory and cannot establish a test-set or publication claim.
