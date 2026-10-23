# MambaTS v3 Transfer Baseline Extension (2026-09-22)

## Protocol

This completed validation-only matrix contains 144 jobs under one fresh source
snapshot: Weather and Exchange-Rate at H96/H336, plus ILI at H24/H36/H48/H60.
Each coordinate has three seeds and six model arms: DLinear, PatchTST, TimesNet,
iTransformer, Mamba-2, and MambaTS v3 AutoCon (`lambda=0.001`). Both phase
validators report `valid=true`, `72/72` unique successful jobs. Every result
has `skip_test=true`; test metric artifacts are absent.

The strict machine-readable output is
[`readout.md`](../../artifacts/exploratory/mambatsv3_transfer_baseline_extension_20260922/readout/readout.md).

## Results

| Dataset | Horizon | Best model | MambaTS v3 AutoCon MSE | vs. best |
| --- | ---: | --- | ---: | ---: |
| Weather | 96 | MambaTS v3 AutoCon | `0.445106` | `0.000%`, 3/3 |
| Weather | 336 | MambaTS v3 AutoCon | `0.614143` | `0.000%`, 3/3 |
| Exchange-Rate | 96 | PatchTST | `0.276653` | `+5.872%`, 0/3 |
| Exchange-Rate | 336 | PatchTST | `0.940276` | `+8.165%`, 0/3 |
| ILI | 24 | DLinear | `0.252986` | `+7.061%`, 0/3 |
| ILI | 36 | DLinear | `0.232297` | `+8.197%`, 0/3 |
| ILI | 48 | DLinear | `0.236036` | `+10.282%`, 0/3 |
| ILI | 60 | PatchTST | `0.240531` | `+9.097%`, 0/3 |

The candidate remains consistently better than TimesNet and Mamba-2 on
Exchange/ILI, but that is not the relevant bar. It is consistently worse than
the winning DLinear/PatchTST arms. Together with ETTh1, this rules out a claim
that the current MambaTS v3 architecture is generally non-inferior.

## Structural Implication

The v3 model already has a shared `seq_len -> pred_len` persistence residual,
initialized as last-value copying and trained as a linear path. Adding a second
unmotivated DLinear duplicate would not explain or solve this failure. The next
test keeps the architecture fixed and raises the maximum budget from 8 to 20
epochs for Exchange/ILI against DLinear and PatchTST. It distinguishes a
short-budget ReZero/optimization effect from a true representational failure.
No test release is justified for these datasets until that validation evidence
exists.
