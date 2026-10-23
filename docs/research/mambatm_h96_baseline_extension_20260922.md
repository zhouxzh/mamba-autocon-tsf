# TimeMachine-Style Mamba H96 Baseline Extension (2026-09-22)

## Protocol

This validation-only screen evaluates the local TimeMachine-inspired
four-path Mamba-1/Mamba-2 implementations with the repository's AutoCon
representation objective. It uses three paired seeds on ETTh1, Weather,
Exchange-Rate, and ILI at H96, with fixed DLinear, NLinear, PatchTST,
TimesNet, iTransformer, Mamba-2 AutoCon, and MambaTS v3 AutoCon references.
All 108 jobs completed. The strict phase validator reports `valid=true`,
`108/108` unique successful jobs, a single frozen source snapshot, and no
test metric artifacts.

The architecture follows the public TimeMachine design at a mechanism level:
two `seq_len -> width` projections, four Mamba paths, residual fusion, and a
direct multi-horizon head. The AutoCon branch is an additional time-indexed
representation derived from the first shared projection; it is not claimed to
be part of the external TimeMachine paper.

## Results

| Dataset | Best baseline/model | MambaTM1 AutoCon | MambaTM2 AutoCon | MambaTS v3 AutoCon |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | iTransformer `0.683150` | `0.693979` | `0.710837` | `0.710443` |
| Weather | MambaTS v3 `0.445621` | `0.480892` | `0.484294` | `0.445621` |
| Exchange-Rate | PatchTST `0.259005` | `0.294565` | `0.301384` | `0.276656` |
| ILI | NLinear `0.207772` | `0.296839` | `0.366585` | `0.251235` |

Mean end-to-end job time confirms the expected constant-factor cost. MambaTM2
takes `42.10s/125.01s/25.63s/15.95s` on ETTh1/Weather/Exchange/ILI, compared
with MambaTS v3's `9.05s/34.33s/5.53s/3.00s`; MambaTM1 is also slower than
MambaTS v3 on every coordinate except ILI by a negligible amount, while still
substantially less accurate there.

The full machine-readable results are in
[`readout`](../../artifacts/exploratory/mambatm_h96_baseline_extension_20260922/readout/readout.md)
and paired seed comparisons are in `paired_comparisons.csv` in the same
directory.

## Decision

The four-path TimeMachine mechanism is a useful external Mamba baseline and
AutoCon compatibility control, but fails both the accuracy and efficiency
requirements for the main model under this protocol. Do not run a long-horizon
extension or held-out test release for it. The next architecture should retain
the direct patch/shared-scan efficiency of MambaTS v3 and focus on the proven
remaining limitation: a lightweight, explicitly controlled cross-variable
path rather than duplicating temporal Mamba branches.
