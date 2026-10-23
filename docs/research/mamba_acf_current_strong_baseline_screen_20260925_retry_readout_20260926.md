# Current-source H96 baseline matrix readout

Date: 2026-09-26  
Scope: ETTh1, ETTm1, Electricity and Traffic; M features; H96; three seeds
(`2021--2023`); validation-only; test execution disabled.

## Accuracy summary

The matrix contains 48 strong-baseline coordinates (DLinear, PatchTST,
TimesNet and iTransformer), 12 MambaACF memory coordinates, 12 AutoCon
relation-null coordinates and 12 active AutoCon coordinates. Values below are
the mean of the three seed-level minimum pure-forecast validation MSE values.

| Dataset | DLinear | PatchTST | TimesNet | iTransformer | MambaACF memory | AutoCon null | AutoCon active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.695156 | 0.685596 | 0.768126 | 0.685701 | 0.742074 | 0.730038 | 0.729716 |
| ETTm1 | 0.410020 | 0.399746 | 0.440079 | 0.396035 | 0.433216 | 0.428115 | 0.428075 |
| Electricity | 0.201873 | 0.194715 | 0.170219 | 0.190936 | 0.157517 | 0.155257 | 0.154993 |
| Traffic | 0.506108 | 0.420740 | 0.427351 | 0.413808 | 0.396986 | 0.394245 | 0.394335 |

The strongest plain baseline per dataset is PatchTST for ETTh1, iTransformer
for ETTm1, TimesNet for Electricity and iTransformer for Traffic. Active
AutoCon is worse than that reference by `6.435%` and `8.090%` on ETTh1/ETTm1,
and better by `8.945%` and `4.706%` on Electricity/Traffic.

Relative to the exact AutoCon relation-null control, active changes are
ETTh1 `-0.044%`, ETTm1 `-0.009%`, Electricity `-0.170%`, and Traffic `+0.023%`
(negative favors active). This is a small, mixed pattern and does not establish
a universal causal gain from the relation objective.

## Resource screen

All rows use synthetic device-resident Traffic H96 input, batch 32, 25 warm-up
forwards and 100 timed forwards on the RTX 5090 D. Data loading, host-device
transfer, training and backward passes are excluded.

| Model | Parameters | Median ms | P95 ms | Peak increment MB |
| --- | ---: | ---: | ---: | ---: |
| DLinear | 18,624 | 0.191648 | 0.196138 | 50.51 |
| iTransformer | 14,944 | 0.914672 | 0.924899 | 33.88 |
| MambaTS v3 AutoCon d32 | 69,707 | 0.625584 | 0.632427 | 138.03 |
| Mamba-1 | 173,790 | 0.610176 | 0.623235 | 92.41 |
| TimeMixer | 183,171 | 0.718736 | 0.727651 | 99.78 |
| Mamba-2 | 171,722 | 0.881088 | 0.892035 | 92.41 |
| TimesNet | 1,302,046 | 2.863280 | 2.880229 | 71.67 |
| PatchTST | 46,080 | 3.940768 | 4.079592 | 373.97 |

The candidate is faster than Mamba-1/2, TimeMixer, TimesNet and PatchTST, but
slower than DLinear and iTransformer. It also has the largest peak increment in
this table. The defensible engineering claim is therefore conditional, not
"faster than every baseline".

## Evidence boundary

All 84 registered jobs have successful artifacts. One Traffic/iTransformer
coordinate initially failed because the immutable source snapshot did not
recognize a later optional `mamba_bipatch_token_mode` argument; it was rerun
with that no-op default omitted. The numerical result is present, but the old
phase's strict validator still reports a live-runner/provenance mismatch. This
matrix remains exploratory and must not be used as a publication-level
confirmatory table until regenerated under one fresh immutable source snapshot.

Artifacts: [`mamba_acf_current_strong_baseline_screen_20260925_retry`](../../artifacts/exploratory/mamba_acf_current_strong_baseline_screen_20260925_retry/),
[`traffic_h96_b32_resource_screen_20260926`](../../artifacts/exploratory/traffic_h96_b32_resource_screen_20260926/).
