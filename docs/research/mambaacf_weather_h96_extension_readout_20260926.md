# MambaACF Weather H96 extension readout (2026-09-26)

## Protocol

This validation-only extension follows the frozen protocol in
[`mambaacf_weather_h96_extension_protocol_20260926.md`](./mambaacf_weather_h96_extension_protocol_20260926.md).
It uses Weather M features, H96, `MeanStd`, causal embedding, one Mamba-2
scale, static train-only `train_peaks_available` ACF memory, and a global,
non-overlap, hard selected-lag relation contract. Both arms ran seeds
2021/2022/2023 for at most eight epochs with test evaluation disabled.

The first active seed-2022 log was overwritten by a duplicate process during
execution. It is excluded. A clean rerun with a new model identifier supplies
the registered seed-2022 active value below. All six retained log/artifact
pairs report `skip_test=true` and no test metric was created.

## Validation MSE

| seed | lambda-zero null | active relation (`lambda=.001`) | relative active delta |
|---:|---:|---:|---:|
| 2021 | 0.467564 | 0.467626 | +0.013% |
| 2022 | 0.463535 | 0.463281 | -0.055% |
| 2023 | 0.465405 | 0.465334 | -0.015% |
| mean | 0.465501 | 0.465414 | -0.019% |

The active relation arm wins 2/3 seeds, but its mean effect is only `-0.019%`.
It is not a practically meaningful AutoCon gain.

The retained Weather MambaTS v3 AutoCon reference from the earlier frozen
baseline pilot is `0.445116`. This MambaACF active mean is `+4.560%` worse.
That comparison crosses source snapshots, so it is only a conservative
screening boundary, not a publication comparison. It nevertheless exceeds the
protocol's 2% continuation margin.

## Diagnostics and decision

The relation objective is not mechanically empty: final global contrastive
anchor coverage is about `45%`, with about `42%` degenerate-positive anchors
and signed-target accuracy about `88%`. The finite target geometry does not
translate into useful forecast improvement here.

Stop this MambaACF ACF-memory branch on Weather H96. Do not release Weather
test evaluation, tune lambda/lag settings, or expand it to longer horizons.
The next candidate is the independently promising MambaTS v3 Weather path,
which needs an exact current-source active/lambda-zero replication before any
same-snapshot strong-baseline rerun.

Artifacts:

- `artifacts/exploratory/mambaacf_weather_h96_extension_20260926/null_s2021.log`
- `artifacts/exploratory/mambaacf_weather_h96_extension_20260926/null_s2022.log`
- `artifacts/exploratory/mambaacf_weather_h96_extension_20260926/null_s2023.log`
- `artifacts/exploratory/mambaacf_weather_h96_extension_20260926/active_s2021.log`
- `artifacts/exploratory/mambaacf_weather_h96_extension_20260926/active_s2022_rerun.log`
- `artifacts/exploratory/mambaacf_weather_h96_extension_20260926/active_s2023.log`
