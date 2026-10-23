# MambaTS high-dimensional profiler audit

_Current-source GPU profiler, 2026-09-30. The audit identifies an optimization target; it does not claim a new forecasting result._

---

## 📋 Protocol

The frozen relation-auxiliary-only `MambaTSv3AutoCon` configurations for Electricity and Traffic were profiled with synthetic device-resident inputs, `B=32`, `L=96`, `H=96`, 10 warm-up forwards and 10 profiled forwards on RTX 5090 D. No dataset, checkpoint, test split or training metric was accessed.

## 🔍 Dominant GPU regions

| Dataset | Dominant region | Self CUDA time over 10 forwards | Interpretation |
| --- | --- | ---: | --- |
| Electricity | Mamba selective scan | `2,265.3 us` | Main recurrent-kernel cost |
| Electricity | `native_layer_norm` | `2,189.8 us` | Two normalization launches per forward path |
| Electricity | Matrix multiplication kernels | `1,435.3 us` | Patch/context and forecast projections |
| Traffic | Mamba selective scan | `7,296.5 us` | Dominant cost grows with channel count |
| Traffic | `native_layer_norm` | `5,779.9 us` | Significant second cost at `C=862` |
| Traffic | Matrix multiplication kernels | `4,170.8 us` | High-dimensional projection cost |

`copy_`, elementwise additions and causal convolution are secondary but non-negligible. The auxiliary representation is disabled by the resource forward contract, so these measurements isolate the forecasting backbone rather than charging the model for a relation-loss computation that is not used during inference.

## ✅ Evidence-based design decision

The next architecture candidate should preserve the existing MambaTS patch-token ordering, low-rank cross-variable context, shared forecast/AutoCon representation contract and train-only AutoCon objective, while testing a Mamba-2 mixer in the same scan location. This directly targets the measured selective-scan cost and is more interpretable than adding another parallel Mamba branch.

The Mamba-2 replacement was subsequently implemented as explicit `mamba_ts_mixer_type=mamba2` and passed CUDA forecast/representation/backward smoke. However, its exact Mamba-1 matched resource control is worse by `25.020%` on Electricity and `26.655%` on Traffic, while also increasing peak allocation. The independent-channel MambaTSv2 path is also slower than v3 by `295.00%/245.85%` on the same shapes. Both branches are stopped before validation training; see the [Mamba-2 mixer resource stop](../../artifacts/exploratory/mambatsv3_mamba2_mixer_resource_smoke_20261001/readout.md) and [MambaTSv2 preflight](../../artifacts/exploratory/mambatsv2_highdim_resource_preflight_20261001/readout.md).

## Source artifacts

- [Electricity profiler JSON](../../artifacts/exploratory/current_source_highdim_mambats_profile_20261001/electricity/profile.json)
- [Traffic profiler JSON](../../artifacts/exploratory/current_source_highdim_mambats_profile_20261001/traffic/profile.json)
- [High-dimensional resource preflight](../../artifacts/exploratory/current_source_highdim_relation_auxiliary_resource_preflight_20261001/readout.md)
