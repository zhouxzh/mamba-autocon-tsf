# Latent Temporal Mamba High-Dimensional Resource Preflight

_Prediction-only resource evidence for a validation-only AutoCon candidate; no accuracy job has been launched._

---

## 📋 Candidate

`LatentTemporalMambaAutoCon` compresses each multivariate observation to at most 16 latent variables, average-pools four observed time steps into 24 temporal tokens, and uses one Mamba-1 scan. A DLinear-style seasonal/trend prediction is the full-channel base; a zero-initialized Mamba residual modifies that prediction. The AutoCon representation is interpolated from the same Mamba states, so the auxiliary loss and forecast correction share the latent compressor, scan, and decoder path.

This is a local architecture, motivated by the efficiency problem of a `C x patch_count` scan. It is not presented as a reproduction of an external repository.

## ⚙️ Fixed-shape result

The preflight uses RTX 5090 D, FP32, device-resident synthetic inputs, `B=32`, `L=96`, `H=96`, 25 warm-up forwards, and 100 timed prediction-plus-AutoCon-representation forwards. Relation loss is excluded because it is training-only. Both models returned representation tensors with shape `[32,96,64]`; this is the fairer comparison for an AutoCon model than a prediction-only path. The reference uses the completed CPU-4 stride-24 MambaTSv3 configuration under the same fixed-shape protocol.

| Dataset | Latent temporal median | Stride-24 v3 median | Latency change | Latent parameters | v3 parameters | Peak allocated memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Electricity C321 | `0.3763 ms` | `0.4255 ms` | `-11.57%` | `37,137` | `55,281` | `41.43 MB` vs `104.17 MB` |
| Traffic C862 | `0.5038 ms` | `0.8727 ms` | `-42.28%` | `54,449` | `81,790` | `95.43 MB` vs `268.71 MB` |

All four preflight commands completed under the GPU guard without a trigger. Temperature peaked at `40°C`; fan telemetry was unavailable for these short commands and is therefore not interpreted as zero.

## 🔍 Decision

The speed gate passes for the two registered high-dimensional shapes, so the candidate is eligible for a validation-only accuracy test. It is not eligible for a held-out release yet. The next registered test is [`latent_temporal_mamba_highdim_autocon_validation_20261001.json`](latent_temporal_mamba_highdim_autocon_validation_20261001.json): 60 validation-only jobs with an exact `lambda=0` AutoCon control, the active AutoCon arm, a stride-24 MambaTSv3 reference, and seven strong baselines.

The future reader must report every dataset/model/seed coordinate. Promotion requires the active AutoCon arm to improve its exact null control and remain within 2% of the strongest external baseline on both datasets. Training elapsed time must not replace this fixed-shape resource measurement.

The pre-authorization audit is [`plan_audit.json`](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/plan_audit.json). It verified the exact plan hash, all 60 registered jobs, the `skip_test=true` boundary, and the resource-record hashes before launch. The subsequent validation run is documented in [`latent temporal validation stop`](latent_temporal_mamba_highdim_validation_20261001.md); the original plan and launch sidecar remain immutable provenance records.

## 🔗 Artifacts

- [Electricity resource JSON with representation](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/with_representation/electricity/model_resources.json)
- [Traffic resource JSON with representation](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/with_representation/traffic/model_resources.json)
- [Electricity stride-24 reference with representation](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/with_representation/reference_electricity/model_resources.json)
- [Traffic stride-24 reference with representation](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/with_representation/reference_traffic/model_resources.json)
- [Electricity GPU guard event](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/gpu_guard/electricity_event.json)
- [Traffic GPU guard event](../../artifacts/exploratory/latent_temporal_mamba_highdim_resource_20261001/gpu_guard/traffic_event.json)
