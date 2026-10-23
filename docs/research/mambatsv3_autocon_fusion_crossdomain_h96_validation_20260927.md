# Cross-domain Fusion Validation

## Protocol

The validation-only matrix uses fresh seeds `2411--2413` and a shared MambaTSv3 AutoCon configuration on ETTh1 H96, Weather H96, Exchange-Rate H96, and ILI H48. The candidate enables prediction-connected AutoCon representation fusion. DLinear, NLinear, PatchTST, TimeMixer, iTransformer, Mamba2, and TimeMachine-style MambaTM1 are same-source references under the same split, normalization, optimizer, scheduler, batch size, epoch budget, and patience.

All `96/96` jobs finished on `cuda:0`; the four strict phase validators are valid. Every run sets `skip_test=true`. The machine-readable readout is [`readout.md`](../../artifacts/exploratory/mambatsv3_autocon_fusion_crossdomain_h96_validation_20260927/readout/readout.md).

## External-baseline result

| Coordinate | Candidate MSE | Strongest baseline | Gap | Decision |
| --- | ---: | --- | ---: | --- |
| Weather H96 | 0.446518 | TimeMixer 0.462390 | -3.424% | Retain as external-baseline evidence |
| ETTh1 H96 | 0.706591 | PatchTST 0.684875 | +3.170% | Stop |
| Exchange-Rate H96 | 0.286903 | PatchTST 0.259636 | +10.502% | Stop |
| ILI H48 | 0.241198 | NLinear 0.210715 | +14.471% | Stop |

The Weather candidate wins all three paired seeds against each registered baseline. The other three coordinates violate the predeclared 2% external-baseline threshold; they must not enter a held-out release.

## Exact fusion control

A separate fresh three-seed `24/24` validation-only control fixes the candidate architecture and AutoCon loss, changing only `mamba_ts_repr_forecast_fusion`.

| Coordinate | Fusion off | Fusion on | Change | On wins | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh1 H96 | 0.711958 | 0.711011 | -0.131% | 2/3 | Pass |
| Weather H96 | 0.446304 | 0.449469 | +0.724% | 1/3 | Fail |
| Exchange-Rate H96 | 0.280192 | 0.282449 | +0.811% | 1/3 | Fail |
| ILI H48 | 0.238308 | 0.237189 | -0.438% | 2/3 | Fail |

The fusion switch is not a stable cross-domain mechanism. Weather's strong candidate-versus-baseline score cannot be attributed to this switch, and ETTh1's tiny on/off improvement does not overcome its PatchTST gap. Neither result supports a universal AutoCon-fusion claim.

## Publication Boundary

This work adds useful negative-transfer evidence and a stronger multi-family baseline table. It does not create a high-level general forecasting result. The next architecture must be frozen before test access and must pass both exact mechanism and multi-domain external-baseline gates before any held-out release.
