# MambaACF Channel-Gate Screen

Status: **exploratory; not confirmatory**.

The three arms share the raw-source train-only ACF sidecar, variable-token MambaACF AutoCon model, relation target, sampler, data, seeds, and budget. They differ only in the low-rank channel gate: disabled, ordinary, or additionally conditioned on the train-only ACF summary.

Negative deltas favor the treatment arm. The paired sample is six dataset/seed coordinates and is descriptive only.

| Contrast | Metric | n | Mean delta | Relative delta | Wins lower | Two-sided p |
|---|---|---:|---:|---:|---:|---:|
| gate - off | MSE | 6 | -0.00209063 | -0.452% | 4 | 0.1223 |
| gate - off | MAE | 6 | -0.000129155 | -0.034% | 3 | 0.8792 |
| ACF gate - gate | MSE | 6 | 0.00160148 | 0.347% | 2 | 0.2628 |
| ACF gate - gate | MAE | 6 | 0.0007825 | 0.206% | 3 | 0.5969 |
| ACF gate - off | MSE | 6 | -0.00048915 | -0.106% | 5 | 0.4214 |
| ACF gate - off | MAE | 6 | 0.000653345 | 0.172% | 3 | 0.5825 |

## Dataset Splits

| Contrast | Dataset | MSE delta | MSE relative | MAE delta | MAE relative |
|---|---|---:|---:|---:|---:|
| gate - off | ETTh1 | -8.84102e-05 | -0.023% | 0.000969716 | 0.239% |
| gate - off | traffic | -0.00409285 | -0.752% | -0.00122803 | -0.345% |
| ACF gate - gate | ETTh1 | -0.000686436 | -0.180% | -0.000491902 | -0.121% |
| ACF gate - gate | traffic | 0.00388939 | 0.720% | 0.0020569 | 0.580% |
| ACF gate - off | ETTh1 | -0.000774846 | -0.203% | 0.000477814 | 0.118% |
| ACF gate - off | traffic | -0.000203454 | -0.037% | 0.000828876 | 0.233% |

## Gate Activation

Checkpoint scalars are reported to distinguish an ineffective architecture from an adapter that remained closed behind ReZero.

| Arm | Mean gate residual scale | Max absolute gate scale | Mean ACF latent scale | AutoCon loss |
|---|---:|---:|---:|---:|
| off | 0 | 0 | n/a | 0.625292 |
| gate | 0.0812804 | 0.408229 | n/a | 0.615048 |
| acf_gate | 0.0321808 | 0.255312 | 1.02994 | 0.624495 |

## Boundaries

- This screen does not establish a universal accuracy or speed advantage; elapsed time is especially noisy at this short budget.
- It does not test longer horizons, parameter-matched baselines, alternative channel orderings, or balanced relation sampling.
- The ACF-conditioned arm uses a train-only descriptor, but the broader residual/trend source path is not part of this raw-source causality claim.
