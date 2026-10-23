# Variable-Token MambaACF Fair Screen

> Exploratory, non-confirmatory readout. The only relation-effect contrast is `lambda=.03 - lambda=0` within the identical `MambaACFVariableAutoCon` architecture.

## Contrasts

| Treatment - reference | n | MSE delta | MSE relative | MSE wins | MAE delta |
|---|---:|---:|---:|---:|---:|
| memory on - memory off | 6 | -0.005546 | -1.168% | 6/6 | -0.002444 |
| relation lambda=.03 - null lambda=0 | 6 | -0.004630 | -0.979% | 5/6 | -0.002903 |
| calibrated - relation | 6 | -0.000119 | -0.025% | 3/6 | -0.000214 |
| AutoCon null - memory on (descriptive/confounded) | 6 | 0.003574 | 0.762% | 2/6 | 0.001270 |

Negative deltas favor the treatment. Paired t statistics and p-values are descriptive only; no test metric is used for selection.

## Invariants

- Every arm must have six exact dataset-seed coordinates, immutable provenance, complete artifact hashes, and a raw train-only `acf_prior.npz`.
- Sidecar hashes, effective forecast lags, variable-token trunk settings, and relation candidate lags must agree at each coordinate.
- The memory-on to AutoCon-null comparison changes the executable AutoCon branch and is therefore confounded; it is not relation evidence.
- Comparisons with the separate non-variable fair-baseline plan are descriptive context only and must not be pooled.

## Stopping Decision

- Favorable null-to-relation coordinates: `5/6`; by dataset: `{'ETTh1': 2, 'traffic': 3}`.
- Action: **continue only to separately reviewed frozen replication**.
- This screen cannot support a publication claim without a separately reviewed frozen replication.
