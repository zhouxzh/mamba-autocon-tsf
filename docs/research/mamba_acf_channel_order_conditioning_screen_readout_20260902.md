# MambaACF Channel Order and Conditioning Screen

> Exploratory, non-confirmatory readout. All rows use the same memory-backed global-relation `MambaACFVariableAutoCon` architecture; only train-prior order or conditioning is changed.

The three phase validators and the strict replay checks pass for all 18
coordinates. The metrics are nevertheless bound to the immutable launch
snapshots, not to the current working tree: each phase records the same four
post-launch live-source mismatches (`models/MambaACF.py`, `run.py`,
`scripts/benchmark_model_resources.py`, and `scripts/run_training_plan.py`).
This is a provenance boundary, not evidence of a current-tree rerun.

## Contrasts

| Treatment - input-order control | n | MSE delta | MSE relative | MSE wins | MAE delta | p (descriptive) |
|---|---:|---:|---:|---:|---:|---:|
| strength order | 6 | 0.000527 | 0.113% | 3/6 | -0.000189 | 0.6785 |
| ACF token conditioning | 6 | 0.002800 | 0.598% | 2/6 | 0.003088 | 0.1649 |

### Per-dataset paired MSE

| Contrast | Dataset | n | Reference mean | Treatment mean | Delta | Wins |
|---|---|---:|---:|---:|---:|---:|
| strength order | ETTh1 | 3 | 0.384496 | 0.387613 | 0.003117 | 0/3 |
| strength order | traffic | 3 | 0.552047 | 0.549984 | -0.002063 | 3/3 |
| ACF token conditioning | ETTh1 | 3 | 0.384496 | 0.383684 | -0.000812 | 2/3 |
| ACF token conditioning | traffic | 3 | 0.552047 | 0.558459 | 0.006412 | 0/3 |

Negative deltas favor the treatment. Paired t statistics and p-values are descriptive only; no test metric is used for selection.

## Invariants

- Every arm has six exact dataset-seed coordinates, immutable provenance, complete artifact hashes, and a raw train-only `acf_prior.npz`.
- Sidecar hashes, effective forecast lags, relation candidate lags, and all non-treatment trunk settings agree at each coordinate.
- Strength order is replayed from the sidecar and must be non-identity; input control must be identity. The conditioner scale is read from checkpoint state.
- The omitted strength-plus-conditioning interaction and all comparisons with other plans are outside this readout.
- Replay status is recorded in the JSON. The immutable launch snapshots validate the artifacts; any post-launch live-source drift is reported and is not treated as a rerun.

## Screening Decision

- Strength-order rule: `3/6` favorable; by dataset `{'ETTh1': 0, 'traffic': 3}`.
- Conditioning rule: `2/6` favorable; by dataset `{'ETTh1': 2, 'traffic': 0}`.
- Any favorable direction is exploratory only and cannot support a publication claim without a separately reviewed frozen replication.
