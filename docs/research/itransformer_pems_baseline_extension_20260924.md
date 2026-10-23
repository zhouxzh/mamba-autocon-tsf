# iTransformer PEMS Baseline Extension

## Rationale

The completed PEMS matrix contains linear, patch, periodic-convolution, and
Mamba references, while the repository already provides an inverted-variable
Transformer implementation. This plan adds iTransformer as a fixed comparator
before any future Mamba architecture claim is considered.

## Frozen validation contract

- PEMS03, PEMS04, PEMS07, PEMS08; M features; input length 96.
- Horizons 12 and 96; seeds 2021 through 2025.
- At most 20 epochs, patience 5, batch size 32, same optimizer and
  `MeanStd` population-variance normalization contract as the completed PEMS
  matrix.
- `4 x 2 x 5 = 40` validation-only jobs, each with `skip_test=true`.
- Primary metric: minimum pure forecast validation loss. No test output may
  be produced or used to choose a model.

The source snapshot and launch authorization are created only after the
iTransformer contract and plan validation pass. Its results must be compared
descriptively with earlier PEMS provenance roots unless a future full matrix
is rerun from one common snapshot.
