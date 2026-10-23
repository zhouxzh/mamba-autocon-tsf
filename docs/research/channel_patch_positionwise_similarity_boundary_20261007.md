# Channel-Patch Positionwise Similarity Boundary

The canonical Channel-Patch objective flattens each channel's full patch-state
trajectory before computing the ACF-guided contrastive similarity. This screen
tests a positionwise alternative: patch embeddings are normalized separately,
the same patch positions are compared across windows, and those cosines are
averaged. It changes no ACF positive/negative pair, model parameter, Mamba
scan, prediction loss, or deployment computation.

On strict target-excluded PEMS03 H192/25% seed `6083`:

| Similarity | Validation MSE | Difference |
| --- | ---: | ---: |
| Flattened trajectory | 1.177520 | -- |
| Same-position average | 1.177505 | -0.0013% |

Both modes retain active Channel-Patch anchors in every source epoch. The
positionwise change misses the registered `-0.25%` expansion threshold by two
orders of magnitude. It is not expanded to additional seeds, baselines, or
test data.

Machine-readable audit:
`artifacts/exploratory/pems03_channel_patch_positionwise_probe_20261007/audit.json`
