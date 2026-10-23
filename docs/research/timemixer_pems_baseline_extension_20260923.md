# TimeMixer PEMS Baseline Extension

## Rationale

The completed PEMS confirmation matrix compares DLinear, PatchTST, TimesNet,
Mamba2, and MambaTS-v3 AutoCon variants, but omits a modern efficient
multi-scale MLP comparator. TimeMixer is added only as a baseline; it does not
change the stopped MambaTS channel-adapter decision and it contains no
AutoCon branch.

The local implementation preserves the official architecture's decomposed
multi-scale bottom-up seasonal and top-down trend mixing. It uses the same
MeanStd population-variance window normalization as the frozen PEMS matrix.
The model is deliberately channel-correlated (`channel_independence=false`),
with average-pooling downsampling ratio two and one additional scale.

## Frozen validation contract

- Data: PEMS03, PEMS04, PEMS07, PEMS08, using the identical NPZ hashes and
  feature-axis contracts as the completed confirmation matrix.
- Coordinates: M features, input 96, horizons 12 and 96, seeds
  `2021` through `2025`.
- Budget: at most 20 epochs, patience 5, batch 32, same optimizer and
  `MeanStd` normalization.
- Total: `4 x 2 x 5 = 40` validation-only jobs; `skip_test=true`.
- Primary metric: minimum pure forecast validation loss from the checkpoint
  loss log. No test metric may be produced or used for configuration choice.

## Interpretation boundary

This extension reports TimeMixer against the already frozen PEMS baselines
descriptively. It cannot revise the channel-adapter failure decision. It does
not establish a new Mamba claim and cannot select an AutoCon loss weight or
architecture. The result is eligible only as a stronger baseline row for a
future independently frozen Mamba candidate.
