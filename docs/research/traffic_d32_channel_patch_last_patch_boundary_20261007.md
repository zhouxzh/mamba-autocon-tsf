# Traffic d32 Final-Patch Channel-Patch Boundary

The canonical Channel-Patch objective compares complete encoder patch-state
trajectories for two windows selected by train-prefix ACF. This validation-only
screen tests a more prediction-aligned state: it compares only each channel's
last causal patch state, the same state immediately preceding the
next-patch decoder. The ACF pairs, encoder, source/calibration data, frozen
target core, prediction loss, Mamba scan count, and inference graph remain
unchanged.

| Channel-Patch similarity | Validation MSE | Difference |
| --- | ---: | ---: |
| Full patch trajectory | 0.825574 | -- |
| Final causal patch | 0.825462 | -0.014% |

The final-patch loss is active in every source epoch (`0.885`, `0.906`, and
`0.917` active anchors). Although its direction is favorable, it misses the
registered `-0.25%` expansion threshold by a large margin. It is not expanded
to fresh seeds, robustness cells, external baselines, or test data; the
default remains the full-trajectory objective.

All four guarded phases completed without a trigger and peaked at 51C. The
similarity choice exists only during pretraining and is absent from deployed
inference. Machine-readable evidence:
`artifacts/exploratory/traffic_d32_channel_patch_last_patch_probe_20261007/audit.json`.
