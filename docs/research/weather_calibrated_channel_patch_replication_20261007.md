# Weather Calibrated Channel-Patch Replication

## Protocol

This is a new, validation-only Weather H192/10% protocol, registered after the
unadapted Channel-Patch frozen-test extension. It does not replace or amend
that historical test result. During source pretraining, each arm additionally
uses only the earliest Weather 10%-label training prefix (3,161 rows) as a
target-calibration domain. Weather validation and test rows are never opened.

Both arms use the same METR-LA, PEMS-BAY, and Electricity source pool, source
forecast InfoNCE, Weather calibration next-patch/forecast InfoNCE, frozen
shared core, and target head schedule. The treatment alone adds the training-
only full channel-patch ACF InfoNCE at fraction `0.05` to both source and
calibration data. Inference remains the unchanged one-scan prediction path.

## Independent Replication

| Comparison | Validation MSE | Change | Wins | One-SE upper |
| --- | ---: | ---: | ---: | ---: |
| Channel-Patch vs calibrated forecast | 4.510126 vs 4.522796 | -0.280% | 3/3 | 0.99757 |
| Channel-Patch vs official PatchTST | 4.510126 vs 4.620109 | -2.381% | 3/3 | 0.98022 |

The three fresh seeds are `6073--6075`. All source and calibration epochs
record active Channel-Patch anchors. The official PatchTST comparator uses the
same target split, label budget, horizon, target-window budget, seeds, and
chronological validation boundary.

The same-coordinate official iTransformer comparison is also favorable:
candidate/official means are `4.510126`/`5.346460` (`-15.643%`, 3/3 wins,
upper ratio `0.84744`). Official TimeMixer is the required boundary: its mean
is `4.508596`, candidate is `+0.034%`, wins only 1/3, and upper ratio is
`1.00889`. The calibrated protocol therefore does not claim universal external
superiority.

At 25% labels, the same calibrated protocol is only marginally favorable
(`-0.057%`, 2/3 wins; upper ratio `0.99998`) and fails its all-seed promotion
gate. It is retained as a label-budget boundary rather than expanded to new
external baselines or stresses.

The target-channel residual gate does not close the TimeMixer boundary. On the
registered seed `6073`, it reaches `4.507823`, `+0.046%` relative to scalar
Channel-Patch and `+0.610%` relative to TimeMixer despite nonzero learned gates.
It is not expanded.

## Validation Stresses

The frozen checkpoints are favorable to calibrated forecast-only and externally
noninferior to official PatchTST across all six registered validation
conditions: later validation origin, 10%/30% missing channels, and 0.05/0.10
relative noise. Channel-Patch wins all 3/3 pairs in every condition. Its
smallest PatchTST margin is `-1.067%` at 30% missing channels, with upper ratio
`0.99091`.

## Scope Boundary

This is positive support for target-prefix calibrated representation transfer
on Weather. It is not a pristine test result and cannot be used to overturn
the earlier unadapted Weather frozen-test reversal. A new test protocol would
need a fresh, independently governed held-out partition rather than the
historically exposed test artifact.

Machine-readable artifacts:

`artifacts/exploratory/weather_channel_patch_calibration_replication_20261007/audit.json`

`artifacts/exploratory/official_patchtst_weather_channel_patch_calibration_20261007/audit.json`

`artifacts/exploratory/official_itransformer_weather_channel_patch_calibration_20261007/audit.json`

`artifacts/exploratory/official_timemixer_weather_channel_patch_calibration_20261007/audit.json`

`artifacts/exploratory/weather_channel_patch_calibration_replication_20261007/robustness_validation/audit.json`
