# Residual-Target AutoCon Confirmatory Release

## Authorization And Boundary

This release follows the completed three-seed validation bundle documented in
`mambatsv3_residual_autocon_validation_20261003.md`. It opens the untouched
target test block only after source selection, primary selection, and target
best-epoch selection are complete. Test results cannot change source pools,
primary models, target epochs, architecture, or loss settings.

## Frozen Coordinates

| Target | Few-shot | Source selection | Primary selection | Test seeds |
| --- | ---: | --- | --- | --- |
| Traffic H96 | 10% | top-3 ACF from target 10% train prefix | DLinear/NLinear inner calibration tail | 5201, 5202, 5203 |
| Weather H96 | 10% | top-3 ACF from target 10% train prefix | DLinear/NLinear inner calibration tail | 5204, 5205, 5206 |

The candidate uses exact-H96 residual head transfer, source affine ACF loss
at 10% of predictive loss, one Mamba scan, and target-uniform windows. The
selected-primary linear control shares every target split, optimizer, target
epoch budget, and seed. iTransformer is trained from scratch as an external
low-data comparator under the same target split/budget. DLinear/NLinear
selection results and all source-selection artifacts are persisted before any
test loader is instantiated.

## Test Gate

Per target, candidate versus selected-primary control must have mean test MSE
below control, at least two paired seed wins, and a paired one-standard-error
upper MSE ratio below 1.02. iTransformer is descriptive strong-baseline
context; failure against it remains reportable and cannot trigger retuning.

Each job is serial, CPU-limited to four threads, and GPU guarded at 78C/85%
fan. The release is invalid if a required artifact, selector record, or guard
event is missing.

The reader binds Traffic to the pre-existing few-shot-only selection artifact
that selected Electricity/METR-LA/PEMS-BAY and Weather to the artifact that
selected METR-LA/PEMS-BAY/Electricity. These artifacts were created before
the confirmatory test loop and declare both validation and test access false.

## Invalid Smoke Boundary

One implementation smoke invoked `--evaluate-test` with only one target epoch
and eight target windows. Its payload correctly records a test metric, but it
does not satisfy this plan's frozen coordinates and is excluded from all
selection, comparison, summary, and manuscript claims. No confirmatory
coordinate uses that smoke artifact.
