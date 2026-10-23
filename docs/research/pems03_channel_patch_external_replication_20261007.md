# PEMS03 Target-Excluded Channel-Patch Replication

## Scope

This registered validation-only replication tests current `24/8` Mamba-1
channel-patch ACF representation pretraining on PEMS03 at H192 with 10% of the
chronological training prefix. It is deliberately not a test evaluation.

PEMS03 is excluded from all three pretraining sources and from both frozen
initialization checkpoints. Sources are PEMS04, PEMS08, and PEMS07;
initialization checkpoints are PEMS04 and PEMS08. The runner asserts this
exclusion before every pretraining launch, and the read-only audit checks the
stored command for each seed again.

## Registered Comparison

Three fresh seeds (`6055`, `6056`, `6057`) compare matched frozen-core target
heads after null, forecast-only, and channel-patch source pretraining. The
external comparator is the locked official PatchTST supervised implementation
through the local chronological adapter. It uses the same PEMS03 array, label
fraction, horizon, and validation boundary. Neither the Mamba arms nor PatchTST
receives `--evaluate-test`.

| Arm | Mean validation MSE | Channel-patch difference | Wins for channel-patch |
| --- | ---: | ---: | ---: |
| Null | 1.211922 | -0.526% | 3/3 |
| Forecast pretraining | 1.209576 | -0.333% | 3/3 |
| Channel-patch ACF SSL | 1.205552 | -- | -- |
| Official PatchTST | 1.255526 | -3.980% | 3/3 |

The one-standard-error upper ratios are `0.99682` versus forecast pretraining
and `0.96244` versus PatchTST, so both registered noninferiority checks pass.

## Validation Stresses

The frozen checkpoints were then evaluated on the same chronological validation
block under a late-origin shift, 10% and 30% fixed channel masking, and
signal-relative Gaussian noise at `0.05` and `0.10`. Channel-patch remains
noninferior to same-seed official PatchTST in all six conditions, with 3/3
wins per condition. The relative improvements versus PatchTST are `-3.980%`,
`-3.654%`, `-3.622%`, `-2.902%`, `-3.946%`, and `-3.846%` in registered order.
No model was retrained and no test partition was opened for this matrix.

## Interpretation

This is positive evidence that the shared channel-patch representation transfers
to a fourth traffic-network target without target-domain source or
initialization exposure. It is not an end-to-end Q1 submission result: the
result is validation-only, has no PEMS03 perturbation extension, does not
resolve the Weather frozen-test reversal, and does not remove the historical
test-access caveat.

Machine-readable result:
`artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007/audit.json`

Robustness audit:
`artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007/robustness_validation/audit.json`
