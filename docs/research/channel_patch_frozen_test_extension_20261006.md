# Channel-Patch Frozen Test Extension

Status: read-only post-validation mechanism extension. Every candidate,
forecast-only, null, and official PatchTST checkpoint was SHA-256 recorded in
`manifest_before_test.json` before the reader ran. The reader performed no
training or selection. This workspace has historical public test access, so
the results are explicitly not a pristine first-open test evaluation and must
not be used for post-test model selection.

## Test Results

| Target | Channel-patch vs forecast | Wins | Upper | Channel-patch vs PatchTST | Wins | Upper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | -1.360% | 3/3 | 0.98716 | -3.181% | 3/3 | 0.97394 |
| METR-LA H192, 10% | -0.407% | 3/3 | 0.99760 | -4.039% | 3/3 | 0.96562 |
| Weather H192, 10% | +0.277% | 2/5 | 1.00510 | -2.800% | 5/5 | 0.97571 |

The channel-patch objective also remains favorable to exact null on PEMS-BAY
(`-3.680%`, 3/3) and METR-LA (`-0.665%`, 3/3). On Weather it is slightly lower
than null on mean MSE (`-0.110%`) but fails the one-SE noninferiority rule
(`1.00122`).

## Interpretation

PEMS-BAY and METR-LA reproduce the validation mechanism ordering on frozen
checkpoints. Weather does not: despite five test wins against official
PatchTST, the channel-patch extension loses its validation advantage relative
to the already strong forecast-level ACF objective. Therefore Weather cannot
be used as a confirmed channel-patch causal gain. It remains an external
baseline and robustness result, while the representation-mechanism claim must
be narrowed to PEMS-BAY and METR-LA until it is independently replicated on
new data or an independent machine.

The correct paper treatment is to report the Weather test reversal alongside
the two positive traffic results, not to average it away or use it to tune a
new variant.

Artifacts:

- `artifacts/paper_final/channel_patch_frozen_test_extension_20261006/manifest_before_test.json`
- `artifacts/paper_final/channel_patch_frozen_test_extension_20261006/audit.json`
