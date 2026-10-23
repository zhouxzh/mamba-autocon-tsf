# Official PatchTST H192 External-Code Extension Plan

## External Source

- Repository: `yuqinie98/PatchTST`
- Immutable commit: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- License: Apache-2.0
- Imported network: `PatchTST_supervised/models/PatchTST.py`

The unmodified official runner cannot be used because it fixes a 70/10/20
split and tests directly after training. The local adapter imports the pinned
official network unchanged, while enforcing the audited 60/20/20 chronological
split, 10% target prefix, validation-only checkpoint selection, and a single
test read after selection.

## Frozen Coordinates

PEMS-BAY H192, seeds `5410/5411/5412`, `seq_len=96`, 10% target prefix,
five epochs, 64 training windows, official PatchTST default patch geometry
(`patch_len=16`, `stride=8`), and the PEMS-BAY H192 audited active checkpoint
as the fixed comparator.

Validation-only official adapter result: active `2.744908`, official PatchTST
`2.764765`, `-0.718%`, 3/3 active wins. Validation artifacts contain no test
metric. The post-selection test run must preserve all three seeds and cannot
trigger a candidate retrain, source change, or hyperparameter change.

## Interpretation

This is an external-model-code confirmation with a local boundary adapter. It
is not a bitwise reproduction of the upstream runner, because the upstream
split/test lifecycle conflicts with the audited experimental contract.
