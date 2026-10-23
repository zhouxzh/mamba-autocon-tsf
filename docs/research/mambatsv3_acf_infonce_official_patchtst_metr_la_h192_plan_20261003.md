# Official PatchTST METR-LA H192 External-Code Extension Plan

Pinned source: `yuqinie98/PatchTST` commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` (Apache-2.0), imported from
`PatchTST_supervised/models/PatchTST.py` through the chronological adapter.
METR-LA H192 uses seeds 5440--5442, 10% target prefix, `seq_len=96`, five
epochs, 64 windows, and official patch geometry (`patch_len=16`, `stride=8`).

Validation: active `1.573968`, official PatchTST `1.660983`, `-5.239%`, 3/3.
The test extension loads validation-selected official checkpoints once and may
not change candidate, split, source selection, or adapter parameters.
