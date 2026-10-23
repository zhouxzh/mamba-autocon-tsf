# Official iTransformer METR-LA H192 External-Code Extension Plan

Pinned source: `thuml/iTransformer` commit
`c2426e68ca13f74aaec08045c5c724d8ad328124` (MIT), imported from
`model/iTransformer.py` through the chronological adapter. METR-LA H192 uses
seeds 5440--5442, 10% target prefix, `seq_len=96`, five epochs, 64 windows,
one encoder layer, `d_model=24`, four heads, and `d_ff=48`.

Validation: active `1.573968`, official iTransformer `1.729178`, `-8.976%`,
3/3. Test reads validation-selected official checkpoints once and cannot alter
candidate, split, source selection, or adapter parameters.
