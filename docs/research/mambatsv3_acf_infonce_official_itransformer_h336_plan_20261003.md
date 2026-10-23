# Official iTransformer H336 External-Code Extension Plan

Pinned source: `thuml/iTransformer` commit
`c2426e68ca13f74aaec08045c5c724d8ad328124` (MIT), imported from
`model/iTransformer.py` through the local chronological adapter. PEMS-BAY H336
uses seeds 5413--5415, 10% target prefix, `seq_len=96`, five epochs, 64
training windows, one official encoder layer, `d_model=24`, and four heads.

Validation: active `2.386473`, official iTransformer `2.700596`, `-11.632%`,
3/3. Test reads each validation-selected official checkpoint once and cannot
change candidate or adapter parameters.
