# Official TimeMixer H336 External-Code Extension Plan

Pinned source: `kwuking/TimeMixer` commit
`e24610583b36fdd8c76cc17a8df4e65759a5f460` (Apache-2.0), imported from
`models/TimeMixer.py` through the local chronological adapter. PEMS-BAY H336
uses seeds 5413--5415, `seq_len=96`, 10% target prefix, five epochs, 64
windows, channel-dependent multiscale mixing, down-sampling window 2/layer 1,
one PDM layer, `d_model=24`, and `d_ff=48`.

Validation: active `2.386473`, official TimeMixer `2.543952`, `-6.190%`, 3/3.
Test reads each validation-selected official checkpoint once without changing
candidate, source, or adapter configuration.
