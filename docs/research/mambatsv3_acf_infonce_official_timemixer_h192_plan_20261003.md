# Official TimeMixer H192 External-Code Extension Plan

- Repository: `kwuking/TimeMixer`
- Immutable commit: `e24610583b36fdd8c76cc17a8df4e65759a5f460`
- License: Apache-2.0
- Imported network: `models/TimeMixer.py`

The official runner uses a fixed 70/10/20 custom loader and trains then tests.
The adapter imports the pinned official model while enforcing 60/20/20
chronology, 10% target prefix, validation-first checkpoint selection, and a
separate single test read.

PEMS-BAY H192 uses seeds 5410--5412, `seq_len=96`, five epochs, 64 windows,
one PDM layer, channel-dependent mixing, down-sampling window 2/layer 1,
`d_model=24`, and `d_ff=48`. Validation result: active `2.744908`, official
TimeMixer `2.869654`, `-4.347%`, 3/3. Test cannot alter model, adapter, or
candidate settings.
