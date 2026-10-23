# Official iTransformer H192 External-Code Extension Plan

- Repository: `thuml/iTransformer`
- Immutable commit: `c2426e68ca13f74aaec08045c5c724d8ad328124`
- License: MIT
- Imported network: `model/iTransformer.py`

The upstream runner hard-codes a 70/10/20 custom split and invokes test after
training. The adapter imports the pinned official network but enforces the
audited 60/20/20 chronological split, 10% target prefix, validation-selected
checkpoint, and a separate single test read.

Frozen coordinates: PEMS-BAY H192, seeds 5410--5412, `seq_len=96`, five
epochs, 64 training windows, official `d_model=24`, one layer, four heads, and
fixed comparison against the audited active H192 checkpoints. Validation:
active `2.744908` versus official iTransformer `2.944634` (`-6.783%`, 3/3).
No test result may alter sources, candidate checkpoints, or adapter settings.
