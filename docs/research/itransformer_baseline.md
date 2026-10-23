# iTransformer baseline

`models/iTransformer.py` provides a small, dependency-free inverted
Transformer baseline for the benchmark matrix. Each variate is treated as a
token: an input window `[B, T, C]` is transposed to `[B, C, T]`, projected
along the history axis, mixed by a standard PyTorch `TransformerEncoder`, and
projected back to `[B, H, C_out]`.

The implementation is intentionally a fair supervised comparator. It uses
the same normalization, dataset split, horizon, optimizer, and generic
forecasting trainer as the existing DLinear/PatchTST/TimesNet baselines. It
does not use the AutoCon objective or the ACF prior; those remain properties
of the proposed `MambaACFAutoCon` arm. `M` predicts all channels, while `S`
and `MS` preserve the repository's target-channel contract.

This is a lightweight reproduction of the inverted-token design, not a claim
of exact parity with every upstream iTransformer implementation. Report it as
an independently implemented baseline and keep width/depth and training
budget explicit in the experiment manifest.
