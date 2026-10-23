# ACF Channel Ordering

`MambaACF` variable tokens are a sequence, so a fixed channel order can affect
the cross-variable Mamba recurrence. The optional
`--mamba_acf_channel_order strength` mode addresses this with a train-only,
deterministic order:

1. `set_acf_prior()` receives the signed ACF table fitted on the training
   split.
2. For each channel, the dominant peak is the selected lag with the largest
   absolute ACF magnitude. Its signed value is retained for audit and ACF
   conditioning.
3. Channels are sorted by decreasing peak magnitude, then increasing peak lag,
   then original channel index. The last two keys make ties reproducible.
4. Variable tokens are reordered before optional grouping/permutation and are
   restored to input order before the forecast head. No target, validation, or
   test value is read to form the order.

The default `input` mode is an identity path and preserves the historical
forward order. If an ACF-derived mode is enabled before a train prior is
attached, the model also uses identity order; it never estimates an order from
a runtime window. ACF-derived ordering requires the variable-token branch
because temporal-only Mamba has no channel-axis sequence to reorder.

The optional `--mamba_acf_channel_order similarity` mode uses the complete
selected-lag profile instead of one dominant peak. It L2-normalizes each
signed profile, starts from the largest profile norm, and greedily appends the
remaining channel with the largest signed cosine similarity to the current
endpoint. The implementation performs `O(C^2 K)` arithmetic with `O(CK)`
temporary memory (`C` channels, `K` selected lags); it does not create a dense
`C x C` matrix and runs only when the train-only prior is attached. This is a
deterministic seriation heuristic, not a learned graph or a cross-correlation
estimate, so it remains an independent order ablation with input-order and
permutation/reverse robustness controls.

The order and inverse are non-persistent buffers. They are recorded in
`resource_profile()` with an explicit rule name and included in run/resource
architecture signatures, so `input`, `strength`, and `similarity` cannot be
silently pooled in an analysis table.

This is an architectural ablation, not an assumption that stronger ACF always
improves forecasting. It must be compared with the input-order branch under
the same seed, training budget, and validation selection rule.
