# Static ACF Reliability Interpolation Contract

The opt-in `mamba_acf_reliability_interpolation` branch adds one trainable scalar to the MambaACF forecast memory. It interpolates between the horizon-aggregated train-only ACF strength and the strength available at each forecast step and channel:

`amp[h,c] = exp(tanh(alpha) * clip(log((step[h,c]+eps)/(aggregate[c]+eps)), -log(2), log(2)))`.

`alpha` starts at zero, so the disabled path and the enabled path are exactly nested before optimization. The factor is computed from cached, detached static-prior statistics and causal validity only. It does not estimate ACF from the current input and does not inspect future labels.

The branch is valid only for static raw memory (`mamba_use_acf_memory=true`, `mamba_dynamic_acf=false`, `AutoCon_acf_source=raw`) with `mamba_acf_step_reliability=false`. It is orthogonal to the AutoCon relation objective and does not change the ACF prior artifact schema.

Validation coverage:

- model constructor: invalid combinations are rejected;
- direct CLI and training-plan validators: the same constraints are enforced;
- phase validator: true values must appear in the recorded config;
- resource benchmark: on/off values are part of the architecture signature and manifest grouping;
- AutoCon training logs: mean/std/min/max of the learned amplitude are persisted per epoch;
- regression: zero-initialized nesting, nonzero finite gradient, profile metadata, command forwarding, and manifest separation.

No accuracy training has been launched for this candidate. The registered H96 screen is deliberately exploratory and unauthorized pending human review.
