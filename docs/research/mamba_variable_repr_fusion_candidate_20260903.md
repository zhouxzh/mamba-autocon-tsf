# Variable-Token Representation Fusion Candidate

## Scope

`mamba_variable_repr_fusion` is an opt-in AutoCon adapter for the shared
MambaACF variable-token branch. It is valid only for `MambaACFAutoCon` or
`MambaACF1AutoCon` with `mamba_variable_mixer=true`; the default is `false`.
The plain forecast path and all existing configurations remain unchanged when
the switch is absent or disabled.

## Data Flow

The normalized encoder window is converted to one token per channel and passed
through the existing variable Mamba branch. The resulting `[B,C,D]` states are
softmax-weighted over channels, projected with a small `D -> D -> D` adapter,
and broadcast as a near-zero gated residual over the AutoCon temporal representation
grid. The same token realization is reused by the forecast branch when a
representation is requested, so relation-loss gradients have an explicit path
to the variable embedding, variable Mamba blocks, pooling, and projection.

The adapter uses only the normalized observed window. It does not read target,
validation, or test values, and it is not a trainable dense `C x C` map. The
small non-zero initialization keeps the candidate close to the legacy
representation at startup while allowing relation gradients to reach the
variable branch on the first update.

When two arms use the same seed, construction of this optional adapter is
isolated from the shared backbone RNG. The fusion-off/on comparison therefore
starts with bitwise-identical shared parameters; only the registered adapter
is additionally initialized in the treatment arm.

## Audit Boundaries

The resource profile records adapter parameters, activation/MAC proxies,
source shape, and the `window_pooled_variable_token_representation_context`
causality limit. Because a complete encoder window is summarized and broadcast,
this option cannot be claimed as a strictly prefix-causal representation.
No accuracy or speed result is implied by this implementation. A paired,
train-only, unauthorized off/on screen is required before making any claim.
The registered screen uses the explicit PatchTST-compatible `MeanStd` window
contract so the representation comparison is not coupled to the historical
`LastVal` scale choice.

The registered screen is validation-only. Its log must distinguish merely
having an ACF-positive pair from having a usable contrastive row with at least
one in-range denominator negative. A one-candidate positive row has zero
magnitude contrastive cross-entropy and cannot support a representation-learning
mechanism claim.
