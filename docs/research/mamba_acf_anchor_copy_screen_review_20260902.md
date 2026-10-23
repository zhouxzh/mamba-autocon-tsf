# Direct-Anchor Screen Review

**Status:** `exploratory_mamba_acf_anchor_copy_screen_draft`; not authorized for training.

**Plan:** [`mamba_acf_anchor_copy_screen_plan_20260902.json`](./mamba_acf_anchor_copy_screen_plan_20260902.json)

## Estimand

The screen asks whether an independent signed historical anchor helps the
existing variable-token MambaACF AutoCon forecaster. The reference and
treatment use the same model alias, executable model, raw train-only ACF
sidecar, hidden memory gate, relation objective, split, optimizer, seeds and
eight-epoch budget. The only planned change is `mamba_acf_anchor_copy`.

For every forecast step `h` and channel `c`, the treatment chooses the
available configured lag with maximum absolute train ACF, reads
`i = L - 1 + h - lag`, applies the stored ACF sign, and adds the result with a
single scalar ReZero parameter initialized at zero. Unavailable indices are
masked. This is a causal retrieval contract, not a learned future-value
generator.

## Why This Is Separate

The existing soft memory averages several lags and then applies shared gate
and reliability factors. A direct per-channel anchor tests whether that
aggregation is diluting a useful seasonal value. It is kept separate from the
hidden-versus-variable-token gate screen so any paired delta has one
interpretable architectural cause.

The branch is deliberately small: one trainable scalar, indexed gather, no
dense `[B,H,C,K]` tensor, and no change to the AutoCon relation representation.
The sidecar must contain the complete train-only ACF table so evaluation cannot
silently refit a different prior.

## Evidence Boundary

No anchor-copy job has been authorized or trained. The plan expands to 12 jobs
and passes validate-only checks. The read-only analyzer returns `NOT_RUN` and
does not write a readout while phase manifests and validation files are absent.
Structural tests cover per-channel lag selection, causal masking, zero-scale
nested initialization, gradient flow, source restrictions, command forwarding,
sidecar requirements, and resource metadata.

Any future readout is limited to paired descriptive MSE/MAE deltas, anchor
availability/scale diagnostics, artifact hashes, and separately measured
resource costs. Six H96 coordinates cannot establish a general ACF, Mamba,
speed, or publication claim.

## Human Review Before Launch

- Confirm the direct-anchor estimand and the strict causal index/mask.
- Recheck raw dataset and sidecar hashes and the fresh output-root condition.
- Freeze this exact plan and create an approved launch sidecar with evidence
  hashes for every prelaunch gate.
- Run the separate resource benchmark; do not infer end-to-end speed from the
  zero-MAC selection estimate.
- Do not select lags, scales, horizons, or follow-up arms from test metrics.
