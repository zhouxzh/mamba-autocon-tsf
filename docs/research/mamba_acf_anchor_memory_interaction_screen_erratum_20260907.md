# Anchor x Memory Interaction Screen Erratum

**Date:** 2026-09-07  
**Scope:** `mamba_acf_anchor_memory_interaction_screen_plan_20260904.json`  
**Immutable plan SHA-256:** `feb402ba20197715f4e64fa05f843392305154c1ced586a9b0269261a05f7211`

The plan JSON is hash-bound to completed exploratory artifacts and is intentionally not
edited. Its `comparison_contract.interaction_sign` sentence reverses the interpretation
of the registered difference-in-differences:

```text
I = (anchor_memory - memory_only) - (anchor_only - plain)
```

Because lower validation MSE is better, `I < 0` means the combined arm lowers error more
than the sum of the isolated effects (super-additive/complementary); `I > 0` means a
sub-additive or interfering combination. The formula, the analyzer, the protocol, and the
readout use this corrected interpretation.

The plan metadata still says `authorized=false` and `training_started=false`. That was the
state of the immutable draft, not a statement that no artifacts exist: the four planned
phases were later run post-hoc under frozen source snapshot
`2837199515e0ee7fc718c9472c99d42ee3d32d00f65d98f6ae228d1e133144a5`, each with six
successful jobs and a valid phase-validation record. This execution remains exploratory,
validation-only, and non-confirmatory; it was not launch-time preregistered.

The resulting interaction is directionally inconsistent across ETTh1 and Traffic and its
pooled 95% CI crosses zero. It must not be used to claim synergy, select a test arm, or
justify stacking additional mechanisms.
