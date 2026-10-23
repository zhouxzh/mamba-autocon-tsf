# Local-Cluster Robustness Extension

## Scope

This is a post-confirmation, validation-only robustness extension for the
frozen 10%-label checkpoints. It does not train, select, modify, or retest a
checkpoint. Its purpose is to test whether the active versus exact-null
mechanism direction persists under input-side perturbations in the already
closed chronological validation block.

## Checkpoints and Conditions

| Target | Seeds | Source pool |
| --- | --- | --- |
| PEMS-BAY | 5510--5512 | PEMS07, PEMS08, PEMS04 |
| METR-LA | 5520--5522 | PEMS-BAY, PEMS07, PEMS08 |
| Weather | 5530--5532 | METR-LA, PEMS-BAY, Electricity |

Each active/null checkpoint is evaluated on the same six deterministic
conditions: ordinary validation, late validation origins, 10% and 30% missing
input channels, and signal-relative Gaussian noise at 0.05 and 0.10. The
evaluator uses the target's saved few-shot normalization and never accesses the
chronological test block.

## Interpretation Boundary

This extension can strengthen or weaken the robustness rationale, but it is
not a pre-test registration and cannot alter the already completed held-out
effect estimates. All results, including any negative condition, are retained.
