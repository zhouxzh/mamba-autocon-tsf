# Current-tree validator audit (2026-09-02)

This is a read-only audit of the historical three-phase live screen under the
current validator. It is intentionally separate from the historical
`phase_validation.json` files and does not rewrite them.

| phase | expected jobs | current-validator status | reason |
|---|---:|---|---|
| `variable_plain_h96_screen` | 6 | invalid | `scripts/run_training_plan.py` differs from the immutable phase snapshot |
| `variable_memory_h96_screen` | 6 | invalid | `scripts/run_training_plan.py` differs from the immutable phase snapshot |
| `variable_memory_relation_h96_screen` | 6 | invalid | the runner snapshot differs; all six records also lack the repaired sampler metadata and realised lag-pair telemetry |

The exact JSON outputs are in
[`mamba_acf_live_screen_current_validator_audit_20260902`](../../artifacts/exploratory/mamba_acf_live_screen_current_validator_audit_20260902/).
The relation errors include missing `requested`, `effective`, and `reason`
sampler fields, selected-lag support, and per-lag pair counts/exposure. This
is a fail-closed result: the old logs remain useful for their original
exploratory snapshot, but they cannot be promoted to a strict relation
aggregate without rerunning under the repaired contract.

This audit should be read alongside the separate schema-4 data-bound replay.
That replay passed `18/18` checkpoint evaluation and metric checks; it answers
whether the historical inputs can be loaded and reproduced, whereas this
audit answers whether their persisted training metadata satisfies the current
contract.
