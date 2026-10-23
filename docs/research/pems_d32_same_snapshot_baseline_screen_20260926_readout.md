# PEMS d32 same-snapshot baseline screen

Date: 2026-09-26

Protocol: PEMS03/04/07/08, M features, H96, seeds `2181--2183`, 20 epochs,
train-only normalization, validation-only, `skip_test=true`.

## Integrity

All 84 registered jobs completed successfully in one frozen source snapshot.
Strict phase validation passed for strong baselines (`60/60`), the exact
lambda-zero control (`12/12`), and active AutoCon (`12/12`). No test metrics
were generated.

## Forecast Validation MSE

Values are means of the seed-level minimum pure-forecast validation MSE values.

| Dataset | DLinear | PatchTST | TimesNet | iTransformer | TimeMixer | d32 null | d32 active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS03 | 1.059796 | 0.512242 | **0.128559** | 0.289209 | 0.248376 | 0.156476 | 0.156462 |
| PEMS04 | 1.224827 | 0.726009 | **0.149939** | 0.421641 | 0.242396 | 0.200833 | 0.200517 |
| PEMS07 | 1.046834 | 0.619802 | **0.196790** | 0.345864 | 0.314982 | 0.218182 | 0.218207 |
| PEMS08 | 1.204915 | 0.754352 | 0.362501 | 0.485832 | 0.448830 | 0.358935 | **0.358352** |

Against the strongest same-snapshot baseline (TimesNet in every PEMS dataset),
d32 active is worse by `21.704%`, `33.732%`, and `10.883%` on PEMS03/04/07,
and better by `1.145%` on PEMS08. It is therefore non-inferior in only one of
four datasets. The registered stopping rule is triggered: more than two
dataset means exceed the 2% margin.

## AutoCon Paired Contrast

The active arm changes only `AutoCon_lambda` from zero to `0.001`. Relative
active-minus-null MSE changes are PEMS03 `-0.010%` (2/3 seed wins), PEMS04
`-0.160%` (3/3), PEMS07 `+0.012%` (1/3), and PEMS08 `-0.159%` (2/3). These
effects are small and directionally mixed, so this screen does not support a
stable causal AutoCon accuracy claim for d32 MambaTS v3.

## Decision

The d32 full-coverage MambaTS v3 branch is stopped for PEMS H96 as a general
paper candidate. Its test split remains unopened. The PEMS08 result and the
low-latency engineering measurement may be retained as exploratory evidence,
but cannot be pooled into a universal accuracy or self-supervised-learning
claim. Any successor must introduce a pre-specified mechanism that addresses
the large periodic-model gap on PEMS03/04/07 and must be tested under a new
source snapshot.

Artifacts:
[`strong baselines`](../../artifacts/exploratory/pems_d32_same_snapshot_baseline_screen_20260926/strong_baselines/),
[`null`](../../artifacts/exploratory/pems_d32_same_snapshot_baseline_screen_20260926/null/),
[`active`](../../artifacts/exploratory/pems_d32_same_snapshot_baseline_screen_20260926/active/).
