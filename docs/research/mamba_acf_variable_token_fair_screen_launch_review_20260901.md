# Variable-token Fair Screen Launch Review

Date: 2026-09-01 (Asia/Shanghai)

This note records the launch decision for the exploratory 30-job variable-token
screen in `mamba_acf_variable_token_fair_screen_plan.json`.

## Review decisions

- The relation null is implemented with the same `MambaACFVariableAutoCon`
  executable architecture, train-only raw ACF sidecar, lag sampler, split,
  optimizer, and budget as the `lambda=0.03` arm. Only `AutoCon_lambda` differs
  (`0.0` versus `0.03`). This is the only relation-effect contrast that may be
  interpreted as a mechanism control.
- The memory-on/off contrast and the null-to-memory contrast remain descriptive
  because they change the executable branch or objective activation.
- The sidecar is fit on the training split only and is required for every arm;
  its identity and effective lags must match at each dataset/seed coordinate.
- The plan is limited to ETTh1 and Traffic at H96, three seeds, and eight
  epochs. No test metric may select an architecture, lambda, or follow-up run.

## Authorization

The user's continuation request in this session is treated as explicit
authorization to launch this exploratory screen. This is not authorization for
a confirmatory study or a publication claim. The output root is fresh at launch;
failed or invalid artifacts must be preserved and any rerun must use a new
root.

