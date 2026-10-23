# Channel Order and Conditioning Screen Launch Review

Date: 2026-09-02 (Asia/Shanghai)

This note records the launch decision for the independent 18-job exploratory
screen in `mamba_acf_channel_order_conditioning_screen_plan.json`.

## Design Review

- The control arm uses the input CSV order and disables channel conditioning.
- The strength arm changes only `mamba_acf_channel_order` to `strength`. The
  model derives a deterministic order from the attached train-only ACF prior,
  and restores the original channel layout after the variable scan.
- The conditioning arm retains input order and changes only
  `mamba_acf_channel_conditioning=true`. Its four-statistic adapter is
  zero-initialized, so activation and scale must be inspected before any
  mechanism interpretation.
- All three arms use the same `MambaACFVariableAutoCon` alias, executable
  `MambaACFAutoCon`, raw train-only sidecar, H96 split, global hard relation
  target, lambda `0.03`, sampler, optimizer, and eight-epoch budget.
- The strength-plus-conditioning interaction is deliberately omitted. It must
  not be inferred or added after looking at the test metrics.

## Evidence Boundary

This is a paired structural screen on ETTh1 and Traffic with seeds 2021--2023.
It is not confirmatory evidence. The test metric is recorded for a fixed
descriptive readout only; it cannot select an arm, hyperparameter, or follow-up
experiment. Rows must not be pooled with the completed variable-token or
PatchTST baseline plans.

## Authorization

The user's continuation request and the parent-agent authorization in this
session explicitly permit this exploratory launch. This authorization covers
only the 18 jobs in the named plan and the fresh output root
`artifacts/paper_final/mamba_acf_channel_order_conditioning_screen_20260902`.
It does not authorize a confirmatory run or a publication claim.
