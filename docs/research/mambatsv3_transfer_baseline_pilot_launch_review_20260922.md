# MambaTS v3 Transfer Baseline Pilot Launch Review

Scope: the validation-only 54-job pilot registered in
`mambatsv3_transfer_baseline_pilot_20260922.json`. It keeps all six models,
three seeds, and test exclusion. It is not authorization for the separate
144-job extension or for a test release.

Reviewed gates:

- Current Weather, Exchange-Rate, and ILI SHA256 values match the immutable
  fingerprints in the plan.
- Every coordinate includes DLinear, PatchTST, TimesNet, iTransformer,
  Mamba2, and MambaTSv3AutoCon, so no unfavorable baseline can be omitted.
- MambaTSv3AutoCon retains `AutoCon_lambda=0.001`, MeanStd, and the existing
  representation objective. The pilot reports its telemetry but cannot claim
  that the objective itself supplies an accuracy gain.
- `skip_test=true` is required for every job. Results may only be used to
  decide whether the larger validation-only extension is justified.
