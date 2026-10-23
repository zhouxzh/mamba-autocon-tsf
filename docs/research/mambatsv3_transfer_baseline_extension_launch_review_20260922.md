# MambaTS v3 Transfer Extension Launch Review

Scope: the 144-job validation-only matrix in
`mambatsv3_transfer_baseline_extension_20260922.json`.

The matrix retains all six model arms on Weather and Exchange-Rate at H96/H336,
and on ILI at H24/H36/H48/H60. Every coordinate has three seeds and uses the
same M-feature, MeanStd, eight-epoch, batch-size-32 contract. MambaTS v3
retains the existing AutoCon representation objective at `lambda=0.001`.

The launch is exploratory only. Test evaluation is disabled for every job, and
no result may select a revised architecture, AutoCon lambda, or test release.
It is intended to document transfer failures as well as successes before any
cross-dataset confirmation is proposed.
