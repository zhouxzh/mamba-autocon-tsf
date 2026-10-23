# MambaTS v3 Weather Test Release Launch Review

Scope: the 75-job held-out Weather matrix registered in
`mambatsv3_weather_test_release_20260922.json`.

The active model is fixed before test release: MambaTS v3 with the existing
AutoCon representation objective and `AutoCon_lambda=0.001`. The exact
`lambda=0` control remains validation-only mechanism evidence rather than a
test-selection arm. PatchTST, TimesNet, iTransformer, and Mamba-2 remain fixed
baseline arms for every horizon and seed. The matrix uses five fresh seeds
(2031--2035), H96/H336/H720, and a 20-epoch maximum budget with validation-only
early stopping.

Prelaunch checks:

- The Weather CSV SHA256 matches the immutable plan fingerprint.
- The plan expands to 75 jobs: 15 active and 60 baseline jobs.
- All arms use the same split, normalization, sequence length, batch size,
  training budget, and test streaming aggregation. Test data restores no
  checkpoint and cannot alter the matrix.
- Test metrics are read only after all phase validators succeed. The test
  release cannot choose a revised `lambda`, architecture, horizon, or baseline.
- A post-training resource report will measure every listed nonlinear
  comparator on matched shapes and will distinguish prediction-only latency
  from full training-wall-time observations.

The user continuation request authorizes this bounded confirmation launch. It
does not authorize a cross-dataset or universal publication claim.
