# External Baseline Reproduction Protocol

## Purpose

The audited release uses compact local implementations for NLinear, PatchTST,
iTransformer, TimesNet, TimeMixer, and RLinear. They are appropriate for the
within-repository controlled comparisons, but a Q1 submission should add an
external-code confirmation before claiming superiority over published methods.

## Required External Reproductions

Use the official repositories and record immutable commit hashes, environment
files, dataset hashes, full commands, validation selections, checkpoints, and
test logs. Do not select a source pool, loss weight, horizon, or seed after an
external test metric is observed.

| Baseline | Official repository | Required coordinates |
| --- | --- | --- |
| PatchTST | `yuqinie98/PatchTST` | PEMS-BAY H192/H336; METR-LA H192; Weather H192 |
| iTransformer | `thuml/iTransformer` | same four coordinates |
| TimesNet | `thuml/TimesNet` | same four coordinates |
| TimeMixer | `kwuking/TimeMixer` | PEMS-BAY H192/H336; optional METR-LA/Weather extension |
| DLinear/RLinear | `cure-lab/LTSF-Linear` | same four coordinates |

## Fairness Contract

1. Use the same chronological `60/20/20` split, `seq_len=96`, and horizon.
2. Restrict target supervision to the declared 10% or 25% prefix before model
   selection; preserve the remaining target training rows as unavailable.
3. Use the same three test seeds per coordinate and report every seed.
4. Select epoch/checkpoint from validation only. Test once after selection.
5. Match the target normalization contract or report the discrepancy.
6. Measure prediction-only GPU forward latency separately from training time.
7. Retain all negative results and compare only against the frozen audited
   ACF-InfoNCE release, never against post-test retuned candidate checkpoints.

## PatchTST Compatibility Audit

The official PatchTST repository was inspected at main commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` (Apache-2.0). Its supervised
`Dataset_Custom` uses a fixed 70/10/20 split and its `run_longExp.py` invokes
test immediately after every training run. Neither behavior is compatible with
the audited 60/20/20 chronological split, few-shot target prefix, and
validation-selected single test read.

An official-code confirmation must therefore use a separately versioned
adapter that changes only data boundaries and the train/test invocation order:

1. adapter accepts explicit `train_end`, `val_end`, and `few_train_end`;
2. scaler fits only the declared available target prefix;
3. training returns a validation-selected checkpoint without testing;
4. a separate read-only test command loads that checkpoint exactly once.

The adapter itself needs unit tests against the local chronological window
contract before any official PatchTST result can be included. Running the
unmodified official script would be a descriptive different-split experiment,
not a confirmatory comparison.

## Acceptance Rule

External reproductions are confirmatory only if all data hashes, split
boundaries, seeds, and selection metadata are present. Otherwise report them
as a descriptive published-code comparison and do not attach paired
significance to the local candidate.

## Current Status

The adapter has been executed for PEMS-BAY H192 and H336. The H192 external
PatchTST validation result is `2.764765` versus active `2.744908` (`-0.718%`,
3/3); its post-selection test extension is `2.652867` versus active `2.621235`
(`-1.192%`, 3/3, one-SE upper ratio `.9903`). H336 test is `2.314149` versus
active `2.275405` (`-1.674%`, 3/3, upper ratio `.9892`). Strict audits are
`artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_h192_extension_20261003/audit.json`
and `artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_h336_extension_20261003/audit.json`.

The same PatchTST adapter also confirms METR-LA H192: active `2.617203` versus
official PatchTST `2.743069` (`-4.589%`, 3/3, upper ratio `.9559`), audited at
`artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_metr_la_h192_extension_20261003/audit.json`.

The 25%-label PEMS-BAY H192 extension additionally uses a read-only official
checkpoint evaluator which validates the dataset hash plus SHA-256 locks for
the official `PatchTST.py` and backbone source files. Its audit is recorded in
`artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_25_extension_20261003/audit.json`.

The same adapter pattern has been executed for official iTransformer commit
`c2426e68ca13f74aaec08045c5c724d8ad328124`: PEMS-BAY H192/H336 test results
are `-7.054%/-11.360%`, both 3/3. Audits are
`artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h192_extension_20261003/audit.json`
and `artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h336_extension_20261003/audit.json`.

The same iTransformer adapter confirms METR-LA H192: active `2.617203` versus
official iTransformer `2.869208` (`-8.783%`, 3/3, upper ratio `.9158`), audited
at `artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_metr_la_h192_extension_20261003/audit.json`.

The PEMS-BAY H192/25% iTransformer extension uses a read-only evaluator that
binds the dataset plus four official source files (model, embedding, attention,
and encoder) by SHA-256. Its audit is at
`artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_pems_25_extension_20261003/audit.json`.

All official-code extensions remain post-release, so local audited releases
remain the primary evidence.

Official TimesNet is now also adapted from `thuml/Time-Series-Library` commit
`4e938a1767106324dd753b2a44832bf870a0252e`. PEMS-BAY H192/H336 test results
are `-7.523%/-11.493%`, both 3/3, with upper ratios `.9277/.8889`. Audits are
`artifacts/paper_final/mambatsv3_acf_infonce_official_timesnet_h192_extension_20261003/audit.json`
and `artifacts/paper_final/mambatsv3_acf_infonce_official_timesnet_h336_extension_20261003/audit.json`.

Official TimeMixer is adapted from `kwuking/TimeMixer` commit
`e24610583b36fdd8c76cc17a8df4e65759a5f460`. PEMS-BAY H192/H336 test results
are `-3.564%/-5.831%`, both 3/3, with upper ratios `.9720/.9452`. Audits are
`artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h192_extension_20261003/audit.json`
and `artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h336_extension_20261003/audit.json`.

RLinear has no upstream neural-network repository adaptation in this package.
All external code results are post-release extensions; local audited releases
remain the primary evidence.
