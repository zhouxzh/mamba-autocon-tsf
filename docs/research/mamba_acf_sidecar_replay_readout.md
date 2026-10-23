# MambaACF Variable Memory + Relation Persistence Readout

> Exploratory paired screen only. Each dataset has n=3 seeds; no result here is confirmatory evidence or a basis for publication claims.

## Contract and validation

- Plan: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/docs/research/mamba_acf_sidecar_replay_plan.json`; SHA-256 `e7fb6d24943e154ae6a1533eb638344c46d80919a53e2b5df691fc9267096a76`.
- Screen root: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/mamba_acf_sidecar_replay_20260831`; successful jobs: `18/18`.
- Pairing unit: same dataset, feature setting, horizon, and seed. Delta is treatment minus reference; negative is favorable.
- Metric source: `stream_metrics.json` scaled MSE/MAE. Forecast windows are not treated as independent observations.
- Artifact contract: the provenance-snapshot exploratory runs persisted a complete `acf_prior.npz` and a per-job SHA-256 binding; this remains non-confirmatory.
- Evidence boundary: each memory/relation job fitted its train-only prior during training and wrote the sidecar after that fit. The reported test metrics came from that same training run; no metric here came from a later standalone evaluation that loaded the sidecar. This is a persistence/contract replay, not an independent sidecar-loaded reproduction.
- A separate 18-job `is_training=0` standalone sidecar replay has now been completed in `artifacts/exploratory/mamba_acf_sidecar_standalone_replay_20260831_audit/`; its manifest freezes inputs/tolerances and its JSONL records each job. It validates the current evaluation loader against these historical checkpoints but does not turn them into current-tree training or confirmatory evidence.

| Phase | Jobs | Manifest SHA-256 | Provenance ID | Source-tree SHA-256 |
|---|---:|---|---|---|
| `variable_plain_h96_screen` | 6/6 | `f9fcb6ce289842d9f54538b2c155a4fd8b64d298a70e6358dc094853a9bd3d33` | `ca72fd519dfdaed6` | `65be5472042697fa189e8ff35017ee16da18999666ebe5b51828de5afc66b3c6` |
| `variable_memory_h96_screen` | 6/6 | `413990e2c2b79e4dbbe24317eeaad70a21a5630cd7f5497b8910357afc9cf0c7` | `fc1d10548e78cd1d` | `65be5472042697fa189e8ff35017ee16da18999666ebe5b51828de5afc66b3c6` |
| `variable_memory_relation_h96_screen` | 6/6 | `a024ab359335b54a84328eb5f988b7da883c8e7cbafc3fa09c8bd4fef26bf905` | `ac4ea8b221a9560a` | `65be5472042697fa189e8ff35017ee16da18999666ebe5b51828de5afc66b3c6` |

- Source status: the metrics are bound to the immutable provenance snapshot; the live working tree differs in 5 captured source file(s): `exp/exp_long_term_forecasting_with_AutoCon.py, exp/exp_long_term_forecasting_with_mamba.py, models/MambaACF.py, run.py, scripts/validate_training_phase.py`. This readout must therefore be treated as a provenance-snapshot training result, not a current-tree training result.

## Per-seed paired metrics

| Dataset | Seed | Plain MSE | Memory MSE | Memory-Plain | Memory+Relation MSE | Relation-Memory | Plain MAE | Memory MAE | Memory-Plain | Memory+Relation MAE | Relation-Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.387909 | 0.384475 | -0.003434 | 0.382797 | -0.001678 | 0.413996 | 0.410445 | -0.003551 | 0.409013 | -0.001432 |
| ETTh1 | 2022 | 0.384194 | 0.382920 | -0.001274 | 0.380722 | -0.002198 | 0.407511 | 0.405469 | -0.002042 | 0.403975 | -0.001495 |
| ETTh1 | 2023 | 0.385979 | 0.384122 | -0.001857 | 0.381908 | -0.002214 | 0.410734 | 0.407391 | -0.003343 | 0.402128 | -0.005263 |
| traffic | 2021 | 0.554268 | 0.545935 | -0.008333 | 0.543253 | -0.002682 | 0.361929 | 0.361833 | -0.000096 | 0.357185 | -0.004648 |
| traffic | 2022 | 0.555835 | 0.547506 | -0.008329 | 0.544640 | -0.002865 | 0.361712 | 0.360684 | -0.001028 | 0.354655 | -0.006029 |
| traffic | 2023 | 0.558588 | 0.548244 | -0.010343 | 0.544761 | -0.003483 | 0.360852 | 0.358572 | -0.002280 | 0.356414 | -0.002159 |

## Paired summaries

SD is the sample standard deviation across seed-level deltas. Paired t values and p-values are descriptive only (n=3 per dataset).

| Dataset | Contrast | Metric | n | Reference mean | Treatment mean | Delta mean +/- SD | Wins | Relative change | t | two-sided p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | variable memory minus variable plain | MSE | 3 | 0.386027 | 0.383839 | -0.002188 +/- 0.001117 | 3/3 | -0.567% | -3.3924 | 0.0770 |
| ETTh1 | variable memory minus variable plain | MAE | 3 | 0.410747 | 0.407768 | -0.002979 +/- 0.000818 | 3/3 | -0.725% | -6.3053 | 0.0242 |
| ETTh1 | variable memory + relation minus variable memory | MSE | 3 | 0.383839 | 0.381809 | -0.002030 +/- 0.000305 | 3/3 | -0.529% | -11.5259 | 0.0074 |
| ETTh1 | variable memory + relation minus variable memory | MAE | 3 | 0.407768 | 0.405038 | -0.002730 +/- 0.002194 | 3/3 | -0.669% | -2.1552 | 0.1639 |
| traffic | variable memory minus variable plain | MSE | 3 | 0.556230 | 0.547228 | -0.009002 +/- 0.001162 | 3/3 | -1.618% | -13.4218 | 0.0055 |
| traffic | variable memory minus variable plain | MAE | 3 | 0.361498 | 0.360363 | -0.001134 +/- 0.001096 | 3/3 | -0.314% | -1.7931 | 0.2148 |
| traffic | variable memory + relation minus variable memory | MSE | 3 | 0.547228 | 0.544218 | -0.003010 +/- 0.000420 | 3/3 | -0.550% | -12.4252 | 0.0064 |
| traffic | variable memory + relation minus variable memory | MAE | 3 | 0.360363 | 0.356085 | -0.004279 +/- 0.001961 | 3/3 | -1.187% | -3.7784 | 0.0635 |
| pooled descriptive (not independent) | variable memory minus variable plain | MSE | 6 | 0.471129 | 0.465534 | -0.005595 +/- 0.003869 | 6/6 | -1.188% | -3.5425 | 0.0165 |
| pooled descriptive (not independent) | variable memory minus variable plain | MAE | 6 | 0.386122 | 0.384066 | -0.002057 +/- 0.001330 | 6/6 | -0.533% | -3.7881 | 0.0128 |
| pooled descriptive (not independent) | variable memory + relation minus variable memory | MSE | 6 | 0.465534 | 0.463014 | -0.002520 +/- 0.000629 | 6/6 | -0.541% | -9.8099 | 0.0002 |
| pooled descriptive (not independent) | variable memory + relation minus variable memory | MAE | 6 | 0.384066 | 0.380562 | -0.003504 +/- 0.002045 | 6/6 | -0.912% | -4.1966 | 0.0085 |

## Relation objective coverage

The relation arm was required to use objective v3 and strictly positive global coverage series.

| Dataset | Seed | Objective | Valid min | Candidate min | Candidate-anchor min | Positive-count min | Effective lags |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 2021 | 3 | 0.029527 | 0.163162 | 0.993986 | 0.915330 | 96, 120 |
| ETTh1 | 2022 | 3 | 0.029603 | 0.163474 | 0.994340 | 0.917689 | 96, 120 |
| ETTh1 | 2023 | 3 | 0.029595 | 0.162964 | 0.994222 | 0.917453 | 96, 120 |
| traffic | 2021 | 3 | 0.029755 | 0.140081 | 0.995631 | 0.922407 | 96, 120 |
| traffic | 2022 | 3 | 0.029643 | 0.139676 | 0.996420 | 0.918924 | 96, 120 |
| traffic | 2023 | 3 | 0.029743 | 0.139544 | 0.995937 | 0.922020 | 96, 120 |

## Interpretation boundary

- Three seeds per dataset provide a mechanism-screening readout only. The exact paired Wilcoxon minimum p-value is not used to inflate evidence; only descriptive paired t statistics are shown here.
- The pooled row is a cross-dataset descriptive summary, not six independent replications of one data-generating process.
- The variable branch, train-only ACF memory, and relation objective are jointly confounded in the second contrast; this screen cannot identify a causal contribution of each component.
- A separate, frozen confirmation matrix with validation-only selection and an independent evaluation protocol is required before any generalization or publication claim.
