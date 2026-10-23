# MambaACF Variable Memory + Relation Persistence Readout

> Exploratory paired screen only. Each dataset has n=3 seeds; no result here is confirmatory evidence or a basis for publication claims.

## Contract and validation

- Plan: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/docs/research/mamba_acf_live_sidecar_screen_plan.json`; SHA-256 `d311ed122b44c59a36936830763019ede2fe7fd96a4627a554437458439c8040`.
- Screen root: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/mamba_acf_live_sidecar_screen_20260901`; successful jobs: `18/18`.
- Pairing unit: same dataset, feature setting, horizon, and seed. Delta is treatment minus reference; negative is favorable.
- Metric source: `stream_metrics.json` scaled MSE/MAE. Forecast windows are not treated as independent observations.
- Artifact contract: the provenance-snapshot exploratory runs persisted a complete `acf_prior.npz` and a per-job SHA-256 binding; this remains non-confirmatory.
- Evidence boundary: each memory/relation job fitted its train-only prior during training and wrote the sidecar after that fit. The reported test metrics came from that same training run; no metric here came from a later standalone evaluation that loaded the sidecar. This is a persistence/contract screen, not an independent confirmatory reproduction.
- A separate 18-job `is_training=0` standalone sidecar replay has now been completed in `artifacts/exploratory/mamba_acf_sidecar_standalone_replay_20260831_audit/`; its manifest freezes inputs/tolerances and its JSONL records each job. It validates the current evaluation loader against these historical checkpoints but does not turn them into current-tree training or confirmatory evidence.

| Phase | Jobs | Manifest SHA-256 | Provenance ID | Source-tree SHA-256 |
|---|---:|---|---|---|
| `variable_plain_h96_screen` | 6/6 | `a5dca8c1f7ae4b1dc0ce73249f0307c2f31872c61229d73030be9ba150318c63` | `616195a89e6e97d9` | `b9656dad77de2d7234dfa93d00df83334588cb84d8867a757ec18ad40a32e353` |
| `variable_memory_h96_screen` | 6/6 | `c6c438e5bfc5f3b609a460d8affb5aa6bd9bcdffaf7b4d419c4c73d292a2b996` | `01f247542889fff1` | `b9656dad77de2d7234dfa93d00df83334588cb84d8867a757ec18ad40a32e353` |
| `variable_memory_relation_h96_screen` | 6/6 | `2bfd80e47a35b8eb1f3e6a52198aaada08ffd9209be6f798fb57bb8216d06e96` | `82d1c81a9d96ed83` | `b9656dad77de2d7234dfa93d00df83334588cb84d8867a757ec18ad40a32e353` |

- Source status: the metrics are bound to the immutable provenance snapshot; the live working tree differs in 4 captured source file(s): `exp/exp_long_term_forecasting_with_AutoCon.py, exp/exp_long_term_forecasting_with_mamba.py, run.py, scripts/validate_training_phase.py`. This readout must therefore be treated as a provenance-snapshot training result, not a current-tree training result.

## Per-seed paired metrics

| Dataset | Seed | Plain MSE | Memory MSE | Memory-Plain | Memory+Relation MSE | Relation-Memory | Plain MAE | Memory MAE | Memory-Plain | Memory+Relation MAE | Relation-Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.387909 | 0.386251 | -0.001657 | 0.383971 | -0.002280 | 0.413996 | 0.411236 | -0.002760 | 0.410254 | -0.000982 |
| ETTh1 | 2022 | 0.384194 | 0.384213 | 0.000019 | 0.381790 | -0.002423 | 0.407511 | 0.405672 | -0.001839 | 0.404356 | -0.001316 |
| ETTh1 | 2023 | 0.385979 | 0.387093 | 0.001114 | 0.382507 | -0.004586 | 0.410734 | 0.409091 | -0.001642 | 0.402113 | -0.006979 |
| traffic | 2021 | 0.554269 | 0.542046 | -0.012223 | 0.538695 | -0.003351 | 0.361930 | 0.358758 | -0.003172 | 0.353618 | -0.005140 |
| traffic | 2022 | 0.555834 | 0.544566 | -0.011268 | 0.541552 | -0.003014 | 0.361711 | 0.358295 | -0.003416 | 0.352433 | -0.005862 |
| traffic | 2023 | 0.558584 | 0.544001 | -0.014584 | 0.539679 | -0.004321 | 0.360848 | 0.357008 | -0.003840 | 0.353959 | -0.003048 |

## Paired summaries

SD is the sample standard deviation across seed-level deltas. Paired t values and p-values are descriptive only (n=3 per dataset).

| Dataset | Contrast | Metric | n | Reference mean | Treatment mean | Delta mean +/- SD | Wins | Relative change | t | two-sided p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | variable memory minus variable plain | MSE | 3 | 0.386027 | 0.385852 | -0.000175 +/- 0.001396 | 1/3 | -0.045% | -0.2167 | 0.8485 |
| ETTh1 | variable memory minus variable plain | MAE | 3 | 0.410747 | 0.408666 | -0.002081 +/- 0.000597 | 3/3 | -0.507% | -6.0383 | 0.0263 |
| ETTh1 | variable memory + relation minus variable memory | MSE | 3 | 0.385852 | 0.382756 | -0.003096 +/- 0.001292 | 3/3 | -0.802% | -4.1517 | 0.0534 |
| ETTh1 | variable memory + relation minus variable memory | MAE | 3 | 0.408666 | 0.405574 | -0.003092 +/- 0.003370 | 3/3 | -0.757% | -1.5891 | 0.2530 |
| traffic | variable memory minus variable plain | MSE | 3 | 0.556229 | 0.543537 | -0.012692 +/- 0.001707 | 3/3 | -2.282% | -12.8797 | 0.0060 |
| traffic | variable memory minus variable plain | MAE | 3 | 0.361496 | 0.358020 | -0.003476 +/- 0.000338 | 3/3 | -0.962% | -17.8051 | 0.0031 |
| traffic | variable memory + relation minus variable memory | MSE | 3 | 0.543537 | 0.539975 | -0.003562 +/- 0.000679 | 3/3 | -0.655% | -9.0866 | 0.0119 |
| traffic | variable memory + relation minus variable memory | MAE | 3 | 0.358020 | 0.353337 | -0.004683 +/- 0.001461 | 3/3 | -1.308% | -5.5519 | 0.0309 |
| pooled descriptive (not independent) | variable memory minus variable plain | MSE | 6 | 0.471128 | 0.464695 | -0.006433 +/- 0.006996 | 4/6 | -1.365% | -2.2524 | 0.0741 |
| pooled descriptive (not independent) | variable memory minus variable plain | MAE | 6 | 0.386122 | 0.383343 | -0.002778 +/- 0.000879 | 6/6 | -0.720% | -7.7438 | 0.0006 |
| pooled descriptive (not independent) | variable memory + relation minus variable memory | MSE | 6 | 0.464695 | 0.461366 | -0.003329 +/- 0.000958 | 6/6 | -0.716% | -8.5163 | 0.0004 |
| pooled descriptive (not independent) | variable memory + relation minus variable memory | MAE | 6 | 0.383343 | 0.379456 | -0.003888 +/- 0.002481 | 6/6 | -1.014% | -3.8378 | 0.0122 |

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
