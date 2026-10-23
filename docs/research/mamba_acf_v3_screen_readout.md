# MambaACF v3 Relation Screen: Statistical Readout

> Exploratory screen only. The report is generated from the five completed phases under `artifacts/paper_final/mamba_acf_v3_relation_screen`; it does not perform model or hyper-parameter selection.

## Scope and interpretation

- Input root: `/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/mamba_acf_v3_relation_screen`
- Rows: `30` successful runs (5 phases x 2 datasets x 3 seeds).
- Metric source: `stream_metrics.json` scaled metrics (`scaled.mse`, `scaled.mae`); lower is better.
- Pairing key: same dataset, feature setting, horizon, and seed. Delta is treatment minus reference; negative means lower error.
- Inference backend: SciPy available. Tests are two-sided paired one-sample t-tests and Wilcoxon signed-rank tests on seed-level deltas.
- The inferential unit is the seed/run, not an individual forecast window.
- Validated plan SHA-256: `61d3ee6cf62add4dc3512b2f65d86be9ecbce5bdee06c6b991119ad4b8bff306`

### Replayability boundary

These artifacts were produced by source snapshots that predate the `acf_prior.npz` sidecar contract. No `acf_prior.npz` is present under this screen root, and the per-run configs do not contain `mamba_acf_prior_artifact` or `mamba_acf_prior_sha256`. The `available_memory_screen`, relation-enabled, and memory+relation arms therefore cannot be replayed from checkpoint plus config alone by the current runner: the historical code fitted the train-only prior in process, whereas the current runner requires the persisted sidecar and refuses to refit it during checkpoint evaluation. Recomputing those arms requires the original train split and matching immutable source snapshot and should be labeled a historical re-run rather than a sidecar-backed standalone replay. The `plain_screen` arm has both memory and relation disabled and is not dependent on this missing prior sidecar.

### n=3 limitation

Each dataset-level contrast has only three seeds. The pooled row has six runs but combines two datasets and is not six independent replications of one data-generating process. Means, standard deviations, and p-values are descriptive screening evidence; they do not establish a stable gain, generalization, or publication-grade confirmation. No multiplicity correction is applied.

## Per-phase metrics

| Phase | Dataset | n | MSE mean +/- SD | MAE mean +/- SD |
|---|---:|---:|---:|---:|
| plain_screen | ETTh1 | 3 | 0.389664 +/- 0.002807 | 0.411648 +/- 0.002041 |
| plain_screen | traffic | 3 | 0.570445 +/- 0.003733 | 0.389375 +/- 0.002401 |
| available_memory_screen | ETTh1 | 3 | 0.389005 +/- 0.001678 | 0.408272 +/- 0.002361 |
| available_memory_screen | traffic | 3 | 0.555458 +/- 0.005670 | 0.384809 +/- 0.003573 |
| hard_relation_nonoverlap_screen | ETTh1 | 3 | 0.391957 +/- 0.002591 | 0.413486 +/- 0.003919 |
| hard_relation_nonoverlap_screen | traffic | 3 | 0.565820 +/- 0.008408 | 0.384731 +/- 0.004063 |
| soft_relation_nonoverlap_screen | ETTh1 | 3 | 0.392103 +/- 0.001567 | 0.412781 +/- 0.002968 |
| soft_relation_nonoverlap_screen | traffic | 3 | 0.568679 +/- 0.004372 | 0.388632 +/- 0.008201 |
| available_memory_soft_relation_nonoverlap_screen | ETTh1 | 3 | 0.395972 +/- 0.005431 | 0.414361 +/- 0.006776 |
| available_memory_soft_relation_nonoverlap_screen | traffic | 3 | 0.551952 +/- 0.004095 | 0.381977 +/- 0.007107 |
| plain_screen | ALL | 6 | 0.480055 +/- 0.099062 | 0.400511 +/- 0.012361 |
| available_memory_screen | ALL | 6 | 0.472231 +/- 0.091247 | 0.396540 +/- 0.013134 |
| hard_relation_nonoverlap_screen | ALL | 6 | 0.478889 +/- 0.095391 | 0.399109 +/- 0.016149 |
| soft_relation_nonoverlap_screen | ALL | 6 | 0.480391 +/- 0.096759 | 0.400707 +/- 0.014331 |
| available_memory_soft_relation_nonoverlap_screen | ALL | 6 | 0.473962 +/- 0.085542 | 0.398169 +/- 0.018793 |

## Per-seed metrics

| Phase | Dataset | Seed | Horizon | MSE | MAE |
|---|---:|---:|---:|---:|---:|
| plain_screen | ETTh1 | 2021 | 96 | 0.392341 | 0.413267 |
| plain_screen | ETTh1 | 2022 | 96 | 0.386744 | 0.409355 |
| plain_screen | ETTh1 | 2023 | 96 | 0.389907 | 0.412321 |
| plain_screen | traffic | 2021 | 96 | 0.574390 | 0.391000 |
| plain_screen | traffic | 2022 | 96 | 0.569975 | 0.390507 |
| plain_screen | traffic | 2023 | 96 | 0.566970 | 0.386617 |
| available_memory_screen | ETTh1 | 2021 | 96 | 0.390546 | 0.410168 |
| available_memory_screen | ETTh1 | 2022 | 96 | 0.387217 | 0.405627 |
| available_memory_screen | ETTh1 | 2023 | 96 | 0.389252 | 0.409020 |
| available_memory_screen | traffic | 2021 | 96 | 0.559811 | 0.386496 |
| available_memory_screen | traffic | 2022 | 96 | 0.557518 | 0.387226 |
| available_memory_screen | traffic | 2023 | 96 | 0.549046 | 0.380704 |
| hard_relation_nonoverlap_screen | ETTh1 | 2021 | 96 | 0.389882 | 0.411161 |
| hard_relation_nonoverlap_screen | ETTh1 | 2022 | 96 | 0.391129 | 0.411287 |
| hard_relation_nonoverlap_screen | ETTh1 | 2023 | 96 | 0.394861 | 0.418010 |
| hard_relation_nonoverlap_screen | traffic | 2021 | 96 | 0.575015 | 0.386837 |
| hard_relation_nonoverlap_screen | traffic | 2022 | 96 | 0.563921 | 0.387309 |
| hard_relation_nonoverlap_screen | traffic | 2023 | 96 | 0.558524 | 0.380047 |
| soft_relation_nonoverlap_screen | ETTh1 | 2021 | 96 | 0.391701 | 0.411398 |
| soft_relation_nonoverlap_screen | ETTh1 | 2022 | 96 | 0.390777 | 0.410757 |
| soft_relation_nonoverlap_screen | ETTh1 | 2023 | 96 | 0.393833 | 0.416189 |
| soft_relation_nonoverlap_screen | traffic | 2021 | 96 | 0.570454 | 0.383517 |
| soft_relation_nonoverlap_screen | traffic | 2022 | 96 | 0.571884 | 0.398091 |
| soft_relation_nonoverlap_screen | traffic | 2023 | 96 | 0.563698 | 0.384287 |
| available_memory_soft_relation_nonoverlap_screen | ETTh1 | 2021 | 96 | 0.399622 | 0.421691 |
| available_memory_soft_relation_nonoverlap_screen | ETTh1 | 2022 | 96 | 0.389730 | 0.408325 |
| available_memory_soft_relation_nonoverlap_screen | ETTh1 | 2023 | 96 | 0.398564 | 0.413067 |
| available_memory_soft_relation_nonoverlap_screen | traffic | 2021 | 96 | 0.553556 | 0.376211 |
| available_memory_soft_relation_nonoverlap_screen | traffic | 2022 | 96 | 0.555002 | 0.389917 |
| available_memory_soft_relation_nonoverlap_screen | traffic | 2023 | 96 | 0.547298 | 0.379803 |

## Paired delta by seed

| Contrast | Dataset | Seed | Delta MSE | Delta MAE |
|---|---:|---:|---:|---:|
| available memory vs plain | ETTh1 | 2021 | -0.001796 | -0.003099 |
| available memory vs plain | ETTh1 | 2022 | 0.000473 | -0.003728 |
| available memory vs plain | ETTh1 | 2023 | -0.000655 | -0.003301 |
| available memory vs plain | traffic | 2021 | -0.014580 | -0.004504 |
| available memory vs plain | traffic | 2022 | -0.012458 | -0.003282 |
| available memory vs plain | traffic | 2023 | -0.017924 | -0.005913 |
| soft relation vs hard relation | ETTh1 | 2021 | 0.001819 | 0.000237 |
| soft relation vs hard relation | ETTh1 | 2022 | -0.000352 | -0.000530 |
| soft relation vs hard relation | ETTh1 | 2023 | -0.001028 | -0.001822 |
| soft relation vs hard relation | traffic | 2021 | -0.004561 | -0.003321 |
| soft relation vs hard relation | traffic | 2022 | 0.007963 | 0.010782 |
| soft relation vs hard relation | traffic | 2023 | 0.005174 | 0.004240 |
| memory + soft relation vs available memory | ETTh1 | 2021 | 0.009076 | 0.011523 |
| memory + soft relation vs available memory | ETTh1 | 2022 | 0.002514 | 0.002698 |
| memory + soft relation vs available memory | ETTh1 | 2023 | 0.009312 | 0.004047 |
| memory + soft relation vs available memory | traffic | 2021 | -0.006255 | -0.010285 |
| memory + soft relation vs available memory | traffic | 2022 | -0.002515 | 0.002692 |
| memory + soft relation vs available memory | traffic | 2023 | -0.001748 | -0.000901 |

## Paired summaries and tests

Treatment-minus-reference deltas; `win` counts seed-level negative deltas (lower error). `t/p` is the paired one-sample t statistic/p-value; `W/p` is Wilcoxon signed-rank statistic/p-value.

| Contrast | Dataset | n | Delta MSE mean +/- SD | MSE win | t/p (MSE) | W/p (MSE) | Delta MAE mean +/- SD | MAE win | t/p (MAE) | W/p (MAE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| available memory vs plain | ALL | 6 | -0.007823 +/- 0.008071 | 5/6 | -2.3742/0.0636 | 1.0000/0.0625 | -0.003971 +/- 0.001077 | 6/6 | -9.0322/0.0003 | 0.0000/0.0312 |
| available memory vs plain | ETTh1 | 3 | -0.000659 +/- 0.001134 | 2/3 | -1.0063/0.4202 | 1.0000/0.5000 | -0.003376 +/- 0.000321 | 3/3 | -18.2210/0.0030 | 0.0000/0.2500 |
| available memory vs plain | traffic | 3 | -0.014987 +/- 0.002756 | 3/3 | -9.4190/0.0111 | 0.0000/0.2500 | -0.004566 +/- 0.001317 | 3/3 | -6.0061/0.0266 | 0.0000/0.2500 |
| soft relation vs hard relation | ALL | 6 | 0.001502 +/- 0.004515 | 3/6 | 0.8151/0.4521 | 7.0000/0.5625 | 0.001598 +/- 0.005169 | 3/6 | 0.7572/0.4831 | 9.0000/0.8438 |
| soft relation vs hard relation | ETTh1 | 3 | 0.000146 +/- 0.001488 | 2/3 | 0.1702/0.8805 | 3.0000/1.0000 | -0.000705 +/- 0.001040 | 2/3 | -1.1733/0.3615 | 1.0000/0.5000 |
| soft relation vs hard relation | traffic | 3 | 0.002859 +/- 0.006575 | 1/3 | 0.7530/0.5300 | 1.0000/0.5000 | 0.003901 +/- 0.007058 | 1/3 | 0.9573/0.4395 | 1.0000/0.5000 |
| memory + soft relation vs available memory | ALL | 6 | 0.001731 +/- 0.006418 | 3/6 | 0.6605/0.5381 | 8.0000/0.6875 | 0.001629 +/- 0.007132 | 2/6 | 0.5595/0.6000 | 6.0000/0.4375 |
| memory + soft relation vs available memory | ETTh1 | 3 | 0.006967 +/- 0.003859 | 0/3 | 3.1273/0.0888 | 0.0000/0.2500 | 0.006089 +/- 0.004754 | 0/3 | 2.2186/0.1568 | 0.0000/0.2500 |
| memory + soft relation vs available memory | traffic | 3 | -0.003506 +/- 0.002411 | 3/3 | -2.5184/0.1281 | 0.0000/0.2500 | -0.002831 +/- 0.006700 | 2/3 | -0.7319/0.5404 | 2.0000/0.7500 |

## Registered contrast notes

1. `available memory vs plain` isolates the causal ACF memory branch in MambaACF.
2. `soft relation vs hard relation` compares target semantics while keeping the global, non-overlap relation control and the short training budget fixed.
3. `memory + soft relation vs available memory` asks whether the relation objective adds beyond memory alone.

A negative mean delta is directionally favorable, but this screen alone is insufficient to claim a reliable improvement. A future confirmation should pre-register datasets, seeds, training budget, selection rule, and an independent evaluation protocol before inspecting these deltas.

## Phase validation

| Phase | Manifest jobs | Validation |
|---|---:|---:|
| `plain_screen` | 6 | true |
| `available_memory_screen` | 6 | true |
| `hard_relation_nonoverlap_screen` | 6 | true |
| `soft_relation_nonoverlap_screen` | 6 | true |
| `available_memory_soft_relation_nonoverlap_screen` | 6 | true |
