# Strict Evidence Status

> **修正状态（2026-08-26）：** 本页对应的 `strict` 产物是在 AutoCon 局部 logits 稳定化轴修正前生成的冻结 pilot。它们仍可复核，但不应与 [`corrected_training_plan.json`](../research/corrected_training_plan.json) 的新结果混表；论文主结论应等待 corrected 阶段重新生成。

本页是当前严格实验的入口，不替代机器可读结果。每个阶段都由 immutable source snapshot、数据/计划 SHA256、job manifest 和 `phase_validation.json` 绑定；主数值以对应 `artifacts/paper_final/strict/<phase>/report/` 为准。

## Completion

| phase | role | features | jobs | validator |
|:--|:--|:--|--:|:--|
| `selection` | validation-only lambda selection | S | 630 | valid |
| `selection_M` | validation-only lambda selection | M | 630 | valid |
| `confirmatory_core` | ETT confirmatory matrix | S | 450 | valid |
| `confirmatory_M` | ETT confirmatory matrix | M | 450 | valid |
| `secondary_transfer` | Weather and Exchange-Rate transfer | S | 180 | valid |
| `exploratory_ili` | National Illness exploration | S | 180 | valid |

All 2,520 registered jobs completed successfully. Each confirmatory, secondary, and exploratory cell has five seeds; seed is the inferential replication unit.

## Frozen Lambdas

| feature setting | AutoConNetAutoCon | Mamba1AutoCon | Mamba2AutoCon |
|:--|--:|--:|--:|
| S | 0.01 | 0.3 | 0.3 |
| M | 1.0 | 1.0 | 1.0 |

Selection uses training/validation data only and does not call the test loader.

## Current Reports

- [`confirmatory-core-fresh.md`](confirmatory-core-fresh.md): ETT-S primary controlled ablation.
- [`confirmatory-M-fresh.md`](confirmatory-M-fresh.md): ETT-M primary controlled ablation.
- [`secondary-transfer-fresh.md`](secondary-transfer-fresh.md): Weather and Exchange-Rate secondary transfer.
- [`exploratory-ili-fresh.md`](exploratory-ili-fresh.md): National Illness exploratory results.

## Interpretation Boundary

- On ETT-S, Mamba2AutoCon improves the paired MSE in 9/10 dataset-horizon cells (mean relative change -2.14%); Mamba1AutoCon improves 6/10 (mean -0.42%). These are directional seed-paired ablation results, not multiple-comparison-adjusted significance claims.
- On ETT-M, Mamba2AutoCon improves 6/10 paired cells (mean -0.78%) and Mamba1AutoCon improves 5/10 (mean +0.32%). The M version uses a cross-channel averaged ACF prior, not a channel-independent model.
- On Weather/Exchange, Mamba2AutoCon improves all 4 paired cells (mean -2.39%), but DLinear or PatchTST is the absolute MSE winner in every transfer cell. Mamba1AutoCon degrades both Weather cells substantially.
- On weekly National Illness, paired changes are below 0.1% in magnitude and Mamba2AutoCon worsens all four cells. This phase remains exploratory because horizons are steps rather than calibrated days/weeks.
- Across the ETT matrices, PatchTST or TimesNet wins most absolute cells. The evidence supports a bounded AutoCon-on-Mamba ablation analysis, not a universal performance or state-of-the-art claim.

Metrics in the primary tables use the repository's train-standardized space. Original-scale and per-channel metrics are descriptive. Sliding windows and channels are not independent replicates; non-overlapping forecast-origin summaries are sensitivity analyses only.
