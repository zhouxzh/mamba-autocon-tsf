# ACF 与因果季节基线效用审计

**用途：** 人工复核 signed ACF 先验在 Mamba 预测中的可用性。本文档对应独立的探索性脚本和输出，不属于任何 `artifacts/paper_final/` 确认性阶段。

## 结论边界

这项审计只回答两个描述性问题：

1. 在 Electricity（321 通道）和 Traffic（862 通道）的训练段，候选滞后是否有稳定的带符号自相关；
2. 对 `seq_len=96` 的直接因果滞后记忆，在不同预测长度内有多少步确实能读取历史观测，以及简单 last/seasonal baseline 的误差是多少。

它不训练模型、不选择 Mamba 超参数、不使用冻结确认性工件，也不把 ACF 峰值当作预测性能保证。测试段真实值仅用于计算 baseline 误差；ACF、标准化器均只在训练段拟合，候选 lag 在命令行中预先给定。由于高维长 horizon 的全滑窗展开会产生很大的计算量，baseline 使用显式的 `origin_stride` 稀疏 origins；该参数写入元数据，设为 `1` 才是全重叠滑窗。

## 固定定义

候选滞后为 `1, 24, 48, 96, 168, 336, 720`，预测长度为 `96, 336, 720`，输入长度为 `96`。这些默认值可通过 CLI 显式改变，实际值写入 `metadata.json`。

对每个通道，训练段 ACF 使用均值中心化后的 signed 估计：

\[
  \operatorname{ACF}(k)=
  \frac{\sum_{t=k}^{N-1}(x_t-\bar{x})(x_{t-k}-\bar{x})}
       {\sum_{t=0}^{N-1}(x_t-\bar{x})^2}.
\]

这保留正/负号，不取绝对值；汇总同时报告通道均值、中位数、分位数、绝对值强度以及正负比例。`signed_acf_by_channel.csv` 保存每个通道的值，便于复核异质性。

数据划分沿用仓库 `Dataset_Custom` 的边界：前 60% 为 train，后 20% 为 test，中间 20% 为 validation。测试窗口包含 test 起点前 `seq_len` 个历史上下文，因此 test 的窗口数量为 `test_context_rows - seq_len - horizon + 1`。

对 forecast step `h=1,...,H`，候选 lag 的直接历史索引定义为：

\[
  i(h,k)=L-1+h-k,
\]

其中 `L=96`。只有 `0 <= i(h,k) < L` 时才算真实历史可用。`lag_availability.csv` 报告可用步数、比例、首末可用步。不可用位置绝不循环复制旧周期。

## Baseline 定义

- `last / all_steps`：把 origin 最后一个观测值复制到整个 horizon。
- `seasonal_lag / available_only`：只在直接 lag 索引可用的 forecast steps 上计算误差；不可用步不进入分母。
- `seasonal_lag / last_fallback`：可用步使用直接 lag 值，不可用步退回 last 值；覆盖完整 horizon，适合和 last 作描述性比较。

每一行同时给出 train-standardized 和原始量纲的 MAE/MSE/RMSE。标准化器的均值/标准差只由 train rows 计算；原始量纲结果可能被高方差通道主导，因此主阅读顺序应先看 scaled MSE，再查看原始量纲诊断。train baseline 只描述拟合段内的滚动窗口误差，不是独立泛化估计；test baseline 也不是模型确认性结果。

## 运行

在仓库根目录执行：

```bash
conda run --no-capture-output -n mamba python scripts/analyze_acf_forecast_utility.py \
  --root . \
  --output_dir artifacts/exploratory/acf_forecast_utility \
  --seq_len 96 \
  --lags 1 24 48 96 168 336 720 \
  --horizons 96 336 720 \
  --train_ratio 0.6 \
  --origin_stride 24 \
  --force
```

脚本拒绝写入 `artifacts/paper_final/`。输入 CSV 只读打开；输出目录包含：

- `metadata.json`：数据边界、源文件 SHA256、脚本 SHA256、公式和运行参数；
- `signed_acf_summary.csv`：每个 dataset/lag 的通道汇总；
- `signed_acf_by_channel.csv`：逐通道 signed ACF；
- `lag_availability.csv`：每个 dataset/lag/horizon 的因果可用比例；
- `descriptive_baseline_metrics.csv`：train/test 的三类 baseline 误差。
- `origin_stride`：描述性 baseline 使用的 origin 间隔；每行的 `n_windows` 是实际评估的稀疏窗口数，不是该 split 的全部可用窗口数。

## 人工复核要点

1. 检查 `metadata.json` 的 `train_end_exclusive`、`test_context_start_in_full_series` 是否与仓库 loader 一致，并核对源 CSV SHA256。
2. 先查看 `lag_availability.csv`：当 `lag > seq_len` 或 horizon 较短时，可用比例可能很低，不能把 seasonal baseline 的 unavailable 部分误称为观测。
3. 查看 `signed_acf_summary.csv` 的 `positive_fraction` 与 `negative_fraction`。高绝对 ACF 但符号混杂时，单一无符号强度不足以支持统一的 seasonal residual 方向。
4. 对 baseline 比较优先使用相同 dataset、horizon、split 下的 `last_fallback` scaled MSE；`available_only` 的分母不同，只作可用性诊断。
5. 所有结果都是探索性描述。即使 `origin_stride > 1`，窗口之间仍可能重叠，不能把 `n_windows` 当作独立样本量，也不能由 baseline 或 ACF 直接推断 Mamba 的显著性。

## 本次运行记录

命令已完成，参数为 `origin_stride=24`，输出目录为 `artifacts/exploratory/acf_forecast_utility/`。源文件 SHA256 和边界记录在 `metadata.json`。以下是便于人工复核的关键 scaled 数值（完整数值以 CSV 为准）：

| dataset | lag | median signed ACF | positive fraction |
|---|---:|---:|---:|
| Electricity | 96 | 0.893937 | 0.9969 |
| Electricity | 168 | 0.909416 | 1.0000 |
| Traffic | 168 | 0.816431 | 1.0000 |
| Traffic | 336 | 0.782971 | 1.0000 |

| dataset | horizon | last scaled MSE | lag-96 fallback | lag-168 fallback | lag-336 fallback |
|---|---:|---:|---:|---:|---:|
| Electricity | 96 | 1.771323 | 0.376837 | 1.368347 | 1.771323 |
| Electricity | 336 | 1.759001 | 1.369703 | 1.317431 | 1.323262 |
| Electricity | 720 | 1.743988 | 1.568553 | 1.544191 | 1.545500 |
| Traffic | 96 | 1.796423 | 1.488943 | 1.504795 | 1.796423 |
| Traffic | 336 | 1.841975 | 1.747876 | 1.497681 | 1.521043 |
| Traffic | 720 | 1.822717 | 1.785573 | 1.665472 | 1.668347 |

这些 baseline 仍是描述性诊断，不是模型对比的确认性结果；`origin_stride=24` 的稀疏 origin 也不应被当成独立样本。若重新运行参数或数据发生变化，应使用新的输出目录并重新记录 SHA256。
