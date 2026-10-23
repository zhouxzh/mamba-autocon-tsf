# MambaTS Long-Context ACF Screen

**日期：** 2026-09-10  
**状态：** validation-only exploratory；测试集保持关闭。  
**目的：** 检查扩大输入窗口后，train-only ACF memory 是否能合法使用 PEMS 的日周期。

## 动机

PEMS03/04/07/08 的训练段在 lag=288（5 分钟采样下约 24 小时）处平均绝对 ACF 约为 `0.84--0.87`。原 v4 的 `seq_len=96` 无法访问这个 lag，因此本 screen 固定 H=96，把历史窗口扩到 384，并使用 `[24,168,288]` 作为 v4 的静态 train-only ACF lags。

## 合同与工件

- 计划：[`mamba_pems_ts_long_context_acf_screen_20260910.json`](mamba_pems_ts_long_context_acf_screen_20260910.json)
- 工件：[`mamba_pems_ts_long_context_acf_screen_20260910`](../../artifacts/exploratory/mamba_pems_ts_long_context_acf_screen_20260910/)
- 读出：[`readout.md`](../../artifacts/exploratory/mamba_pems_ts_long_context_acf_screen_20260910/readout/readout.md)
- 配对 CSV：[`v4_vs_v3_deltas.csv`](../../artifacts/exploratory/mamba_pems_ts_long_context_acf_screen_20260910/readout/v4_vs_v3_deltas.csv)

矩阵是 PEMS03/04/07/08、H=96、seed=2022、`seq_len=384`、四 epochs，共 8 个作业。v3/v4 使用同一数据划分、AutoCon `lambda=0.001` 和 batch size；只有 v4 开启静态 ACF memory。8/8 作业成功，phase validation 通过，测试指标未生成。

## 结果

| 数据集 | v3 MSE | v4 MSE | 相对变化 | v3 秒 | v4 秒 |
|---|---:|---:|---:|---:|---:|
| PEMS03 | 0.080343 | 0.081336 | `+1.236%` | 55.18 | 56.45 |
| PEMS04 | 0.101762 | 0.101276 | `-0.477%` | 33.28 | 34.62 |
| PEMS07 | 0.105484 | 0.106054 | `+0.541%` | 116.70 | 120.01 |
| PEMS08 | 0.231878 | 0.235697 | `+1.647%` | 29.09 | 30.78 |

v4 仅在 `1/4` 个精确坐标改善，平均相对变化为 `+0.737%`。平均 wall-time 为 v3 `58.56s`、v4 `60.46s`，约增加 `3.2%`。这不是正式效率结论，但不支持“ACF memory 同时更准更快”。

## 结论与停止规则

- 扩大窗口确实解决了 lag=288 的**索引可用性**问题，但没有在当前预算和配置下带来 validation 精度收益。
- 长上下文 v4 不应进入当前论文主线，也不能与 `seq_len=96` 的结果混合排名。
- 该方向按“无稳定精度收益且 wall-time 上升”停止扩展；后续若要重新启动，必须先注册更长预算、更多 seed 和同窗口 TimesNet/PatchTST 强基线。
- 当前论文应把 PEMS H=96 的短板解释为：仅有 ACF 先验和更长输入并不足以解决长期预测，AutoCon 表示目标也没有在该候选上显示独立增益。

