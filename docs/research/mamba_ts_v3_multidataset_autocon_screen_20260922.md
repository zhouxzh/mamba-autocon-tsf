# MambaTS v3 多数据集 AutoCon null screen（2026-09-22）

## 结论

这轮实验把 AutoCon 表示学习目标放在当前最快的 MambaTS v3 主干上，并严格保留同构 `lambda=0` null。24/24 个 H96 validation-only 作业成功，测试集没有运行。

结果不支持 AutoCon 提升预测精度：`lambda=0.001` 相对 `lambda=0` 在 ETTh1、ETTm1、Electricity、Traffic 的 paired mean relative change 分别为 `+0.009%`、`+0.006%`、`+0.008%`、`+0.004%`，每个数据集仅 `1/3` seed 胜出。AutoCon loss 非零，说明表示目标确实执行，但本轮没有独立精度收益。

当前最有价值的结果来自 v3 forecast backbone 本身：它在同一 H96/MeanStd plain baseline 坐标上，相对最佳外部 baseline 的变化为 ETTh1 `+3.54%`、ETTm1 `-0.89%`、Electricity `-6.12%`、Traffic `-11.53%`。这说明它在高维 Electricity/Traffic 上具有竞争力，在 ETTh1 仍落后 PatchTST，但比原始 Mamba-1/Mamba-2 更接近论文目标。

## 合同与工件

- 计划：[`mamba_ts_v3_multidataset_autocon_screen_20260922.json`](mamba_ts_v3_multidataset_autocon_screen_20260922.json)
- 严格分析器：[`scripts/analyze_mamba_ts_v3_multidataset_autocon_screen.py`](../../scripts/analyze_mamba_ts_v3_multidataset_autocon_screen.py)
- 机器读数：[`artifacts/exploratory/mamba_ts_v3_multidataset_autocon_screen_20260922/readout/`](../../artifacts/exploratory/mamba_ts_v3_multidataset_autocon_screen_20260922/readout/)
- 数据集：ETTh1、ETTm1、Electricity、Traffic；M features；H96；seed `2021/2022/2023`
- 两个 phase：`MambaTSv3AutoCon lambda=0` 与完全相同架构的 `lambda=0.001`
- 所有作业使用 `MeanStd`、相同 optimizer/预算、`skip_test=true`
- 计划仍为 `authorized=false` exploratory snapshot，不能追溯性升级为预注册确认实验。

## 与外部 baseline 的 H96 描述性比较

相同数据集、horizon、seed 的 plain baseline 参考来自 216-job 矩阵。`MambaTSv3AutoCon` 两个 lambda 的均值几乎一致，因此这里使用 `lambda=0.001` 作为候选列。

| 数据集 | 最佳外部 baseline | MambaTSv3AutoCon | 相对变化 |
| --- | ---: | ---: | ---: |
| ETTh1 | PatchTST `0.685596` | `0.709939` | `+3.55%` |
| ETTm1 | PatchTST `0.399746` | `0.395694` | `-0.89%` |
| Electricity | TimesNet `0.170231` | `0.159817` | `-6.11%` |
| Traffic | PatchTST `0.420740` | `0.371546` | `-11.52%` |

这些是 validation 描述性结果，不是测试排名、显著性检验或非劣效结论。Traffic 的优势值得进入更长 horizon 复验；ETTh1 的 PatchTST 差距必须保留为失败分析。

## 与 v4 ACF memory screen 的关系

同日完成的 v4 ACF screen 显示，给 v3 加 train-only ACF memory 和 representation fusion 后，Traffic 平均退化约 `+0.64%`，ETTh1 退化约 `+0.14%`；只有 Electricity 改善约 `-0.58%`，ETTm1 改善约 `-0.39%`。因此 ACF memory 目前不能作为快速 MambaTS 主线的默认组件。

这不是 ACF 实验的最终否定：旧 v4 工件是在 sidecar 接口补齐前产生的，当前代码新增了 MambaTS compact ACF prior 的保存/加载和几何校验。任何后续 v4/v5 复验都必须使用新的 source snapshot，并通过 sidecar hash replay；不能把旧 screen 追溯性改写成可回放结果。

## 资源读数

在 RTX 5090 D、FP32、device-resident synthetic H96 输入、50 warm-up/200 timed forward 的同形状协议下：

- ETTh1/C7/B32：MambaTSv3 `0.246 ms`，MambaTSv3AutoCon `0.273 ms`，PatchTST `0.291 ms`，Mamba-2 `0.672 ms`。
- Traffic/C862/B8：MambaTSv3AutoCon `0.646 ms`，PatchTST `0.886 ms`，Mamba-2 `0.670 ms`。

这只证明 prediction-only device-resident 负载；不等价于 train-step、数据加载或端到端训练速度。Traffic 的 v3 AutoCon 峰值分配约 `152.6 MB`，高于 PatchTST 和 Mamba-2，必须在正式效率表中报告。

## 论文门槛

当前最诚实的论文定位是“具有 AutoCon 表示学习接口的高效 patch/variable-scan MambaTS backbone”，而不是“AutoCon 带来稳定精度增益”。进入正式论文主表前仍缺：

1. 新 source snapshot 下的 sidecar-backed v4/v5 复验或明确停止 ACF memory 主线；
2. H336/H720、至少 5 seeds、完整 test release；
3. PatchTST、TimesNet、Mamba-2 的同合同训练速度、train-step、显存和参数量；
4. 对 ETTh1 长期预测落后的结构性解释。

