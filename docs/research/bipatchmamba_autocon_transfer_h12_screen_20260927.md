# BiPatchMamba + AutoCon H12 迁移复核

## 结论

本轮是四数据集、72-job、validation-only 筛选，使用公开 Bi-Mamba+ 的双向 Mamba/patch-token 思路；没有生成或读取测试指标。每个数据集使用两个 fresh seeds，外部基线包括 DLinear、NLinear、PatchTST、TimeMixer、iTransformer 和 MambaTSv3，并同时保留 BiPatch plain、AutoCon null（`lambda=0`）与 active（`lambda=.001`）。

严格 reader 检查通过 72/72 作业、统一 source snapshot、完整配置和无测试工件。但 active 在四个数据集都没有通过 external gate：

| 数据集 | Active | Null | Plain | Active vs null | wins/null | 最强基线 | Active vs 基线 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| ETTh1 | 0.320870 | 0.322460 | 0.316512 | -0.493% | 2/2 | MambaTSv3 0.312156 | +2.791% |
| Electricity | 0.122824 | 0.125729 | 0.108246 | -2.311% | 2/2 | TimeMixer 0.105503 | +16.417% |
| Traffic | 0.343654 | 0.347799 | 0.324494 | -1.192% | 2/2 | iTransformer 0.324679 | +5.844% |
| Weather | 0.248507 | 0.246464 | 0.220066 | +0.829% | 1/2 | TimeMixer 0.220755 | +12.572% |

## 机制判断

AutoCon active 相对 exact null 在 ETTh1、Electricity、Traffic 改善，说明表示学习路径确实参与了优化；但 active 在所有数据集都比 plain BiPatch 差，当前“表示到预测的加性修正”是有害或不稳定的。不能把 active 相对 null 的改善写成最终架构收益，也不能释放测试集。

资源预检显示 BiPatch active 的预测-only 中位延迟在 ETTh1/Electricity/Traffic/Weather 分别为 `0.388/0.439/0.877/0.398 ms`；它在 Electricity/Traffic 快于同形状 PatchTST 和 TimeMixer，但在 ETTh1/Weather 不快于所有非线性基线。因此速度只能逐数据集报告，不能宣称通用加速。

下一步固定双向 patch 主干，替换加性 representation correction 为有界、零初始化的 representation gate，再做小规模验证。完整机器读数见 [`strict readout`](bipatchmamba_autocon_transfer_h12_screen_20260927_readout/readout.md)。
