# PEMS 现代基线扩展审计

日期：2026-09-26  
状态：80/80 validation-only 基线作业已完成；不构成同快照候选比较或论文主表

## 已完成范围

本地 PEMS03/04/07/08 NPZ 数据均已通过数组 schema、通道轴、切分与 SHA-256 审计。两个独立的
冻结基线扩展各完成 40 个作业：M features、输入长度 96、预测步长 12/96、seed 2021--2025、
最多 20 epoch、`skip_test=true`。

| 基线 | 工件根目录 | 作业数 | 源码树 SHA-256 | 测试结果文件 |
| --- | --- | ---: | --- | ---: |
| iTransformer | `itransformer_pems_baseline_extension_20260924` | 40/40 | `158cd2baf8830dacd020d6f1638394f8aa761380febbd477ab193e01565f7cd1` | 0 |
| TimeMixer | `timemixer_pems_baseline_extension_20260923` | 40/40 | `158cd2baf8830dacd020d6f1638394f8aa761380febbd477ab193e01565f7cd1` | 0 |

全部 80 个 `loss_logs.json` 都存在且最佳 validation loss 有限。两个模型互相之间可在同一
源码快照、数据合同和训练预算下比较；它们与较早 PEMS MambaTS v3 候选的源码快照不同，不能直接
做显著性检验、相对效应量或论文表格排名。

## 五 seed 验证 MSE

下表是每个数据集与步长的五 seed 最小纯预测 validation loss 均值，数值只在同一数据集和
horizon 内有可比性。

| 数据集 | H | iTransformer | TimeMixer |
| --- | ---: | ---: | ---: |
| PEMS03 | 12 | 0.076468 | **0.065152** |
| PEMS03 | 96 | 0.289258 | **0.254709** |
| PEMS04 | 12 | 0.100887 | **0.081536** |
| PEMS04 | 96 | 0.422680 | **0.246799** |
| PEMS07 | 12 | 0.071580 | **0.065467** |
| PEMS07 | 96 | 0.348413 | **0.298046** |
| PEMS08 | 12 | 0.098689 | **0.092593** |
| PEMS08 | 96 | 0.491867 | **0.450937** |

TimeMixer 在这 8 个共享坐标上均低于 iTransformer。该观察不等同于它优于旧的
MambaTS v3、TimesNet 或其它历史基线，因为这些行来自不同的代码快照和训练批次。

## 对后续实验的影响

PEMS 不再只是候选的早期探索数据：它现在有多变量线性、patch Transformer、周期卷积、
inverted-variable Transformer、multi-scale MLP 和 Mamba 家族等多类比较器。下一轮候选验证
必须在一个新的、同源码冻结计划中同时重跑这些基线与 `MambaTSv3` plain、AutoCon-null 和
AutoCon-active；不能从本文件的 80 个结果中借用基线数值来判定候选非劣。

原始机器读出见 [`iTransformer readout`](../../artifacts/exploratory/itransformer_pems_baseline_extension_20260924/readout/readout.md)
和 [`TimeMixer readout`](../../artifacts/exploratory/timemixer_pems_baseline_extension_20260923/readout/readout.md)。
