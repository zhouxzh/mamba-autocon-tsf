# MambaTS v6 H96 结构筛选：停止

## 候选与动机

`MambaTSv6/MambaTSv6AutoCon` 是面向低维失败坐标的局部结构修改。此前 MambaTS v3 在 Exchange-Rate 和 ILI 即使使用 20 epoch 预算，仍持续落后 DLinear/PatchTST；而表示门控的线性 expert 也没有稳定改善。

v6 因而不再以 Mamba 预测头作为起点。它使用共享 DLinear 风格的 moving-average trend/seasonal 分解预测作为初始化主路径，Mamba patch scan 只通过一个零初始化 ReZero 标量学习残差校正。AutoCon 仍作用于真实时间索引的 Mamba 表示 `[B,T,d_ff]`，并使用同一 `MeanStd`、`lambda=.001` 合同。该设计不是把 DLinear 作为测试时回退：分解头与 Mamba 均在同一训练目标下更新，但初始化时预测严格等价于分解路径。

## 冻结验证合同

- 计划：`mambatsv6_h96_screen_20260923.json`；四数据集 H96、三 seed、最多 20 epoch、patience 5。
- 数据集：ETTh1、Weather、Exchange-Rate、ILI。
- 七臂：DLinear、NLinear、PatchTST、iTransformer、MambaTS v3 AutoCon、MambaTS v6 AutoCon、MambaSM AutoCon。
- 完整性：冻结源码快照中 `84/84` job 成功；phase validator `valid=true`；每个作业 `skip_test=true`，无测试指标。
- 严格读取：[`readout.md`](../../artifacts/exploratory/mambatsv6_h96_screen_20260923/readout/readout.md) 同时核验坐标覆盖、source snapshot、AutoCon 参数、v6 runtime variant 和无测试工件。

## 验证结果

三 seed validation MSE 均值，负百分比表示 v6 更低：

| 数据集 | 最佳比较臂 | v6 MSE | 相对最佳 | 相对 MambaTS v3 | v6 胜出 v3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | iTransformer `0.683150` | `0.704783` | `+3.166%` | `-0.795%` | `2/3` |
| Weather | v6 `0.443833` | `0.443833` | `0.000%` | `-0.395%` | `2/3` |
| Exchange-Rate | PatchTST `0.259005` | `0.270135` | `+4.299%` | `-2.357%` | `3/3` |
| ILI | NLinear `0.207772` | `0.212847` | `+2.450%` | `-15.280%` | `3/3` |

v6 成功缩小了 Exchange/ILI 的缺口，也让 ILI 超过 DLinear、Weather 成为七臂最佳。但它仍在 ETTh1、Exchange 和 ILI 各有一个强基线更低，不能宣称跨数据集非劣。

## 同形状资源测量

Weather/C21、`seq_len=96`、`pred_len=96`、`d_model=32`、`d_ff=64`、batch=8、CUDA device-resident 前向、20 warm-up 和 100 timed forwards：

| 模型 | 参数量 | median latency (ms) | sequences/s |
| --- | ---: | ---: | ---: |
| DLinear | 18,624 | 0.095 | 81,980 |
| PatchTST | 46,080 | 0.281 | 28,386 |
| MambaTS v3 AutoCon | 62,053 | 0.278 | 28,546 |
| MambaTS v6 AutoCon | 80,678 | 0.372 | 21,482 |

v6 既未比 v3/PatchTST 快，也远慢于线性路径；资源测量只代表该硬件上的 prediction-only 前向，不等于端到端训练吞吐。资源工具的确认性 artifact-hash 认证不适用于本 exploratory launcher 的轻量 phase validator；其输入配置、冻结 source snapshot 和严格 accuracy readout 仍可在相邻工件中复核，不能将该测量升级为确认性速度结论。

## 决策

停止 v6 的长 horizon 和测试释放。它证明“线性初始化 + Mamba 残差”比直接 Mamba 显著改善低维 ILI/Exchange，但在三项核心要求中仍失败两项：没有跨数据集达到强基线非劣，也没有速度优势。v6 保留为可复现负/机制消融；下一候选必须在不增加 v6 分解开销的前提下，针对 Exchange 的线性预测误差建立新的可证伪机制。
