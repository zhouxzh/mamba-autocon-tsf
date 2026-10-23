# Traffic 原生基线 pilot 读数

日期：2026-09-07  
状态：单数据集、单 seed、4 epoch、validation-only 工程 pilot；不是确认性结果。

## 合同和公平性边界

本 pilot 在 Traffic（862 通道）、H96、`seed=2021` 下运行 DLinear、NLinear、
PatchTST、TimesNet、iTransformer、Mamba-1 和 Mamba-2。所有作业使用相同的训练/验证
切分、batch size=32、学习率、优化器、4 epoch 和 `skip_test=true`，没有读取 test split。
普通基线使用各自原生的模型与归一化路径；`AutoCon=false` 时，配置里保留的
`AutoCon_wnorm=LastVal` 不会启用 MambaACF 的 `MeanStd` 窗口归一化。因此下表是
“native baseline pilot”，不能与使用 AutoCon/MeanStd 的 MambaACF 候选直接做论文排名。

Mamba-1/2 使用 `mamba_d_state=16`、`d_conv=4`、`expand=2`、`headdim=16`、单尺度、
`mamba_causal_embedding=true`；其它基线不携带 Mamba 专用开关。训练耗时只从日志中
读取 epoch wall time；GPU 资源是独立的 synthetic prediction-only 测量。

## Validation 结果

| 模型 | 参数（可训练） | 最佳 validation MSE | 最佳 epoch | 4 epoch wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| DLinear | 18,624 | 0.513800 | 3 | 8.577 |
| NLinear | 9,312 | 0.508901 | 4 | 8.939 |
| Mamba-1 | 169,630 | 0.497026 | 4 | 10.338 |
| Mamba-2 | 167,562 | 0.494128 | 4 | 25.533 |
| PatchTST | 46,080 | 0.438770 | 4 | 22.443 |
| iTransformer | 14,944 | 0.442530 | 4 | 11.616 |
| TimesNet | 1,302,046 | 0.431119 | 4 | 20.456 |

在这组短预算 native pilot 中，TimesNet 的验证损失最低，iTransformer 和 PatchTST
次之；Mamba-2 比 Mamba-1 略好，但仍高于 PatchTST `12.6%`、高于 TimesNet `14.6%`
（相对各自 MSE）。这只是一个 seed 的方向性结果，不能替代多 seed 确认。

## GPU prediction-only 资源

协议：RTX 5090 D、FP32、输入驻留 device、`B=8`、10 次 warm-up、50 次 timed forward；
不包含数据加载、CPU/GPU 传输、反向、优化器或 relation loss。

| 模型 | 总参数 | median (ms) | p95 (ms) | sequences/s | 峰值增量 (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLinear | 18,624 | 0.078736 | 0.087202 | 100,032.8 | 13.258 |
| NLinear | 9,312 | 0.033632 | 0.040725 | 229,690.7 | 7.576 |
| Mamba-1 | 173,790 | 0.427360 | 0.440315 | 18,677.0 | 23.104 |
| Mamba-2 | 171,722 | 0.716608 | 0.738003 | 11,116.7 | 23.104 |
| PatchTST | 46,080 | 0.888256 | 0.898965 | 8,990.6 | 95.188 |
| iTransformer | 14,944 | 0.342352 | 0.352885 | 23,170.9 | 8.629 |
| TimesNet | 1,302,046 | 2.796272 | 2.813582 | 2,859.3 | 17.918 |

Mamba-1/2 的 synthetic forward 比 PatchTST 快，但 iTransformer 更快且验证损失更低；
DLinear/NLinear 在速度和内存上明显占优。因训练和推理协议、归一化路径以及模型容量
仍不完全相同，这些数字只能用于基线覆盖和资源预算，不能写成 Mamba 的普遍加速结论。

## 对论文主线的意义

- 当前 Mamba-1/2 没有在这组 native pilot 中形成“精度不低于基线且更快”的 Pareto
  证据；Mamba-2 相比 Mamba-1 的改善不足以弥补与 PatchTST/TimesNet 的精度差距。
- MambaACF/AutoCon 候选必须在同一 normalization、同一训练预算和多 seed 下重新对齐
  后，才能与这些 baseline 比较；不能把 AutoCon 的 `MeanStd` 结果和本表直接合并。
- 这组 pilot 证明当前代码已经可以稳定运行完整的 baseline 入口，并给出了后续多数据集
  screen 的最低覆盖模板。下一步应优先固定 normalization parity，再扩展 ETTh1、ETTm1、
  Electricity 和 Traffic 的 3 seed/H96，最后才考虑多 horizon 确认性实验。

原始工件：[`artifacts/exploratory/baseline_traffic_pilot_20260907`](../../artifacts/exploratory/baseline_traffic_pilot_20260907)。
