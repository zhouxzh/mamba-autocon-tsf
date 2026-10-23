# MambaSM H96 结构筛选：停止

## 目的与结构

本次筛选实现了一个 S-Mamba/TimesMamba 启发的本地候选 `MambaSM` 与 `MambaSMAutoCon`。模型将历史窗口由 `[B,T,C]` 转换为逐变量 token `[B,C,T]`，以共享线性投影得到 `[B,C,D]`，再沿变量轴执行正向和反向 Mamba-1 scan、残差 FFN 和直接多步预测头。AutoCon 版本另以每个历史时刻的跨变量表示 `[B,T,D]` 计算既有关系损失。

这不是外部 S-Mamba 或 TimesMamba 的逐行复现。它的作用是检验“显式双向变量扫描 + 保持 AutoCon 时序表示”是否能修复当前 MambaTS v3 在低维数据集的缺口，同时维持更低资源开销。

## 冻结验证合同

- 计划：`mambasm_h96_screen_20260922.json`，SHA-256 `609af78cc72f5f1da79e1fb3b436e634711bd538e35f4405c5ada435ac34dfaf`。
- 数据集：ETTh1、Weather、Exchange、ILI；H96；seed `2021/2022/2023`。
- 模型臂：DLinear、NLinear、PatchTST、iTransformer、MambaTS v3 AutoCon、MambaSM AutoCon。
- 预算：最多 20 epoch、patience 5；候选与 MambaTS v3 使用相同 `AutoCon=True`、`MeanStd`、`lambda=.001`。
- 完整性：`72/72` 作业均成功；`phase_validation.json` 为 `valid=true`；每一条均 `skip_test=true`，无测试指标。

## 验证结果

下表为三 seed 的 validation MSE 均值。百分比是 MambaSM 相对同坐标最佳基线的变化；负值才代表改善。

| 数据集 | 最佳比较臂 | MambaSM | 对最佳基线 | 相对 MambaTS v3 | 相对 MambaTS v3 胜出 seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | iTransformer `0.683150` | `0.691130` | `+1.169%` | `-2.718%` | `3/3` |
| Weather | MambaTS v3 `0.445616` | `0.453806` | `+1.838%` | `+1.846%` | `0/3` |
| Exchange | PatchTST `0.259005` | `0.280882` | `+8.448%` | `+1.535%` | `1/3` |
| ILI | NLinear `0.207772` | `0.266697` | `+28.359%` | `+6.149%` | `0/3` |

它只在 ETTh1 上稳定优于当前 MambaTS v3，但仍未达到 iTransformer；Weather、Exchange、ILI 均退化，其中低维 Exchange/ILI 的差距尤其明显。因此该变量扫描不解决当前问题。

## 同形状资源测量

在 CUDA、device-resident prediction-only 前向条件下，固定 `seq_len=96`、`pred_len=96`、`d_model=32`、`d_ff=64`、`e_layers=1`、batch=8，进行 20 warm-up 和 100 次测量：

| 通道数 | 模型 | 参数量 | median latency (ms) | sequences/s |
| ---: | --- | ---: | ---: | ---: |
| 7 | MambaSM | 38,176 | 0.281 | 25,608 |
| 7 | MambaTS v3 | 61,016 | 0.279 | 28,481 |
| 21 | MambaSM | 38,624 | 0.399 | 19,965 |
| 21 | MambaTS v3 | 62,053 | 0.307 | 29,582 |

MambaSM 的参数较少，但并不更快；在 21 通道的形状下反而约慢三成。该结果不代表端到端训练吞吐，也不应外推至其他硬件。

## 决策

`MambaSM/MambaSMAutoCon` 作为已实现的公开结构基线保留，但停止其长 horizon 扩展和测试释放。它不满足“精度不低于强基线且更快”的主模型门槛。后续候选必须优先解决低维线性/趋势主导序列，而不是增加变量轴 Mamba 扫描；AutoCon 仍保留为受控表示学习接口，不能因该模型包含该损失而声称精度收益来自对比学习。
