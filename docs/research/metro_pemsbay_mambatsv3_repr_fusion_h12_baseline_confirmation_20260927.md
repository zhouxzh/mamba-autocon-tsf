# MambaTSv3 AutoCon representation fusion: same-source H12 confirmation

这是 validation-only 的同源基线确认，不生成或读取 held-out test 指标。

## 结果

| 数据集 | fusion candidate | 最强基线 | 相对最强基线 | 外部 gate |
| --- | ---: | ---: | ---: | --- |
| METR-LA | `0.275437` | MambaTSv3 `0.285294` | `-3.455%` | 通过 |
| PEMS-BAY | `0.299235` | MambaTSv3 `0.303959` | `-1.554%` | 通过 |

基线包括 DLinear、NLinear、PatchTST、TimeMixer、iTransformer 和 plain MambaTSv3；共 36 个 GPU 作业，phase validator 为 `valid=true`，与候选共享源码快照。候选本身的 fusion off/on 配对也在两个数据集均 `3/3` 胜出。

## 效率边界

RTX 5090 D、B32、L96、H12 的统一 prediction-plus-representation 资源复测中，fusion-off/on 的 median latency 为 METR-LA `0.310/0.323 ms`、PEMS-BAY `0.331/0.349 ms`，treatment 额外开销约 `4.4%/5.2%`，低于预设 10% 预算。该候选仍快于同形状历史 PatchTST/TimeMixer 的 prediction-only 读数，但这不是完整端到端训练速度，也不能对 DLinear/NLinear 作无条件速度优越性比较。

## 决策

该候选可以进入冻结的 held-out confirmation 设计，但不能直接写成论文最终结果。测试确认必须预先固定候选、基线、五个 seed、streaming metric 和非劣界；测试读数只有在全部 phase validator 通过后才能读取。
