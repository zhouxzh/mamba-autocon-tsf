# NLinearVariableMambaAutoCon Traffic 归一化消融读出

日期：2026-09-25  
状态：探索性 validation-only；测试目录为空

## 设计

Traffic H336/H720、seed 2022/2023，严格复跑同源码 iTransformer 和冻结的 `NLinearVariableMambaAutoCon`。候选只把 `AutoCon_wnorm` 从前一轮的 `LastVal` 改为 `MeanStd`；Mamba 结构、预测连接表示、ACF relation objective、lambda、数据划分和训练预算全部不变。

## 结果

| Horizon | LastVal 候选均值 | MeanStd 候选均值 | MeanStd 相对 LastVal | iTransformer 均值 | MeanStd 相对 iTransformer |
|---:|---:|---:|---:|---:|---:|
| 336 | 0.376081 | 0.374894 | -0.32% | 0.368431 | +1.75% |
| 720 | 0.402033 | 0.405335 | +0.82% | 0.397886 | +1.87% |

MeanStd 只在 H336 有小幅改善，在 H720 退化，并且两个 horizon 均未达到 iTransformer 非劣。该归一化差异不是 Traffic 缺口的稳定解释，因此停止继续搜索 `LastVal/MeanStd/ReVIN` 组合。

## 审计边界

- 8/8 作业成功，两个 phase 的测试目录均为空，未使用测试集选择配置。
- 数据哈希：`cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`。
- 这是归一化消融，不构成 AutoCon relation loss 的独立贡献证据，也不构成跨数据集泛化或论文主表结果。

