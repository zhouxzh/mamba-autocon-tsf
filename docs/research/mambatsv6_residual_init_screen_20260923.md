# MambaTS v6 残差初始化筛选：停止

## 假设

v6 在 Exchange 三个 checkpoint 中把初始为零的 Mamba residual scale 学到约 `0.10--0.12`，且 checkpoint 级反事实显示关闭已学残差会使 validation MSE 上升。因此本次只检验一个事先记录的值：将 raw residual scale 从 `0` 初始化为 `0.1`。其来源是 v6 Exchange 三个完成 checkpoint 的实际门值，不是本轮网格搜索。

除这一初始化外，warm 与 zero-init v6 都使用相同 DLinear 风格主路径、Mamba scan、`AutoCon=True`、`MeanStd`、`lambda=.001`、数据划分、20 epoch 上限和 patience 5。

## 冻结合同

- 数据集：Exchange-Rate、ILI；H96；seed `2021/2022/2023`。
- 模型臂：DLinear、NLinear、PatchTST、iTransformer、zero-init v6 AutoCon、warm v6 AutoCon。
- 完整性：`36/36` 成功；phase validator `valid=true`；所有作业 `skip_test=true`，无测试指标。
- 严格 readout：[`readout.md`](../../artifacts/exploratory/mambatsv6_residual_init_screen_20260923/readout/readout.md) 校验所有六臂坐标、冻结 source snapshot、AutoCon 参数和 warm 臂唯一的 `mamba_ts_residual_init=.1` 修改。

## 结果

| 数据集 | 最佳比较臂 | warm v6 MSE | 相对 zero-init v6 | warm 胜出 zero-init | 相对最佳 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exchange-Rate | PatchTST `0.259005` | `0.269641` | `-0.181%` | `2/3` | `+4.111%` |
| ILI | NLinear `0.207772` | `0.215418` | `+1.211%` | `0/3` | `+3.694%` |

提前打开残差在 Exchange 的均值改善极小，未缩小相对 PatchTST 的核心缺口；并且在 ILI 稳定退化。因此没有可推广的初始化收益。

## Exchange 通道级诊断

为判断 v6 的缺口是否只来自少数变量，对三个 seed 的 Exchange validation checkpoint 逐通道计算 scaled MSE。均值结果如下（通道顺序沿用数据文件）：

| 模型 | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | ch7 | ch8 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatchTST | .4725 | .2917 | .1960 | .4487 | .0032 | .2137 | .1647 | .2816 | .2590 |
| v6 zero-init | .5102 | .2943 | .2114 | .4523 | .0036 | .2313 | .1742 | .2837 | .2701 |

差距并非由单个异常通道造成：v6 在 8 个通道均未取得三 seed 的平均优势，主要误差质量集中在前 4 个通道。因而当前没有足够证据支持增加一个通道选择器或通道门控作为下一候选；这条诊断只用于排除“少数坏通道”解释，不构成测试集结论。

## 决策

停止 residual-init 方向，不做更大的初始化网格、不开放测试集、不做资源主张。此结果支持一个边界性解释：Mamba residual 在已训练的 Exchange checkpoint 中确实有效，但仅调整其初始门值不足以使其超过强线性/patch 基线。后续候选需要改变残差的预测信息或优化目标，而不是继续调 ReZero 标量。
