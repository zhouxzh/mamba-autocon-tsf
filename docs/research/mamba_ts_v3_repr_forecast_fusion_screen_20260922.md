# MambaTS v3 预测连接 AutoCon 表示 screen（2026-09-22）

## 结构动机

此前 v3 的 AutoCon representation 只供辅助损失使用，预测 head 不读取它。这里新增全局共享的 representation-to-forecast ReZero residual：初始尺度为零，训练后 MSE 和 AutoCon 都直接作用于同一表示。模型级测试验证了零尺度预测不变，以及 MSE 对融合尺度存在非零梯度。

## 结果

24/24 个 H96 validation-only 作业完成，覆盖四数据集和 seed `2021/2022/2023`；严格 reader 检查每个 config 均为 `mamba_ts_repr_forecast_fusion=true`，并确认测试工件为零。

| 数据集 | `lambda=0` MSE | `lambda=.003` MSE | active vs null | 胜出 seed |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.708066 | 0.709703 | `+0.229%` | 1/3 |
| ETTm1 | 0.387871 | 0.388185 | `+0.081%` | 1/3 |
| Electricity | 0.163316 | 0.162223 | `-0.666%` | 3/3 |
| Traffic | 0.375235 | 0.375026 | `-0.055%` | 3/3 |

这证明“预测连接”不是形式化修补：AutoCon 在高维 Electricity 上出现一致改善。但 ETT 两个低维数据集退化，故不能宣称普遍收益，也不能进入测试集或论文主表。

## 效率与下一步

Traffic/C862/B8 的 prediction-plus-representation 合成 GPU forward 中，全局融合为 `0.720 ms`、峰值增量 `152.6 MB`。它仍快于历史 PatchTST prediction-only smoke 的 `0.886 ms`，但两者的表示路径和基准协议不同，不能写成正式端到端速度结论。

通道保留的完整时间网格版本在第一个 Traffic 训练作业超过 7 分钟后按效率门槛中止；该未完成矩阵没有精度结果。后续 patch-level 通道保留资源 smoke 降至 `1.415 ms`、`343.7 MB`，仍约为全局融合的两倍，且慢于 PatchTST。因此该通道保留候选已停止，不应重启。

- 冻结计划：[`mamba_ts_v3_repr_forecast_fusion_screen_20260922.json`](mamba_ts_v3_repr_forecast_fusion_screen_20260922.json)
- 严格读数：[`readout.md`](../../artifacts/exploratory/mamba_ts_v3_repr_forecast_fusion_screen_20260922/readout/readout.md)
- 全局资源 smoke：[`model_resources.json`](../../artifacts/exploratory/mamba_ts_v3_global_repr_fusion_resource_20260922/model_resources.json)
- patch-level 通道保留资源 smoke：[`model_resources.json`](../../artifacts/exploratory/mamba_ts_v3_channelwise_patch_resource_20260922/model_resources.json)

下一轮只能尝试不随通道数线性放大激活的选择性/低秩路由，并且必须保留 active/null 控制和强基线速度测量。
