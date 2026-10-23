# MambaTS v3 全局预测连接 AutoCon 校准（2026-09-22）

## 问题与合同

预测连接的全局 representation residual 在 Electricity 有一致收益，但 `lambda=.003` 使 ETTh1/ETTm1 退化。本校准在同一冻结结构下比较 `lambda=0`、`.0003`、`.001`、`.003` 和 `.003` 前四 epoch warm-up；四数据集、H96、三个 seed、8 epoch，测试集不运行。

启动前的选择规则是：候选在四个数据集的平均 MSE 均不得退化，且至少三个数据集须有多数 seed 改善。没有任何候选满足该规则。

## 结果

| 候选 vs null | ETTh1 | ETTm1 | Electricity | Traffic | 通过规则 |
| --- | ---: | ---: | ---: | ---: | --- |
| `.0003` | `+0.033% (2/3)` | `+0.020% (1/3)` | `-0.387% (2/3)` | `+0.144% (1/3)` | 否 |
| `.001` | `+0.177% (2/3)` | `+0.053% (1/3)` | `-0.752% (3/3)` | `+0.208% (1/3)` | 否 |
| `.003` | `+0.229% (1/3)` | `+0.081% (1/3)` | `-0.662% (3/3)` | `-0.056% (3/3)` | 否 |
| `.003` + warm-up 4 | `+0.085% (2/3)` | `+0.044% (1/3)` | `-0.755% (3/3)` | `+0.186% (1/3)` | 否 |

warm-up 相对同一 `.003` 无 warm-up 在 ETTh1/ETTm1 改善 `-0.144%/-0.037%`，但 Traffic 退化 `+0.243% (0/3)`。这排除了“只需减小或延迟原损失”这一解释。

## 审计与边界

- 计划：[`mamba_ts_v3_global_repr_autocon_calibration_20260922.json`](mamba_ts_v3_global_repr_autocon_calibration_20260922.json)
- 严格 reader：[`analyze_mamba_ts_v3_global_repr_autocon_calibration.py`](../../scripts/analyze_mamba_ts_v3_global_repr_autocon_calibration.py)
- 原始读数：[`readout.md`](../../artifacts/exploratory/mamba_ts_v3_global_repr_autocon_calibration_20260922/readout/readout.md)
- `60/60` 作业成功；reader 验证五阶段 source hash、配置、融合开关、精确坐标、`effective_lambda` 调度和零测试工件。

结论是停止当前全局融合的 λ/warm-up 调参，而不是停止 AutoCon 研究。下一种候选必须改变表征几何、正负样本或 MambaTS 主干容量，并仍以 active/null、强基线和效率合同验证。
