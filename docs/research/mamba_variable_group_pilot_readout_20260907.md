# Mamba variable-token 分组 pilot 读数

日期：2026-09-07  
状态：单数据集、单 seed、validation-only 工程 pilot；不是论文结果。

## 目的和合同

该 pilot 只测试已有 `MambaACFVariableAutoCon` 路径的 `mamba_variable_group_size`。`group_size=1` 是逐通道 variable-token Mamba；`group_size=16` 先把连续通道 token 平均成组，做一次较短的 variable scan，再把组状态广播回原通道。除分组大小外，模型、AutoCon relation、ACF lag-pair sampler、归一化、优化器、数据划分、seed 和训练预算完全相同。

数据为 Traffic（862 通道）、H96、`seed=2021`、4 epoch、`hard_selected + mamba_acf_relation_lag_mask=true`、`AutoCon_lambda=0.03`。两臂均使用同一 ACF relation telemetry；没有读取 test split。

## 精度和训练时间

| group size | best validation loss | epoch trajectory | epoch-time sum (s) |
| ---: | ---: | --- | ---: |
| 1 | 0.387689 | 0.433492 / 0.396448 / 0.387689 / 0.388231 | 35.638 |
| 16 | 0.395293 | 0.443534 / 0.407520 / 0.395831 / 0.395293 | 37.145 |

相对 `group_size=1`，`group_size=16` 的 best validation loss 上升 `1.96%`，4 epoch 训练/验证 wall time 上升 `4.23%`。两臂的 relation telemetry 相同（global contrastive-active anchors 约 `84--85%`，degenerate positives 约 `8--9%`，lag exposure 约 `0.31--0.35`），因此该退化不是由 relation sampler 失效造成的。

## 资源扫描

所有记录使用 Traffic/C862/H96、`B=8`、RTX 5090 D、10 次 warm-up、50 次 timed prediction-only forwards；不包含反向、优化器、数据加载或传输。

| group size | total/trainable params | median ms | p95 ms | peak increment MB | sequences/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 169,242 / 166,041 | 1.236992 | 1.259379 | 32.167 | 6,453.9 |
| 4 | 169,242 / 166,041 | 1.310160 | 1.330582 | 21.221 | 6,095.9 |
| 8 | 169,242 / 166,041 | 1.319792 | 1.336298 | 21.221 | 6,058.1 |
| 16 | 169,242 / 166,041 | 1.338240 | 1.357774 | 21.221 | 5,971.7 |
| 32 | 169,242 / 166,041 | 1.324848 | 1.359315 | 21.221 | 6,025.3 |
| 64 | 169,242 / 166,041 | 1.318112 | 1.343360 | 21.221 | 6,059.3 |

相对 `group_size=1`，`group_size=16` 的 median/p95 延迟分别增加 `8.19%/7.81%`，吞吐下降 `7.47%`，峰值增量下降 `34.03%`。其它组大小也都比 group 1 慢：median 增幅约 `5.71%--8.19%`。参数量不变，因为分组只改变 token scan 长度，没有删除每通道 embedding/head。

## 结论

- 当前“分组 scan 后广播”实现没有产生速度优势，不能作为论文中的 efficient Mamba 方案。
- 它确实降低了高维 Traffic 的峰值工作集，但精度退化且训练时间上升；该收益不足以抵消主目标中的速度要求。
- 主要原因是 variable embedding、每通道 forecast head、广播和 kernel launch 仍然存在，缩短 recurrent token 序列没有转化为端到端前向收益。
- 该分支按停止规则不进入正式多数据集 screen；保留资源和单 seed 结果作为负结果，避免后续重复尝试同一实现。

更合理的下一步是设计真正减少每通道计算的轻量路径，例如共享时间 Mamba + 低秩/分组输出投影，并让 AutoCon 表示只在训练时启用、部署时关闭 auxiliary projection；这需要新的模型身份、参数审计和 paired pilot，不能把当前 group 结果解释成已解决。

原始工件：[`artifacts/exploratory/mamba_variable_group_pilot_20260907`](../../artifacts/exploratory/mamba_variable_group_pilot_20260907)。
