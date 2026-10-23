# MambaACF 逐预测步可靠性与有效记忆诊断契约

_人工复核稿 · 2026-09-03 · 仅代码/契约改进，未启动训练。_

## 结论

`mamba_acf_step_reliability` 是一个默认关闭的可回滚开关。关闭时，forecast
memory 使用一个按通道聚合、并广播到全部预测步的可靠性；开启时，每个预测步
只使用该步仍然合法的 lag，并保留 `[batch, horizon, channel]` 的可靠性形状。
该选项不改变 ACF prior、signed lag 系数或主干预测头，只改变 residual memory 的
可靠性乘子。

CLI、模型构造、资源 profile 和训练计划的默认值均为 `false`。旧配置缺失该字段时
仍按 `false` 解释；因此旧 checkpoint 的默认数值路径不变。显式开启时，
`resource_profile()` 记录：

- `acf_step_reliability=true`
- `acf_reliability_mode="stepwise"`
- `acf_step_reliability_path_active`（仅当 memory 分支实际有 prior/动态 ACF 时为真）

## 可靠性定义

静态 prior 的逐步可靠性由合法 lag 的 softmax 权重和 train-only ACF 幅值给出：

```text
r[h,c] = sum_k weight[h,c,k] * abs(acf[c,k]) * valid[h,k]
```

开启逐步模式时，`r[h,c]` 直接乘到该预测步的 seasonal residual；关闭时使用
跨 horizon 的 `aggregate_reliability[c]` 并广播。无效 lag 始终由 causal validity
mask 排除，不会复制边界值或读取未来标签。

## 有效记忆强度

训练日志和显式评估诊断保留原有的 `acf_effective_memory_rms`，并新增：

```text
acf_effective_memory_to_forecast_rms
  = RMS(effective_memory) / max(RMS(base_forecast_before_ACF), 1e-12)
```

该值只在 residual 加回预测之前、以 detached tensor 计算，因此不进入梯度图，
不改变任何预测数值。它比绝对 RMS 更适合跨归一化设置判断 ACF 分支是否真正参与
预测；同时应结合 `acf_runtime_gate_mean` 和 `acf_reliability_mean` 解读。评估路径
默认不收集 GPU->CPU 诊断，需显式打开 `mamba_collect_eval_acf_diagnostics`。

## Lag-bank 与 direct-anchor

当前两者的支持域是有意独立的：`mamba_acf_channel_lag_bank` 只约束 soft forecast
memory；`mamba_acf_anchor_copy` 仍从完整的已配置 lag 列表选择每个通道的最大
`abs(ACF)`。`resource_profile()` 明确记录
`acf_anchor_copy_lag_support="full_configured_lag_bank"` 和
`acf_anchor_copy_uses_channel_lag_bank=false`，避免把二者的组合误写成同一支持域。
若未来要让 anchor 遵守稀疏 bank，应作为新的显式候选并重新冻结配置和实验计划。

## 回归覆盖

`tests/test_mamba_acf_contract.py` 覆盖：

1. aggregate/stepwise profile 模式和默认路径；
2. stepwise 静态 prior 的实际激活状态；
3. 相对强度诊断的 detached/non-invasive 性质；
4. channel lag-bank 与 direct-anchor 同时请求时的独立支持域语义。
