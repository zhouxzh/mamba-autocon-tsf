# MambaACF AutoCon 损失尺度审计

**日期：** 2026-09-02  
**状态：** 代码审计与实验登记；没有启动新的精度训练。

## 审计范围

本审计只回答一个问题：当前 AutoCon relation objective 的数值尺度、梯度归属和 signed 项，是否足以支持把 `AutoCon_lambda=0.03` 当作通用默认值。它不评价任何尚未运行的候选结构。

## 已确认的计算路径

训练器先对模型返回的 representation 做一次 `F.normalize`，`ACFRelationLoss.forward` 进入后又做一次归一化。对非零向量这是数学上的幂等操作；在 CPU/GPU synthetic probe（`B=5,T=6,D=7`）中，single 与 double normalization 的 loss 差不超过 `3.6e-7`，梯度相对差不超过 `2.4e-7`，零向量梯度也保持有限。因此当前不删除任一层：内层是 loss API 的输入边界，外层保留训练器历史语义。未来若清理，必须提升 objective 版本并提供等价回归。

当前训练目标仍是：

```text
total = MSE + effective_lambda * AutoCon_relation
```

`effective_lambda` 由 CLI 的 `AutoCon_lambda` 和可选 warm-up 决定。现有 v3 日志中，relation candidate coverage 约 `2.977%`、active-anchor 约 `91.447%`；`AutoCon_relation` 均值约 `0.578`，加权 auxiliary/MSE 的标量比例约为 `2.7%--8.7%`。这些是 loss 数值比例，不等于参数更新比例。

在 synthetic MambaACF probe 中，按共享 trunk 参数计算，`0.03 * ||g_aux|| / ||g_mse||` 约为 `0.071--0.124`（5 个 seed），两类梯度 cosine 约为 `-0.020--0.059`。这说明 auxiliary update 是中等偏小、方向近乎正交的正则项；整模型梯度不能用来做 GradNorm，因为 forecast head 只收到 MSE，`repr_head` 主要收到 auxiliary 梯度。

当前代码还把 relation loss 拆成四个 detached epoch-level 字段并写入 `loss_logs.json`：`acf_relation_global_magnitude_ce`、`acf_relation_global_signed_hinge_contribution`、`acf_relation_local_magnitude_ce` 和 `acf_relation_local_signed_hinge_contribution`。其中 `signed_hinge_contribution` 已包含 `sign_weight`，同一 scope 的 CE 与 contribution 之和可以重构该 scope 的 relation 均值；空 scope 或未启用 AutoCon 写入 `null`。旧日志缺少这些字段时视为 unavailable，不回填或与新日志混合。

另一个重要边界是 signed 项：历史 v3 的 `sign_hinge` 均值约 `0.00103`，乘 `sign_weight=0.25` 后只占 auxiliary 数值约 `0.0415%`。`sign_accuracy` 约 `91%` 不能单独证明模型学到了负/正相位关系，可能来自正 ACF 和相似度偏置。当前不放大 `sign_weight`，也不把现有结果写成强 signed supervision。

## 对模型归因的限制

`MambaACF.forward` 在 forecast ACF memory 加入前，从 shared hidden 生成 AutoCon representation。默认 `mamba_acf_repr_fusion=false` 时，relation loss 不直接监督 ACF memory 或 forecast gate；它监督的是表示分支。只有显式打开 representation fusion，才可把 ACF context 纳入 relation 表示，而且这应作为独立消融。

此外，普通 `DataEmbedding.TokenEmbedding` 使用 circular `Conv1d`。因此 raw ACF lookup 的 forecast-origin 索引可以是因果的，但默认整个网络不是严格因果；需要 `mamba_causal_embedding=true` 才能使用严格左填充的 embedding 路径。

## 工程决策

- 保留 double normalization，不在没有新 objective 版本和等价测试前改变训练语义。
- 不引入自动 GradNorm、uncertainty weighting 或盲目增大 `sign_weight`。
- 把 `AutoCon_lambda` 当作需配对验证的超参数；先固定架构、sidecar、sampler、seed 和 validation gate，再比较预先登记的 lambda 网格。
- 训练日志同时报告 MSE、auxiliary 数值、coverage、sign diagnostics；若要报告梯度，必须只在共享 trunk 参数交集上计算。

对应的未授权 lambda 校准计划见 [`mamba_acf_lambda_calibration_screen_plan_20260902.json`](mamba_acf_lambda_calibration_screen_plan_20260902.json)。该计划不允许使用 test metric 选 lambda，也不产生确认性或发表结论。
