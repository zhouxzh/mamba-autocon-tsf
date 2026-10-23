# CMamba-lite / AutoCon 工程 Pilot 复核

**状态：** 当前代码树上的工程性、验证集-only pilot；不是预注册确认性实验，也不支持论文或速度结论。

## 复核范围

本次只比较 `MambaACFVariableAutoCon` 的 variable-token 分支中 GDD adapter 的 off/on。两轮均为 ETTh1、Traffic，H96，seed=2021，最多 8 个 epoch，`MeanStd` 归一化，raw train-only ACF sidecar，`skip_test=true`。第一轮使用旧的 GDD 位置，第二轮把 variable-token `LayerNorm` 放到 GDD 之前以匹配 CMamba 的 block 顺序；两轮输出分别位于：

- [`mamba_cmamba_lite_autocon_initscale001_pilot_20260904`](../../artifacts/exploratory/mamba_cmamba_lite_autocon_initscale001_pilot_20260904/)
- [`mamba_cmamba_lite_autocon_initscale001_postnorm_pilot_20260904`](../../artifacts/exploratory/mamba_cmamba_lite_autocon_initscale001_postnorm_pilot_20260904/)

所有作业都成功完成，但每个数据集只有一个 seed，指标来自 `loss_logs.json:val_loss`，没有测试集指标。

## 数字

| GDD 位置 | 数据集 | off 最佳 val MSE | on 最佳 val MSE | on/off | 最佳 epoch |
|---|---|---:|---:|---:|---|
| 原实现（GDD 后再归一化） | ETTh1 | 0.7362870 | 0.7458854 | +1.304% | 1 / 1 |
| 原实现（GDD 后再归一化） | Traffic | 0.3805935 | 0.3869565 | +1.672% | 4 / 4 |
| CMamba 顺序（先归一化再 GDD） | ETTh1 | 0.7362871 | 0.7452553 | +1.218% | 1 / 1 |
| CMamba 顺序（先归一化再 GDD） | Traffic | 0.3806033 | 0.3880464 | +1.956% | 4 / 4 |

结论是两轮配对都不支持 GDD；这不是显著性结论，只是停止该工程分支的明确 pilot 信号。

## 为什么没有转化成收益

1. `.01` 初始化解决了严格零 gate 导致的仿射层零梯度问题，但没有保证方向正确。checkpoint 中 GDD raw scale 最终约为 ETTh1 `-0.01047`、Traffic `-0.01590`，adapter 只形成很小的负修正。
2. 预测路径还要乘外层 `variable_residual_scale` 和 `variable_gate`；Traffic 的外层尺度约 `-0.24`，gate bias 约 `-1.78`，所以 GDD 对最终预测的直接影响很弱，而它仍会进入 variable representation fusion 和 relation loss。
3. Traffic 的 GDD 参数从约 `171,355` 增至 `200,672`，增量 `29,317`（约 `17.1%`）。这超过原 screen 设定的严格资源预算，不能把 on/off 当作无成本公平比较。
4. ETTh1 最佳 epoch=1、Traffic 最佳 epoch=4；8 epoch 的短预算只足以做机制筛查，不能判断更长训练是否会反转方向。

## 决策

- GDD off/on 的两轮 pilot 均失败，按候选停止规则停止继续调 rank、gate 或归一化顺序。
- 保留 `mamba_variable_gdd_init_scale` 和 GDD 位置修正作为可复现实验接口；默认值仍为 `0.0`，不会改变旧配置。
- 后续精度工作优先验证已有较强信号的 train-only ACF memory、horizon-available lag bank 和 variable-token branch；这些分支要在同一当前源码、同一归一化和 sidecar 合同下重新配对。
- 不把本 pilot 与 `artifacts/paper_final` 的历史矩阵合并，也不把它写成 CMamba 复现或效率提升。

## 当前代码验证

本轮修改后的全量回归为 `581 passed, 40 warnings`。相关 runtime/resource contract、GDD gradient contract、训练计划转发和资源签名测试均通过。
