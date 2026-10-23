# TimeMachine-Mamba 基线当前读数

日期：2026-09-07  
状态：代码与资源 smoke 已完成；训练 screen 仍未授权、未启动。

## 这次加入了什么

新增 `MambaTM1`、`MambaTM2` 以及对应的 `MambaTM1AutoCon`、`MambaTM2AutoCon`。实现位于 [`models/MambaTimeMachineAutoCon.py`](../../models/MambaTimeMachineAutoCon.py)，结构参考 TimeMachine 官方实现的两级 latent projection、四路 Mamba 分支、残差融合和整段 horizon head：[官方仓库](https://github.com/Atik-Ahamed/TimeMachine)。

plain 模型不创建 AutoCon 表示分支；AutoCon 版本复用第一阶段 forecast projection，沿时间 token 轴生成 `[B, seq_len, d_ff]` 表示，供现有 AutoCon trainer 使用。`mamba_tm_channel_independent` 和 `mamba_tm_residual` 是显式配置项，并已进入 runner、计划校验、资源签名和模型注册。

计划中的 `mamba_causal_embedding=true` 只约束已有 MambaACF/普通 Mamba 路径；TimeMachine 实现直接从当前 look-back 值做投影和扫描，没有使用未来标签，因此该开关对 `MambaTM1/2` 不改变结构，已在解释边界中单独保留。

## 已验证证据

- `pytest -q`：598 passed，43 warnings。
- 资源工具定向测试：105 passed。
- 真实 CUDA 小步前向/反向：Mamba-1/Mamba-2 的 plain 与 AutoCon 均输出正确 forecast shape；AutoCon 表示 shape 为 `[B, seq_len, d_ff]`，forecast head 梯度非零。
- 训练 smoke：此前已完成 MambaTM2 plain/AutoCon 各 1 epoch 的 ETTh1 小配置运行；这些输出只证明训练链路可用，不进入结果表。

## 结构成本预筛查

输入为合成 device-resident 张量，ETTh1 形状、`B=4`、`seq_len=96`、`pred_len=96`，RTX 5090 D，warmup 10 次、计时 30 次；不包含数据加载、CPU-GPU 传输、反向传播或 checkpoint I/O。原始记录见 [`artifacts/exploratory/mamba_timemachine_resource_smoke_20260907`](../../artifacts/exploratory/mamba_timemachine_resource_smoke_20260907)。

| 模型 | 总参数 | median 前向 | p95 前向 |
| --- | ---: | ---: | ---: |
| Mamba2 | 74,175 | 0.666 ms | 0.683 ms |
| MambaTM2 | 304,496 | 1.512 ms | 1.608 ms |
| PatchTST | 108,448 | 0.268 ms | 0.306 ms |

这不是端到端速度结论，但足以确定当前 TimeMachine 四路结构约为普通 Mamba2 的 4.1 倍参数、2.3 倍预测前向延迟；因此它应作为多路径结构基线/AutoCon 兼容性对照，不能作为轻量化或“更快”主方法。后续若要追求效率，应单独注册共享权重、减少分支或 patch/token 化的候选，并重新做容量匹配和训练步耗时测量。

## 尚未得到的结果

[`mamba_timemachine_baseline_screen_plan_20260907.json`](mamba_timemachine_baseline_screen_plan_20260907.json) 登记了 5 个本地数据集、11 个模型臂、3 个 seed、H96，共 165 个训练作业。计划当前 `authorized=false`，没有正式 validation/test 指标，不能声称 TimeMachine 或 AutoCon 改善了预测精度，也不能据此判断论文可发表性。

正式启动前仍需人工复核：数据 fingerprint、模型容量公平性、plain/AutoCon 的 estimand、训练步资源协议和 test 指标冻结规则。资源 smoke 与训练结果必须分开归档。
