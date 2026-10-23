# MambaACF 尺度条件与局部 ACF 复核稿

更新日期：2026-09-02。本文是当前代码、资源工件和未授权实验登记的人工复核入口，不是论文结果表。

## 结论先行

当前 `MambaACF` 已有两个默认关闭的候选机制，均保留 AutoCon 表示分支：

- `mamba_acf_scale_conditioning`：用训练集拟合的 ACF 先验和当前归一化 encoder window 的相关性摘要，给已有多尺度 fusion 的 scale logits 加一个小的全局 bias。
- `mamba_acf_local_agreement`（CLAM）：在静态 ACF memory 的 lag 权重上，用当前 window 与训练先验的 signed agreement 做样本/通道级重加权。

两者都已经通过结构和 CUDA 合同测试，但**都没有当前源码下的预测精度结果**。最新资源测量使用 prediction-only forward（AutoCon 表示头显式关闭），仍显示它们增加前向成本，不能称为加速方案。

## Scale conditioning

实现使用六个 window-dependent 描述量：加权 signed correlation、加权 absolute correlation、与 prior sign 的 agreement、lag-weighted absolute correlation、peak absolute correlation 和 high-lag activity。至少 16 个观测对的 lag 才参与摘要；当前窗口在 channel gate 之前保存，shared/CI 两条 scale fusion 路径分别使用相同语义的 bias。conditioner 的 ReZero 标量零初始化，因此关闭时不改变初始预测路径；先验来源固定为 raw train-only ACF。

### 当前资源读数

| 数据集 | scale-off 参数 | scale-on 参数 | median 延迟 | 吞吐 | 峰值显存增量 |
|---|---:|---:|---:|---:|---:|
| ETTh1, C=7, B=32 | 52,757 | 53,048 | 1.401 -> 1.659 ms (+18.43%) | 22,813 -> 19,280 seq/s (-15.49%) | 13.211 -> 13.211 MB |
| Traffic, C=862, B=1 | 185,326 | 185,617 | 1.419 -> 1.686 ms (+18.80%) | 702.7 -> 592.8 seq/s (-15.65%) | 2.908 -> 2.908 MB |

读数来自同一 RTX 5090 D、device-resident synthetic input、100 warm-up/300 timed forwards；两臂使用相同配置和 `balanced` lag sampler，并通过 `return_repr=False` 关闭辅助表示输出。当前工件在 [`resource_repr_control_current_20260902_v3`](../../artifacts/exploratory/resource_repr_control_current_20260902_v3/)；旧的 `resource_scale_conditioning_current_20260902_v2` 和 `resource_repr_control_current_20260902_v2` 是缓存修复前/完整表示路径的历史读数，不应与本表混用。这里的 latency/throughput 是架构负载测量，不是端到端训练时间，也不是统计重复。

同一协议还测了一个单尺度速度候选：只把 `mamba_scales` 从 `[1,2]` 改为 `[1]`，并保持 `causal_embedding`、普通 channel gate、raw static memory、relation 配置和 batch 完全一致。ETTh1 的 median 为 `1.40067 -> 0.85872 ms`（`-38.69%`），Traffic 为 `1.41931 -> 0.86306 ms`（`-39.19%`）。这说明第二条 Mamba 扫描是主要结构开销之一，但没有单尺度精度结果；它只能作为下一轮配对 screen 候选。原始记录见 [`resource_single_scale_matched_current_20260902`](../../artifacts/exploratory/resource_single_scale_matched_current_20260902/)。

## CLAM（local ACF agreement modulation）

CLAM 只作用于静态、train-only ACF memory 的 lag mixture。对满足 `length - lag >= 16` 的 lag，计算当前 normalized observed window 的 signed correlation，并乘以训练先验的 sign 和 `sqrt((length-lag)/length)` confidence。这个 agreement 通过一个零初始化的 ReZero 标量进入 `exp(logit)` lag reweight；memory value 的 sign 仍来自训练先验，避免短窗口噪声改写先验方向。无合格 lag 时回退到历史 static-memory 路径；CLI/plan 对 raw source、static memory 和 lag 支持做前置拒绝。

CLAM 不修改 AutoCon relation loss 或 representation head，不读取 decoder/target/future 值。当前 ETTh1/Traffic、`seq_len=96` 且 lags `[24,168,336]` 的资源对照如下（只有 lag 24 满足 16-pair 条件）：

| 数据集 | CLAM-off median | CLAM-on median | 延迟变化 | 参数变化 | 峰值显存变化 |
|---|---:|---:|---:|---:|---:|
| ETTh1, C=7, B=32 | 1.401 ms | 1.734 ms | +23.80% | +1 | 无变化 |
| Traffic, C=862, B=1 | 1.419 ms | 1.718 ms | +21.08% | +1 | +3.128 MB |

原始工件在 [`resource_repr_control_current_20260902_v3`](../../artifacts/exploratory/resource_repr_control_current_20260902_v3/)。Traffic 的峰值显存增量为 `6.036 - 2.908 = 3.128 MB`；该差值受 allocator 状态影响，应视为本协议下的观测值。当前实现缓存了同一 window 的 local agreement 因子，减少重复 `exp`，但总体瓶颈仍在 ACF/多尺度前向。结果只说明当前实现的计算代价和形状合同通过，**不说明 CLAM 能提高预测精度**。

## 数据矩阵与授权状态

`mamba_acf_scale_conditioning_screen_plan_20260902.json` 已通过 validate-only，展开为 162 个严格配对的 H96 作业（6 个数据集、7 个基线、3 seeds）；其 `execution.authorized=false`，本轮没有启动训练。CLAM 目前只有代码/资源合同，没有已登记的精度 screen，也没有生成训练工件。与之分开的单尺度/多尺度精度计划 [`mamba_acf_single_vs_multiscale_screen_plan_20260902.json`](mamba_acf_single_vs_multiscale_screen_plan_20260902.json) 只登记 12 个 ETTh1/Traffic job，亦未授权、未训练。

若批准下一步，建议先做一个小型 current-tree 配对 screen：ETTh1 与 Traffic、H96、同一 MambaACF AutoCon 配置、local off/on、3 个 seed。预先固定 paired scaled-MSE/MAE、seed-level aggregation、训练时间、median/p95 latency、峰值显存和 source/config/sidecar hash；若精度不改善且无法抵偿开销，应停止扩展。该 screen 仍需人工授权，不能由本文件自动启动。

关系表示的时间坐标约束已在 runner 中前置检查：`mamba_token_stride>1` 配合 `relation_scope=local` 或 `both` 会被拒绝；当前计划和后续正式 relation 实验统一使用 `stride=1`。这项 guard 只修复实验合同，不等于精度改善。

## 当前可支持与不可支持的陈述

可以支持：Mamba-1/Mamba-2 的实现入口存在；scale/CLAM 的 raw-source、prior-sign、causal-window 和 shared/CI 广播合同已测试；资源工具和训练计划能记录相关字段；当前实现的资源开销已被测量。

不能支持：scale conditioning 或 CLAM 提高预测精度；AutoCon/ACF 普遍优于基线；MambaACF 普遍更快（当前预测专用资源表反而慢于 Mamba-1/2）；synthetic forward 读数代表端到端训练速度；当前候选已经达到论文发表所需证据。

人工复核建议先看本页，再看 [`manual-review.md`](../manual-review.md) 和 [`mamba_current_status.md`](mamba_current_status.md)；[`mamba_acf_core_status_20260901.md`](mamba_acf_core_status_20260901.md) 仅作历史快照。
