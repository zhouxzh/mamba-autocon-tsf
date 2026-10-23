# 当前源码资源基线（预测专用路径）

更新日期：2026-09-02。该页只报告 device-resident synthetic forward 的结构资源读数，不报告预测精度、训练吞吐或统计显著性。

## 协议

- GPU：RTX 5090 D；PyTorch `2.8.0+cu128`；CUDA `12.8`。
- 输入：固定随机种子的 device-resident synthetic window；ETTh1 使用 `B=32`，Traffic 使用 `B=1`。
- 每个模型：100 次 warm-up、300 次计时；传输和数据加载不计入。
- Mamba-1/2 与 MambaACF 声明 `supports_return_repr_control` 时，计时调用 `return_repr=False`。这表示部署预测路径不返回 AutoCon 辅助表示；DLinear/PatchTST 使用其默认 forward。

## 读数

| 数据集 | 模型 | 总参数 | 可训练参数 | median ms | p95 ms | seq/s |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | DLinear | 18,624 | 18,624 | 0.06685 | 0.07144 | 474,086 |
| ETTh1 | PatchTST | 46,080 | 46,080 | 0.28283 | 0.29383 | 112,241 |
| ETTh1 | Mamba-1 | 36,135 | 31,975 | 0.37789 | 0.38509 | 84,564 |
| ETTh1 | Mamba-2 | 34,067 | 29,907 | 0.65765 | 0.66951 | 48,585 |
| ETTh1 | MambaACF AutoCon（scale/CLAM off） | 52,757 | 52,724 | 1.40067 | 1.42065 | 22,813 |
| Traffic | DLinear | 18,624 | 18,624 | 0.05837 | 0.06186 | 16,979 |
| Traffic | PatchTST | 46,080 | 46,080 | 0.27739 | 0.28543 | 3,584 |
| Traffic | Mamba-1 | 173,790 | 169,630 | 0.38667 | 0.39982 | 2,562 |
| Traffic | Mamba-2 | 171,722 | 167,562 | 0.67165 | 0.68400 | 1,487 |
| Traffic | MambaACF AutoCon（scale/CLAM off） | 185,326 | 185,293 | 1.41931 | 1.43896 | 703 |

原始基线工件：[`resource_baseline_current_20260902_v2`](../../artifacts/exploratory/resource_baseline_current_20260902_v2/)。MambaACF 的预测专用工件（当前源码，含 CLAM 因子缓存）：[`resource_repr_control_current_20260902_v3`](../../artifacts/exploratory/resource_repr_control_current_20260902_v3/)。

### 单尺度速度候选

为评估多尺度扫描是否是主要开销，固定 MambaACF AutoCon、raw 静态 ACF memory、`return_repr=False` 和上述计时协议，仅将 `mamba_scales` 从 `[1,2]` 改为 `[1]`。这是待精度验证的候选，不改变默认配置。

| 数据集 | `[1,2]` median ms | `[1]` median ms | 延迟变化 | 吞吐变化 |
|---|---:|---:|---:|---:|
| ETTh1, C=7, B=32 | 1.40067 | 0.85872 | -38.69% | 22,813 -> 37,193 seq/s |
| Traffic, C=862, B=1 | 1.41931 | 0.86306 | -39.19% | 703 -> 1,158 seq/s |

单尺度工件见 [`resource_single_scale_matched_current_20260902`](../../artifacts/exploratory/resource_single_scale_matched_current_20260902/)。两臂均固定 `causal_embedding=true`、普通 channel gate、raw static ACF memory 和相同 batch/计时设置；这只是 device-resident synthetic forward 的负载对照。单尺度是否保留预测精度、以及是否仍值得保留 AutoCon 关系学习，必须由同预算多 seed screen 决定。

### Reliability interpolation 结构资源 smoke

先做的 construction smoke 只验证模型构造和 manifest 字段，没有 GPU benchmark、训练或预测精度读数。该 smoke 使用的是最小结构配置（`channel_gate=false`、`relation_loss=false`、非因果 embedding），固定 ETTh1/H96、raw 静态 prior、`[1,2]` scales，关闭/开启开关的结构读数为：

| 开关 | 总参数 | 可训练参数 | 静态 buffer | 插值参数 | path active |
| --- | ---: | ---: | ---: | ---: | ---: |
| `false` | 48,450 | 46,305 | 0.632393 MB | 0 | false |
| `true` | 48,451 | 46,306 | 0.632393 MB | 1 | true |

对应工件为 [`reliability_resource_off_20260902`](../../artifacts/exploratory/reliability_resource_off_20260902/) 与 [`reliability_resource_on_20260902`](../../artifacts/exploratory/reliability_resource_on_20260902/)。因此该候选的额外容量是一个标量，且不会增加静态 ACF buffer；这部分不是速度或精度结论。

随后对同一候选做了单次 GPU 前向资源测量。这里使用 canonical 资源配置（`channel_gate=true`、`relation_scope=global`、`causal_embedding=true`），所以参数量不能与上面的最小结构 smoke 混比。协议为 ETTh1/M/H96、`B=32`、device-resident synthetic input、50 次 warm-up、200 次计时，`return_repr=false`，不含数据传输和数据加载：

| 开关 | 总参数 | 可训练参数 | median ms | p95 ms | seq/s | peak allocated MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `false` | 52,757 | 52,724 | 1.399312 | 1.423832 | 22,822.95 | 22.510742 |
| `true` | 52,758 | 52,725 | 1.506592 | 1.531974 | 21,196.90 | 22.511230 |

相对关闭开关，这一次测量的 median/p95 延迟分别为 `+7.67%`/`+7.60%`，吞吐为 `-7.12%`；`steady_state_peak_increment` 均为 `13.211426 MB`，没有观测到额外峰值增量。它是一次配对的资源负载读数，不能单独判断真实训练开销或统计稳定性。原始工件见 [`reliability_resource_benchmark_off_20260902`](../../artifacts/exploratory/reliability_resource_benchmark_off_20260902/) 与 [`reliability_resource_benchmark_on_20260902`](../../artifacts/exploratory/reliability_resource_benchmark_on_20260902/)。

对应的精度筛选已登记为 [`mamba_acf_single_vs_multiscale_screen_plan_20260902.json`](mamba_acf_single_vs_multiscale_screen_plan_20260902.json)，runner 已展开 `12` 个 job 并通过数据/配置验证，但 `execution.authorized=false`，没有训练工件。计划固定 ETTh1/Traffic、H96、seed `2021/2022/2023`、raw train-only ACF memory、global signed relation 和 `AutoCon_lambda=0.03`；两臂唯一架构差异是 temporal scales。该计划的比较阈值和停止规则不能由本资源表替代。

## 解释边界

在这组相同窗口和宽度下，MambaACF AutoCon 的预测专用 forward 约为 Mamba-2 的 `2.1x` 延迟（ETTh1/Traffic 均近似如此），也慢于 Mamba-1、PatchTST 和 DLinear。省略表示头只带来约 2% 的 MambaACF 延迟改善；主要成本仍在多尺度 Mamba、通道门控、ACF memory 和 horizon projection。因此当前代码不支持“ACF 让 Mamba 更快”或“比所有基线更高效”的陈述。

scale conditioning 与 CLAM 的 on/off 读数和未授权训练状态见 [`mamba_acf_scale_conditioning_review_20260902.md`](mamba_acf_scale_conditioning_review_20260902.md)。所有读数都是架构负载观察，不能替代同一训练预算、同一数据划分和多 seed 的预测精度实验。

关系目标的坐标合同也已加到计划验证器：启用 MambaACF AutoCon relation loss 时，`mamba_token_stride>1` 只允许 `relation_scope=global`；`local/both` 必须使用 `stride=1`。这是防止 lag 监督错位的输入校验，不是新的预测结果。
