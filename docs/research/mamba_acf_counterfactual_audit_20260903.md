# MambaACF ACF 残差与同权重反事实审计草稿

> 状态：人工复核草稿。本文档只整理已有 checkpoint 的只读重放和代码公式，不启动训练，不把反事实结果写成独立重训的因果效果。

## 1. 审计问题

本审计回答两个不同问题：

1. ACF memory 是否真的进入已经训练好的预测路径，模型在预测时对该分支有多大依赖？
2. 当前代码中的 signed ACF 如何从训练集先验变成一个受可用性约束的残差校正？

第一个问题使用同一 checkpoint 的 memory-on/off 反事实；第二个问题使用源码和 sidecar 合同核对。二者都不能替代“memory-on 与 memory-off 分别重训”的正式机制估计。

## 2. 反事实协议

### 2.1 固定坐标和来源

| 字段 | 固定值 |
|---|---|
| phase | `acf_memory_relation_h96` |
| 模型 | `MambaACFAutoCon` |
| 数据 | `ETTh1`, `traffic` |
| 预测长度 / 输入长度 | `H=96`, `L=96` |
| seed | `2021, 2022, 2023`，每个数据集 3 个 |
| 特征 | `M`，输出通道数分别为 7 和 862 |
| ACF 来源 | train-only、raw、signed prior |
| 选定 forecast lags | `[72, 96, 120]` |
| 完整 ACF 表 | 最大 lag `720` |
| ACF 温度 | `4.0` |
| embedding / stride / scales | causal embedding、`stride=1`、`scales=[1]` |
| relation | global、`hard_selected`、`lambda=0.03` |
| 归一化 | `LastVal` |
| 测试聚合 | streaming window-weighted；`origin_stride=96` |

工件根目录为 `artifacts/paper_final/mamba_acf_fair_baseline_screen_20260901/acf_memory_relation_h96/`。重放使用该 phase 的 immutable source snapshot，而不是当前可能发生漂移的工作树；source tree SHA-256 为 `89e80c9ed59202d0a45fdd2ff422bcffbb86aaaf714091b31b5ed81a262bf4bf`，记录的 git head 为 `f394705adbf5217811276f5e59576c6e7ea7586c`。环境记录为 Python `3.9.25`、PyTorch `2.8.0+cu128`、`mamba_ssm=2.3.1`、NVIDIA RTX 5090 D。

数据集内容哈希为 `ETTh1=f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`、`traffic=cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`。逐 seed 工件索引如下；完整的其余 artifact hashes 仍以对应 phase 的 `phase_validation.json` 为准。

| 数据 | seed | config hash | checkpoint SHA-256 | ACF prior SHA-256 |
|---|---:|---|---|---|
| ETTh1 | 2021 | `4d5169d90107` | `9425bb9b02f33ebdbe51d3096e495c765ec98e15278b16bfd3c59e090e68688d` | `5a31ee71318cabfd0862b763f57ef3b954b6aeadbf478cd6ad8fe4199078905d` |
| ETTh1 | 2022 | `1220b877fc65` | `cb68bcad89d431ba92857aeab9c3d0cc1f2982ecf3e4ebc7c2620c4e30c36ec2` | `5a31ee71318cabfd0862b763f57ef3b954b6aeadbf478cd6ad8fe4199078905d` |
| ETTh1 | 2023 | `648f30518133` | `46a24e38d408f164468da29aa971631db0260685e58e6095ed749f727103ba67` | `5a31ee71318cabfd0862b763f57ef3b954b6aeadbf478cd6ad8fe4199078905d` |
| traffic | 2021 | `a90cd4c960a5` | `cb00025110fe12571b0a89fa42ba5e8b728f612b9d0115b69bb5574e42643569` | `45c1716bc5dcd27ba345f824aa20c49b4472b81c17d38f267617b58c41e86334` |
| traffic | 2022 | `2a5eefb46690` | `faf7f0fdab37d28a221074d39b3f50e0680882fb88f2e7936ad0af38e3922bb2` | `45c1716bc5dcd27ba345f824aa20c49b4472b81c17d38f267617b58c41e86334` |
| traffic | 2023 | `d7dbe495fd51` | `4792d984668e6b5b805fd10b6e4000e4fad90ea4af99f15ee07d0b6453986cd9` | `45c1716bc5dcd27ba345f824aa20c49b4472b81c17d38f267617b58c41e86334` |

### 2.2 操作定义

对每个数据集和 seed：

1. 加载相同的 `checkpoint.pth`，并按 `config.json` 校验相同的 `acf_prior.npz`。
2. 在完全相同的测试窗口、batch 顺序和数据归一化下计算原始配置 `use_acf_memory=true` 的输出。
3. 仅将内存模型实例的 `use_acf_memory` 改为 `false`，不重置参数、不改变 dropout/eval 状态、不重新训练，再计算同一批窗口。
4. 计算
   `Delta_MSE = MSE_off - MSE_on`，以及 `relative_rise = 100 * (MSE_off / MSE_on - 1)`。
5. 同时记录 `acf_runtime_gate_mean`、`acf_effective_memory_rms` 和 on/off 输出差的 RMS。

这是“同一已训练函数对分支屏蔽的敏感性”估计。它说明 checkpoint 是否使用了 memory，不能说明重新训练一个 memory-off 模型会得到同样的差值。

### 2.3 源快照重放读数

以下数字由完整测试集只读重放得到，显示 6 位小数以避免把浮点累加误差解释成实验精度。

| 数据 | seed | on MSE | off MSE | off 相对上升 | on MAE | off MAE | 输出差 RMS | gate 均值 | effective memory RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 2021 | 0.398514 | 0.471780 | 18.385% | 0.421874 | 0.466508 | 0.256230 | 0.380823 | 0.244256 |
| ETTh1 | 2022 | 0.388908 | 0.438604 | 12.778% | 0.408032 | 0.439976 | 0.213908 | 0.320994 | 0.204600 |
| ETTh1 | 2023 | 0.392078 | 0.463292 | 18.163% | 0.414479 | 0.457388 | 0.250492 | 0.387098 | 0.240367 |
| traffic | 2021 | 0.573470 | 0.632941 | 10.370% | 0.386473 | 0.420050 | 0.223594 | 0.254511 | 0.220751 |
| traffic | 2022 | 0.558801 | 0.618975 | 10.768% | 0.385922 | 0.417639 | 0.224286 | 0.257206 | 0.220257 |
| traffic | 2023 | 0.560083 | 0.622727 | 11.185% | 0.385308 | 0.416101 | 0.235373 | 0.272381 | 0.231653 |
| ETTh1 均值 | - | 0.393167 | 0.457892 | 16.463% | - | - | - | - | - |
| traffic 均值 | - | 0.564118 | 0.624881 | 10.771% | - | - | - | - | - |

测试流包含 ETTh1 `2785` 个窗口、`30` 个 origin 抽样点，以及 traffic `3413` 个窗口、`36` 个 origin 抽样点。窗口有重叠，窗口和 origin 不能作为独立重复；主统计单位仍是 seed。

### 2.4 读法和不一致值处理

- 所有 6 个 seed 的 off MSE 都高于 on MSE，且 gate/effective-memory 均非零。这支持“该 checkpoint 学会使用 ACF residual expert”的窄结论。
- 这是同权重屏蔽诊断，不是 memory-on/off 独立重训的平均处理效应，也不是 ACF 先验在总体上的因果证明。
- 本文档以 source snapshot 重放和 `stream_metrics.json` 为准。此前工作笔记中出现过不一致的 ETTh1 即时值（约 `0.444` 一组）；由于它们无法与本 phase 的 config、loader、stream artifact 同时对齐，暂不纳入结果表，须另行追溯后才能使用。
- 这组 phase 只有 H96、每数据集 3 个 seed，故不报告 seed-level 显著性，不把跨数据集 pooled 均值当作独立重复，也不把 origin/window bootstrap 当作确认性置信区间。

## 3. ACF 残差校正公式

设归一化后的观测窗口为 `x[b,t,c]`，时间长度为 `L`，预测步为 `h=1,...,H`，训练集拟合的每通道 signed ACF 为 `a[c,k]`，选定滞后集合为 `K`。当前 static raw memory 的核心索引和掩码为

```text
i(h,k) = L - 1 + h - k
v(h,k) = 1{0 <= i(h,k) < L}
```

只要 `v=0`，该滞后不能提供观测值，不能循环复制旧周期或把端点 padding 当作真实观测。设温度为 `tau`，则可用滞后的权重为

```text
q(c,k)       = log(max(|a[c,k]|, eps)) / tau
w(h,c,k)     = v(h,k) exp(q(c,k)) / sum_j v(h,j) exp(q(c,j))
s[b,h,c]     = sum_k w(h,c,k) sign(a[c,k]) z[b,i(h,k),c]
```

其中 `z` 是 memory source：

- `raw`：`z=x`；
- `residual`：`z=x-centered_smoother(x)`；
- `trend`：`z=centered_smoother(x)`。

当前严格确认候选应优先使用 `raw`。代码中的 residual/trend 采用每个输入窗口独立的居中平滑，而先验拟合可能在完整 train 序列上分解；二者在窗口边界并不数值等价。

静态缓存还定义了 ACF 强度和可用性可靠性。逐步可靠性可写为

```text
r_step(h,c) = sum_k w(h,c,k) |a[c,k]| v(h,k)
```

默认 `acf_step_reliability=false` 时，代码使用 horizon 聚合的 `r_aggregate(c)`，并将其广播到所有 `h`。隐藏状态门为

```text
g[b,h,c] = sigmoid(acf_gate(hidden[b,L-1,:]))
e[b,h,c] = g[b,h,c] * r(h,c) * s[b,h,c]
forecast_normalized = forecast_base + e
```

通道独立模式下 gate 的形状按 `(b,c)` 展开；variable-token 模式则从变量 token 产生逐通道 gate。最后再执行既有的 restore/inverse normalization。

实现对应关系：

- `_build_static_acf_memory_terms` 预先生成 `[H,C,K]` 的系数、索引和 valid mask；
- `_static_acf_memory` 按 lag 逐项累加，避免在高维 batch 中长期保留完整 `[B,H,C,K]` 候选张量；
- `_forecast_normalized` 将 `effective_memory` 作为 residual 加到主预测；
- `acf_prior`、系数和索引是 train-only 非持久化输入，sidecar 缺失时静态 memory 必须报错；
- `AutoCon` relation 表示在另一条 `hidden_full -> repr_head -> ACFRelationLoss` 路径上。默认 `mamba_acf_repr_fusion=false` 时，relation loss 不直接监督 memory、`acf_gate` 或 forecast head。

## 4. 高效使用 ACF 的建议

### 4.1 先验选择

训练集 FFT 只拟合一次，并保存 signed compact peaks 和完整 ACF 表。对候选滞后使用

```text
score(k) = mean_c |a[c,k]| * legal_forecast_fraction(k)
```

其中 `legal_forecast_fraction(k) = mean_h v(h,k)`，只在 train peak 集合内排序。当前 `L=H=96`、`K=[72,96,120]` 时，可用比例分别为 `0.75, 1.00, 0.75`。长滞后若有效 pair 很少，应同时报告 support，不应只报告 ACF 大小。

### 4.2 高维 Traffic 的结构选择

Traffic 的 `C=862` 通道在共享 Mamba trunk 中先压到 `d_model=32`，再由共享时间头恢复通道。ACF 的共同滞后表因此只能提供低维的共享残差提示，可能丢失每通道不同周期。后续若重新训练，优先级应是：

1. 每通道稀疏 peak 或 grouped lag bank，避免把所有通道强行绑定到同一组 lag；
2. 低秩、逐通道的 horizon amplitude/gate，而不是 dense `C x C` 交互；
3. 与 channel-independent、variable-token 结构做同一 seed、同一预算的配对比较；
4. 对 fixed/random-sign 或 availability-matched sham 做归因对照。

这只是下一轮实验设计建议。direct-anchor 分支当前只有结构和资源 smoke，没有 accuracy 结果，不能提前写成改进。

### 4.3 严格因果表述

若论文使用“严格左因果”措辞，至少固定 causal embedding、`token_stride=1`、单尺度 `scales=[1]`、非双向扫描，使用 prefix-causal normalization，并避免 residual/trend 的居中全窗口变换。当前 `LastVal/Mean/ReVIN` 都是 full-window normalization，因此不能满足该条件。ACF lookup 本身按 `L-1+h-k` 可因果地读取历史，但这不自动保证整个表示网络或 memory source 因果。

## 5. 统计与报告边界

主报告至少应保留以下字段：

```text
protocol_type = same_checkpoint_counterfactual
training_performed = false
phase, dataset, seed, model, horizon, seq_len
checkpoint_sha256, acf_prior_sha256, config_hash
source_tree_sha256, dataset_sha256, environment
acf_source, selected_lags, temperature, validity_rule
memory_on_mse, memory_off_mse, delta_mse, relative_rise
memory_on_mae, memory_off_mae, output_delta_rms
gate_mean, effective_memory_rms
window_count, origin_count, origin_stride
```

解释规则：

- 同一 seed 的 on/off 是配对观测；跨数据集不得当作独立重复。
- 反事实表只证明已训练 checkpoint 的分支依赖；独立重训 screen 才能估计训练后性能差异，必须另列 provenance、optimizer、预算和 sidecar。
- MSE/MAE 优先报告 normalized scaled space，original-space 指标作为补充；MAPE/MSPE 在接近零的目标上不作为主证据。
- 任一 checkpoint、config、sidecar、数据哈希或 source snapshot 不匹配，都应将对应坐标标为 invalid，而不是用另一条记录替换。
- 当前证据仍是 H96、每数据集 3 seed 的 exploratory diagnosis，不能据此宣称普适 ACF 增益、严格因果收益、SOTA 或端到端速度优势。

## 6. 现有文档建议更新位置

1. [`mamba_acf_core_status_20260901.md`](mamba_acf_core_status_20260901.md)：在“配对 relation null control”之后加入本审计的短表和链接；在“ACF 的高效用法”中补充 `i(h,k)`、valid mask、gate residual 公式，并明确 relation 与 memory 的路径分离。
2. [`mamba_current_status.md`](mamba_current_status.md)：在“ACF 效用审计”附近添加本文件链接，只保留 ETTh1/traffic 的均值摘要和“同权重反事实，不是重训效果”的一句边界。
3. [`../manual-review.md`](../manual-review.md)：在现有 sidecar replay/因果性限制条目后增加：反事实重放不提升训练证据等级；早先未对齐的 ETTh1 即时值不能进入主表。
4. [`../README.md`](../README.md)：把本文件列为“反事实与 ACF residual 公式审计”，并标注只读、未训练。
5. 如需写入论文附录，引用 phase 的 `provenance.json`、`phase_validation.json`、每个 run 的 `config.json`、`acf_prior.npz` 和 `stream_metrics.json`，不要只引用本摘要表。
