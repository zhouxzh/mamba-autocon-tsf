# 核心审计：Mamba、AutoCon 与 ACF

> **现行状态（2026-10-09）：** 本页以下正文是 2026-09-10 的历史架构筛选快照，
> 不能作为当前论文主结论。当前证据主线是 one-scan Mamba-1 local-cluster
> ACF-InfoNCE，且已完成三域 independent-host active/null 复现和独立官方
> PatchTST 扩展。请先读
> [主干选择](research/final_backbone_selection_20261009.md)、
> [论文骨架](research/local_cluster_manuscript_outline_20261009.md) 和
> [提交交接](research/mambatsv3_acf_infonce_submission_handoff_20261008.md)。
> 只读重验：`python scripts/verify_final_local_cluster_evidence.py`。

**更新时间：** 2026-09-10  
**用途：** 人工复核当前代码和结果；不是论文定稿。

## 结论先行

最新的 v3 主干 screen 说明：MambaTSv3 在 H96 的高维数据上明显强于原始 Mamba-1/Mamba-2，但严格的 `lambda=0.001` 对 `lambda=0` 对照在 H96 以及新的 H336/H720 复验中都没有稳定改善，长预测最大平均退化为 `+0.110%`。v4 的 ACF memory 扩展还在 Traffic 上退化约 `0.64%`。因此当前主线应是高效 v3 backbone + AutoCon 接口，ACF memory 作为待复验/可停止的消融，而不是把 ACF 组件堆叠到主模型。详细数字见 [`mamba_ts_v3_multihorizon_autocon_screen_20260922.md`](research/mamba_ts_v3_multihorizon_autocon_screen_20260922.md)。

最新的 48-job MambaACF 候选 screen 已完成四数据集、H96、三 seed 的 ACF-off/memory/AutoCon null/relation 配对。`lambda=0.001` 候选在 Electricity 和 Traffic 低于同坐标外部 baseline 的最佳均值，但 ETTh1/ETTm1 仍落后 PatchTST；相对同架构 `lambda=0` 的 relation 增量为 `-0.04%/-0.01%/-0.15%/+0.02%`。这支持继续做长 horizon 复验，不支持“AutoCon 已经提升精度”或“普遍优于基线”。同时，immutable plan 仍标记 `authorized=false`，所以该 screen 不能追溯性地写成预注册确认实验。完整合同和 sidecar/hash 审计见 [`mamba_acf_multidataset_candidate_screen_20260910.md`](research/mamba_acf_multidataset_candidate_screen_20260910.md)。

2026-09-10 代码与实验合同更新：普通 Mamba-1/Mamba-2 及其 AutoCon 变体现在真正响应 `mamba_causal_embedding=true`，显式开启时使用已有的左填充 `CausalDataEmbedding`；默认关闭时仍走历史 `DataEmbedding`。MambaTS v3/v4/v5 已完成 PEMS 强基线、fresh-seed、AutoCon lambda 校准、TimesNet 交叉复核和 train-only ACF 探索；v5 signed anchor 已归档为中性消融。当前工作树的后续回归为 `828 passed, 45 warnings`。

单独使用 Mamba 做时间序列分析是可行的，但应把它定位为**时间维度编码器**，
而不是天然的多变量预测器。当前仓库的主要失败不是 Mamba 扫描太慢，而是
Traffic 的 862 个变量先被压到 `d_model=32` 的共享状态，异质的通道动态和
跨变量关系在压缩时丢失。

因此当前最合理的论文定位是：

> train-only ACF 引导的、保留通道预测路径的 Mamba 机制研究。

不能定位成“纯 Mamba 普遍优于 PatchTST”或“AutoCon 必然带来增益”。

最新的完整 plain-baseline 矩阵已经补齐了此前缺失的同来源坐标：四个数据集、三
horizon、三 seed、DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2，共 `216/216`
个 validation-only 作业。结果显示 ETTh1 三个 horizon 都由 PatchTST 最好，
Electricity 三个 horizon 都由 TimesNet 最好；Mamba-2 主要在 Electricity/Traffic
上优于 DLinear/NLinear，但仍普遍落后 PatchTST/TimesNet。因而 plain Mamba 不能
作为“精度不低于全部基线”的论文主模型；它现在的价值是高维数据上的可运行控制和
效率候选。完整表、成对 seed delta 和 provenance 见
[`mamba_multidataset_plain_baseline_full_screen_20260910.md`](research/mamba_multidataset_plain_baseline_full_screen_20260910.md)。

PEMS 四数据集的 MambaTS v3 screen 中，v3 相对 Mamba-2 在两个 fresh seed 的 16/16 坐标改善，且短预算下比 DLinear/PatchTST 明显更低；但 TimesNet 在部分 H=96 坐标更强。AutoCon 直接接入共享编码器后，`lambda=0.03` 会造成小幅退化；两轮零权重控制校准显示 `lambda=0.001` 仅在 11/16 坐标改善、平均变化 `-0.0095%`，因此当前只能说“低权重近似保持预测性能”，不能说 AutoCon 已经带来增益。详细数值见 [`mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md`](research/mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md)。

固定 `lambda=0.001` 后，v3 AutoCon 与 TimesNet 的 16 个精确 PEMS 坐标中 `10/16` 改善；H=12 为 `8/8`、平均 `-19.50%`，H=96 只有 `2/8`、平均 `+6.74%`。因此论文主张应限定为短期候选，不能写成全跨度 SOTA；PEMS03/04 H=96 的 TimesNet 优势是必须保留的失败分析。详见 [`mamba_pems_ts_v3_autocon_strong_baseline_20260910.md`](research/mamba_pems_ts_v3_autocon_strong_baseline_20260910.md)。

v4 在 v3 上增加 train-only 静态 ACF residual，并用零初始化 ReZero gate 控制其影响。16 个 PEMS/seed/horizon 坐标全部完成 validation-only 训练，ACF gate 全部为非零；相对 v3 在 H=12 为 `3/8`、平均 `+0.074%`，H=96 为 `5/8`、平均 `-0.057%`。相对 TimesNet，H=12 为 `8/8`、平均 `-19.449%`，H=96 为 `2/8`、平均 `+6.678%`。这说明分支已激活但没有解决长期短板，v4 仍是 exploratory 候选，详见 [`mamba_pems_ts_v4_autocon_screen_20260910.md`](research/mamba_pems_ts_v4_autocon_screen_20260910.md)。

为验证 lag=288 日周期，另做了同窗口 `seq_len=384` 的 8-job v3/v4 配对；v4 仅 `1/4` 胜出，平均相对变化 `+0.737%`，平均 wall-time 约增加 `3.2%`。这条长上下文方向已停止扩展，说明“让 ACF lag 可访问”本身不足以解决 H=96 短板；详见 [`mamba_pems_ts_long_context_acf_screen_20260910.md`](research/mamba_pems_ts_long_context_acf_screen_20260910.md)。

最新的 ACF representation-fusion screen 把 train-only raw ACF summary 注入 v4 AutoCon 的 representation，16/16 作业成功且 representation gate 全部非零；相对原 v4，H=12 为 `3/8`、平均 `+0.075%`，H=96 为 `4/8`、平均 `-0.244%`，相对 TimesNet H=96 仍为 `+6.379%`，wall-time 约增加 `3.7%`。这支持“AutoCon 表示路径已接收 ACF 信息”的实现结论，不支持“AutoCon/ACF 带来稳定精度收益”。详见 [`mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.md`](research/mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.md)。

最新的 variable-token 分组 pilot 也没有解决效率问题：Traffic/C862 上把 variable scan
按 4--64 个通道分组后，峰值显存约降 34%，但 median forward latency 反而增加
`5.71%--8.19%`，`group_size=16` 的短训 validation loss 退化 `1.96%`。这条“分组后广播”
实现已停止扩展；后续效率方案必须真正减少每通道 embedding/head 或改用低秩输出，而不能只缩短
recurrent token 序列。

`mamba_variable_repr_fusion` 的单因素 pilot 也已停止扩展：ETTh1/Traffic H96、seed=2021
的 best validation loss 分别退化 `0.216%/0.096%`，prediction-only 延迟在两个数据集上
方向相反，不能支持加速或精度提升。它只增加一个小型表示 adapter，并不减少 variable
embedding、Mamba scan 或 forecast head 的每通道工作，因此不能作为效率方案；完整读数见
[`mamba_variable_repr_fusion_pilot_readout_20260907.md`](research/mamba_variable_repr_fusion_pilot_readout_20260907.md)。

## 代码实际做了什么

默认 MambaACF 的数据流为：

```text
x [B,T,C]
  -> 窗口归一化（默认 LastVal）
  -> DataEmbedding: C -> d_model
  -> 共享时间 Mamba 扫描 [B,T,d_model]
  -> 时间投影 + 线性残差 -> [B,H,C]
  -> 可选 train-only ACF forecast-memory residual
```

AutoCon 默认拿到的是 encoder representation；relation loss 与预测 head 分离，
所以默认情况下 relation loss 不能被解释为直接训练 ACF memory。新增的
`mamba_acf_relation_representation=forecast` 只是一个受限消融：它把最终归一化
预测做全 horizon pooling 后用于 global relation loss，仍不是逐预测步的动力学目标。

历史 `Mamba1AutoCon`/`Mamba2AutoCon` 也不是裸 Mamba forecaster：它们只替换
AutoConNet 的特征提取器，预测端仍是 AutoConNet 的 trend/seasonal head。论文中
必须把 Mamba-1、Mamba-2、AutoCon head 和 MambaACF 分开命名。

## 当前结果能支持什么

冻结的高维后验表（不是当前源码重训，也不是原始预注册确认）如下，指标为
scaled MSE：

| 数据集 / H | PatchTST | Mamba2 | Mamba2AutoCon |
|---|---:|---:|---:|
| Electricity / 96 | 0.184448 | 0.183130 | 0.182321 |
| Electricity / 336 | 0.207953 | 0.211477 | 0.210460 |
| Electricity / 720 | 0.252428 | 0.241182 | 0.241146 |
| Traffic / 96 | 0.501767 | 0.648306 | 0.649773 |
| Traffic / 336 | 0.516942 | 0.623489 | 0.623169 |
| Traffic / 720 | 0.550696 | 0.659914 | 0.658317 |

Traffic 上 Mamba2 比 PatchTST 高约 `29.2%/20.6%/19.8%`；AutoCon 没有改变
方向。这是广泛的通道级现象，不是少数异常通道，说明瓶颈在跨通道表达和
共享低维状态，而不是简单增加 Mamba 层数。

已有短预算 exploratory screen 给出的窄结论是：同一已训练函数中 ACF memory
确实被使用；独立重训的 memory on/off 在 H96、每数据集 3 seed 下约有
`-1.168%` 的 pooled MSE 变化，`lambda=.03` relation 相对 null 约 `-0.979%`。
这些结果只有筛选意义，不能替代多数据集、至少 5 seed 的确认性重训，也不能
据此声称普遍超过基线。

最新的 direct signed ACF anchor fresh-seed held-out replication（ETTh1/Traffic、H96、
5 seed、当前冻结源码）为 ETTh1 `-0.563%`（`5/5`）与 Traffic `-0.978%`（`4/5`）
scaled MSE，合计 `9/10` 配对点下降。这只是单机制的 exploratory 证据。随后 24-job
anchor x soft-memory validation-only 2x2 screen 的合并臂相对 plain 为 ETTh1
`-0.656%`（`3/3`）与 Traffic `-0.475%`（`2/3`），但 interaction 在 ETTh1 为正、
Traffic 为负，pooled 95% CI 跨零。因此不能把“两个分支都可有小幅下降”写成协同，
也不应继续堆叠 ACF 开关。两个 screen 的 causal embedding、seed cohort 与
test/validation 路径不同，禁止合并统计；详见
[`interaction readout`](research/mamba_acf_anchor_memory_interaction_screen_readout_20260904.md)
和 [`erratum`](research/mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md)。

在同一份公平 H96 screen 中，`MambaACF`（shared temporal path + ACF memory）
相对 `PatchTST` 的 MSE 仍为 `+1.912%`、MAE 为 `+6.375%`，且只有 `0/6`
个配对点胜出；相对 `DLinear` 的较低误差不能归因于 ACF，因为模型 head、容量和
训练路径同时改变，训练耗时还约增加 `118%`。因此“memory 在同架构内有线索”
与“模型超过外部基线”是两个不同命题，必须分别报告。该读数见
[`mamba_acf_fair_baseline_screen_readout_20260901.md`](research/mamba_acf_fair_baseline_screen_readout_20260901.md)。

此外，`artifacts/baselines/naive_metrics.csv` 是 `features=S` 的单变量 naive
结果，不能与这里的多变量 `features=M` 神经网络表直接排名；多变量 naive 只能使用
对应扩展 screen 的独立工件并注明协议差异。

## ACF 应该怎样用

按当前证据，优先级如下：

1. **逐通道 forecast-memory support。** 当前实现的
   `mamba_acf_channel_lag_bank` 在同一个 train-only lag bank 上按
   `abs(ACF) * legal_step_fraction` 为每个通道保留 top-k 槽位。它只改变
   forecast residual 的支持域，不增加 learned `C x C` 参数；这是最直接的
   ACF 利用方式。
2. **严格的 horizon legality。** 只允许索引
   `seq_len - 1 + h - lag` 落在已观测窗口内，长 lag 在前段 horizon 不可用时
   必须被 mask，而不能复制边界值。逐步 reliability 可以作为独立消融。
3. **通道保持的主干。** 在 Traffic 上应比较 shared temporal Mamba、
   channel-independent/分组 Mamba，以及低秩 cross-channel mixer；否则 lag
   residual 只能修补共享状态，无法恢复被压缩的通道动态。
4. **关系学习只做辅助目标。** self-ACF 只描述单通道周期，不能推出通道间图。
   若要建跨变量关系，应在 train split 估计 lagged cross-correlation，并使用
   稀疏图或低秩混合器；不能把 self-ACF profile 排序当作 cross-correlation。

不建议把通道 `strength/similarity` 排序作为主方法。它们是低成本 order ablation，
默认仍为 `input`；排序改变变量-token 的邻接假设，必须配合 reverse/permutation
审计。`similarity` 的一次性 CPU 构造在 `C=862,K=8` 约 `5.98 ms`，但没有
任何精度结果。

## 已完成的资源检查

Traffic 形状 `B=8,T=96,H=96,C=862`、Mamba-2 单尺度、raw static prior、
50 warm-up/200 timed prediction-only forwards：

| lag-bank | median ms | p95 ms | peak allocated MB | trainable params |
|---|---:|---:|---:|---:|
| off | 0.780848 | 0.861395 | 42.444824 | 150667 |
| on, top-k=2 | 0.780432 | 0.861320 | 42.645020 | 150667 |

这是一个 device-resident synthetic resource smoke。中位数差异约 `-0.05%`，
在测量噪声范围内，应记录为“没有证明速度变化”，而不是速度提升。原始记录在
[`channel-lag-bank resource artifact`](../artifacts/exploratory/mamba_acf_channel_lag_bank_resource_current_20260903/)。

## 下一轮最小实验门槛

不要直接把所有开关组合成一个大矩阵。建议按以下顺序：

1. 固定 `MeanStd`（与 PatchTST population variance 合同一致），在 ETTh1、
   ETTm1、Electricity、Traffic 上补齐 DLinear/NLinear/PatchTST/TimesNet/
   Mamba1/Mamba2 的三 horizon plain baseline。
2. 在 ETTh1、Electricity、Traffic 的 H96、3 seed validation-only screen 中，
   只比较 Mamba2、MambaACF-off、train-only memory、channel-preserving/variable
   branch 和 memory+branch；每次只改变一个结构因素。
3. 只有候选相对 MambaACF-off 不退化、Traffic 不再明显落后 PatchTST、每个
   seed 的 sidecar/config/hash 完整且端到端预算公平，才升级到四数据集、三
   horizon、5 seed 的确认性矩阵。
4. 选择始终只看 validation；test 只在冻结后描述性报告。窗口和通道不是
   独立重复，统计单位是 dataset-horizon-seed 配对。

当前上述新计划均为 `execution.authorized=false`，没有新的训练指标。已执行的
anchor x memory screen 是其 draft 计划下的事后 exploratory 运行，不改变该原则。
完整阶段
和证据边界见 [`research/mamba_paper_execution_protocol_20260903.md`](research/mamba_paper_execution_protocol_20260903.md)
与 [`research/mamba_current_status.md`](research/mamba_current_status.md)。

## 可复核命令

```bash
conda run --no-capture-output -n mamba python -m pytest -q
conda run --no-capture-output -n mamba python scripts/smoke_mamba_acf_forecast_relation.py
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_multidataset_plain_baseline_expansion_plan_20260903.json \
  --validate_only
conda run --no-capture-output -n mamba python scripts/analyze_mamba_acf_channel_lag_bank_screen.py \
  --plan docs/research/mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json
```

当前完整回归为 `828 passed, 45 warnings`；这只证明代码和实验合同，没有产生新
预测精度结果。lag-bank screen 是 validation-only：主指标来自
`checkpoints/<setting>/loss_logs.json` 的最佳 `val_loss`（validation scaled
MSE），因此其 `skip_test=true` 且 `stream_metrics=false`；后者只有执行独立
test 阶段才会生成。最后一条命令在当前未授权、未训练状态应返回 `NOT_RUN`
（退出码 `3`）。
