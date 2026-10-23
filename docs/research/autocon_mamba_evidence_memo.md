# AutoCon 与 Mamba 历史源快照/证据审计备忘录

**审计日期：** 2026-08-31  
**用途：** 人工复核 2026-08-31 源快照和冻结结果的证据边界。本文不修改、也不重新解释冻结工件；下文精确源码行号仅对应审计日快照，不能当作 current-tree 行号。当前实现应以 [`autocon_mamba_acf_core_review.md`](autocon_mamba_acf_core_review.md) 和 [`mamba_current_status.md`](mamba_current_status.md) 为准。

## 结论先行

1. **Mamba 编码器不是普适的时间序列方案。** 历史表中的 `Mamba1/Mamba2` 都是无 relation loss 的 Mamba encoder + AutoConNet-compatible forecasting head；结果支持一个有条件的判断：该编码器路径在 Electricity 这类高维、平滑数据上有竞争力，在 Traffic 上明显落后于 PatchTST。因此不能写成“Mamba 普遍优于 Transformer/卷积模型”，也不能仅凭当前结果声称 Mamba 更快。
2. **历史 Mamba+AutoCon 不是“纯 Mamba 预测器”。** 历史模型只替换了 AutoConNet 的特征提取器，预测仍由 AutoConNet 的多尺度趋势分解和线性季节外推完成。它可作为“更换编码器后的 AutoCon 消融”，不能作为纯 Mamba 架构证据。
3. **AutoCon 的收益是异质的，而不是稳定增益。** 修正后的 ETT-M 结果中，Mamba-1+AutoCon 在 8/10 个 MSE 单元改善，平均相对变化 -0.77%；Mamba-2+AutoCon 在 7/10 个单元改善，平均 -2.38%，但多重校正后各只有极少数单元显著，不能推出普遍效果。
4. **当前新 MambaACF 仍是 pilot 证据。** 新增 Traffic/H96、3-seed、3-epoch 的 fixed memory 相对 plain 平均降低 MSE 0.739%，而可用性加权 selector 在 fixed memory 上再降低 2.182%（每个 seed 方向一致）；按正式协议的双侧 exact Wilcoxon 为 `p=0.25`，且只覆盖一个数据集和一个 horizon；它是下一轮正式消融的候选，不是确认性结论。
5. **旧的 18-run variable/memory/relation screen 还没有可独立加载的 ACF sidecar。** 这些工件是在 sidecar 契约补齐前生成的，因而只能作为运行内 train-only prior 的方向性读数；论文级 replay 必须重新生成完整 `acf_prior.npz`、哈希绑定和 resolved manifest。随后生成的 18 个 provenance-snapshot 作业带 sidecar，但其指标仍来自训练进程内的 prior 拟合，不是后来独立加载 sidecar 的评估；当前已完成独立 `is_training=0` 回放，结果见 [`mamba_acf_sidecar_replay_audit.md`](mamba_acf_sidecar_replay_audit.md)，它只验证评估合同，不增加训练效果证据。每单元只有 3 个 seed、global relation coverage 约 3%，不能替代确认性矩阵。

## 代码审计

### 历史 AutoConNet

[`models/AutoConNet.py:59`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/AutoConNet.py:59) 使用 `DataEmbedding`，随后是 `DilatedConvEncoder`；[`models/AutoConNet.py:68`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/AutoConNet.py:68) 到 [`models/AutoConNet.py:73`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/AutoConNet.py:73) 定义多尺度趋势头、时间长度映射和线性季节头。预测路径在 [`models/AutoConNet.py:104`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/AutoConNet.py:104) 到 [`models/AutoConNet.py:131`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/AutoConNet.py:131)。

历史 AutoCon 目标在 [`layers/losses.py:297`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/layers/losses.py:297) 定义：局部/全局候选通过 ACF 排名选取 top-1 正样本。旧训练器先对训练 split 做分解，再通过 `statsmodels.acf` 计算 ACF；默认 source 是 residual，见 [`exp/exp_long_term_forecasting_with_AutoCon.py:41`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/exp/exp_long_term_forecasting_with_AutoCon.py:41)。旧路径随后将 ACF 取绝对值并在多变量情况下求均值，见 [`exp/exp_long_term_forecasting_with_AutoCon.py:135`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/exp/exp_long_term_forecasting_with_AutoCon.py:135)。

这条路径有三个需要在论文中披露的限制：

- ACF 符号被丢弃，正相关和反相周期不能区分；
- 距离超出 ACF 表时会 clamp 到最后一个值，远距离关系可能被错误赋值；
- 随机 mini-batch 未必包含有用的时间间隔，导致全局关系目标稀疏。

当前工作树已对这两个 legacy 类做了最小边界修正：身份样本由距离矩阵的对角位置定义，表外距离不再 clamp 到末端，而是从候选分母和正样本集合同时移除。该修正不改 `ACFRelationLoss` 主体。历史日志仍代表旧代码口径；标准完整 ACF 表且不存在非零 lag 的精确 1 时，主体数值通常不会变化，但包含表外距离、常数/完美周期序列或 AutoConCI 的结果应重新训练，不能直接和旧结果合并。

### 历史 Mamba1/2AutoCon

[`models/Mamba2AutoCon.py:167`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba2AutoCon.py:167) 到 [`models/Mamba2AutoCon.py:187`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba2AutoCon.py:187) 将 `DataEmbedding` 后接 Mamba-2 residual blocks；但 [`models/Mamba2AutoCon.py:198`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba2AutoCon.py:198) 到 [`models/Mamba2AutoCon.py:206`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba2AutoCon.py:206) 明确保留 AutoConNet 的趋势/季节预测头，且 [`models/Mamba2AutoCon.py:247`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba2AutoCon.py:247) 到 [`models/Mamba2AutoCon.py:299`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba2AutoCon.py:299) 仍按原 AutoConNet 方式生成预测。Mamba-1 只是复用这一共享头并替换 block，见 [`models/Mamba1AutoCon.py:22`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/Mamba1AutoCon.py:22)。

因此历史结果回答的是“把 Mamba 编码器插入 AutoConNet-compatible forecasting path 后会怎样”，不是“纯 Mamba 是否适合 forecasting”。

### 新 MambaACF

[`models/MambaACF.py:157`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:157) 定义了独立模型：共享时间 Mamba、多尺度分支、直接 horizon projection、线性 residual，以及可选变量 token 分支。它仍含 embedding 和线性 skip，所以更准确的称呼是 **Mamba-centered direct forecaster**，而不是只由 Mamba 算子组成的模型。表示学习状态经过独立 `repr_head`，见 [`models/MambaACF.py:412`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:412)，避免对 forecasting state 直接施加对比目标。

## ACF 如何被更有效地使用

当前实现已经形成一条比旧 AutoCon 更可审计的路径：

1. **只用训练 split 拟合 signed ACF。** [`utils/acf_guidance.py:286`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/utils/acf_guidance.py:286) 使用 FFT、分块通道计算和可选去趋势；分块降低拟合阶段的临时峰值，forecast forward 使用峰值选择和 compact prior 路径，sidecar 另保存完整测量表供 relation 合同复核，且避免用验证/测试值选周期。
2. **把 ACF 变成关系目标，而不是硬编码预测值。** [`layers/losses.py:7`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/layers/losses.py:7) 的 `ACFRelationLoss` 在查表后再做通道归约，利用 `abs(cosine)` 拟合强度、额外 hinge 项拟合符号，并按有效距离 mask；身份关系由对角位置决定，而不是由 ACF 值是否等于 1 决定。
3. **让 batch 真正包含目标 lag。** [`data_provider/samplers.py:20`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/data_provider/samplers.py:20) 的 `ACFLagPairBatchSampler` 将相隔选定 lag 的窗口成对放入 batch，减少关系目标在随机 batch 中为空的情况。训练器记录 valid fraction 和 active-anchor fraction，见 [`exp/exp_long_term_forecasting_with_AutoCon.py:298`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/exp/exp_long_term_forecasting_with_AutoCon.py:298)。
4. **把 ACF 用作有可靠性标记的 causal memory。** [`models/MambaACF.py:651`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:651) 只读取历史中确实存在的 lag，并返回 validity mask；[`models/MambaACF.py:755`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:755) 用 signed ACF 的符号、幅值和可用比例控制 residual，而不是把不可用长 lag 循环复制到未来。

这条设计仍需实验验证。ACF 是统计先验，不等于未来可预测性；固定 lag 可能不适合所有数据集，且长 horizon 会出现历史窗口无法提供的 lag。当前新增的 `train_peaks_available` 只从 train-only 峰值中保留合法候选，并按可用 forecast-step 比例排序；它避免了 H96 中把完全不可用的 lag 336 纳入 memory。Traffic/H96 的 3-seed 结果见 [`artifacts/mamba_acf_selector_replication_20260831/`](../../artifacts/mamba_acf_selector_replication_20260831/)，但仍需多数据集、多 horizon 验证。因此论文应把“ACF 提供结构化先验”作为假设，不能把 ACF 峰值本身写成精度保证。

审计中曾发现先验来源不一致的风险：普通 `MambaACF` 与 `MambaACFAutoCon` 可能分别使用 raw/residual 信号，且 memory 读取信号未必与 prior 相同。当前实现已让一次运行内的 prior、relation 和 causal memory 遵循同一 `AutoCon_acf_source`；探索性配置仍默认 `residual`，而严格确认方案固定为 `raw`。原因是现有 prior 在完整 train 序列上做分解，推理 memory 却在每个 encoder window 内独立分解，residual/trend 在窗口边界并不数值相同；`_memory_signal` 保证的是信号族一致，不是这种边界差异自动消失。正式实验必须把 source 写入配置并保持 plain/AutoCon 一致，严格方案还必须使用 raw，除非先实现并复核窗口一致的分解。

## 已有结果的可用性

### 可作为描述性/配对证据

修正后的 ETT-M 阶段有 `450/450` 个有效作业；统计以 seed 为配对单位，并进行了 Holm 和 BH-FDR 校正，见 [`confirmatory_M/statistics/paper_statistics.md`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/corrected/confirmatory_M/statistics/paper_statistics.md)。该阶段同样使用 `posthoc_verified`、`publication_ready=false` 的选择 provenance，只能作为受限、后验的配对证据。其 MSE 配对结果为：

| 比较 | 改善单元 | 平均相对变化 | 校正后结论 |
| --- | ---: | ---: | --- |
| Mamba1AutoCon - Mamba1 | 8/10 | -0.77% | Holm/FDR 均无显著单元 |
| Mamba2AutoCon - Mamba2 | 7/10 | -2.38% | Holm/FDR 各 1 个单元通过；不能外推为普遍增益 |

原始配对差可在 [`confirmatory_M/raw_report/mamba_paired_ablation.csv`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/corrected/confirmatory_M/raw_report/mamba_paired_ablation.csv) 复核。

高维阶段的结果也已完整覆盖 `270/270`（Electricity 321 通道、Traffic 862 通道、5 seeds、H=96/336/720），见 [`highdim_confirmatory_M_v3/phase_validation.json`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/corrected/highdim_confirmatory_M_v3/phase_validation.json)。代表性的 train-standardized MSE 如下：

| 数据集/预测长度 | PatchTST | Mamba2 | Mamba2AutoCon |
| --- | ---: | ---: | ---: |
| Electricity / 96 | 0.184448 | 0.183130 | 0.182321 |
| Electricity / 336 | 0.207953 | 0.211477 | 0.210460 |
| Electricity / 720 | 0.252428 | 0.241182 | 0.241146 |
| Traffic / 96 | 0.501767 | 0.648306 | 0.649773 |
| Traffic / 336 | 0.516942 | 0.623489 | 0.623169 |
| Traffic / 720 | 0.550696 | 0.659914 | 0.658317 |

这些数字支持“数据集依赖性”：Mamba-2 在 Electricity 的两个 horizon 胜过 PatchTST，在 Traffic 三个 horizon 分别高约 29.2%、20.6%、19.8%。它们不能单独支持 SOTA 或通用优势。

### 只能作为探索性或实现 gate

- 新 `MambaACF` Traffic pilot：plain MSE `0.600947`，channel gate + causal embedding MSE `0.602418`，旧组合各仅一个 seed、3 epochs；结果见 [`artifacts/mamba_acf_pilot_20260831/results.txt`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/mamba_acf_pilot_20260831/results.txt)。新增 selector 复核为 3 seed 平均 `0.588080`，fixed memory `0.601217`，plain `0.605691`，但仍不能与 50-epoch、不同宽度的冻结基线直接作效果结论。
- 旧 high-dimensional smoke、synthetic resource smoke：只能证明数据/设备/工件链路可运行。
- `strict` 阶段：属于修正前实现，不能和 `corrected` 阶段合并。

### 不能写成确认性结论的原因

高维确认性结果虽然覆盖完整，但其 provenance 明确写的是 `prelaunch_replay_of_posthoc_attestation`，selection sidecar 为 `posthoc_verified` 且 `publication_ready=false`，见 [`highdim_confirmatory_M_v3/provenance.json`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/corrected/highdim_confirmatory_M_v3/provenance.json)。这证明了复现绑定一致，不等同于原始 launch-time 预注册选择。论文必须如实标注这一限制。

## 速度结论的边界

现有资源审计固定 batch=32、25 次 warm-up、100 次 GPU 计时，且只测 device-resident synthetic forward，协议见 [`resource_benchmark/README.md`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/artifacts/paper_final/corrected/resource_benchmark/README.md)。ETTh1、H=96 的单次延迟约为：AutoConNet `0.260 ms`、PatchTST `0.292 ms`、Mamba1 `0.370 ms`、Mamba2 `0.664 ms`。所以当前证据不能写“Mamba 更快”；若论文主张效率，必须补充同一硬件上的端到端训练时间、数据加载、显存和有效预测吞吐，并使用同一模型容量/epoch 协议。

## 建议的论文定位与下一步

最稳妥的研究问题是：**在跨变量混合受控、且训练集 ACF 存在稳定周期结构时，Mamba 编码器配合 signed-ACF 关系学习和 causal lag residual 是否能以较低资源达到与强基线相当的预测误差。**

进入正式论文主表前至少需要：

- 对 MambaACF plain、ACF relation、causal memory、variable mixer 做预先固定的逐项消融；
- 在 Electricity、Traffic 及 ETT 上用相同 seed、epoch、宽度和选择协议比较 DLinear、PatchTST、TimesNet、Mamba1、Mamba2；
- 报告每个 dataset/horizon 的 paired seed delta、置信区间、Holm/FDR，以及通道级分布；
- 单独测量参数量、峰值显存、训练时间和端到端吞吐；
- 将所有新模型选择放在独立 validation gate，避免把 pilot 或 post-hoc 选择写成确认性结果。

**审计判断：** 当前仓库足以支撑一份诚实的架构/消融研究报告和下一轮正式实验设计；还不足以支撑“纯 Mamba 普遍适合时间序列”“ACF 稳定提升精度”或“Mamba 更快”的论文主张。
