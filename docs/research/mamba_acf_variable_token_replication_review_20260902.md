# Variable-Token MambaACF Replication Review

**状态：** `exploratory_variable_token_replication_draft`，待人工复核，未授权训练。

**计划：** [`mamba_acf_variable_token_replication_plan_20260902.json`](./mamba_acf_variable_token_replication_plan_20260902.json)

## 审计结论

当前严格确认协议没有和 variable-token candidate 对齐，这是有意的边界，而不是漏登记。确认协议使用四个数据集、H96/H336/H720、50 epoch 和五个 seed；已完成的 variable screen 只有 ETTh1/Traffic、H96、三个 seed、八个 epoch。两者不应合并为一张结果表。

更重要的是当前实现的计算顺序：`MambaACF.forward` 先从共享 temporal hidden 生成 `repr_head` 表示并计算 AutoCon，再调用 `_forecast_normalized`。variable-token mixer 只在 forecast 分支产生修正。因此 `lambda=.03` 相对 `lambda=0` 的收益，最多说明共享表示辅助目标与变量-token预测分支组合后的增量；不能写成“relation loss 学到了 variable-token ACF 结构”。

已有 non-variable fair-baseline screen 仍有内部完整的旧 source snapshot，但它和当前源码树不同，且使用不同的宽度/预算合同。它可以作为历史 exploratory context，不能作为本计划的 current-tree paired control，也不能与 variable screen 的 delta 池化。

## 最小冻结矩阵

| Phase | 模型/设置 | 数据 | Horizon | Seeds | 作业数 |
|---|---|---|---:|---|---:|
| shared memory off | `MambaACF`, variable mixer off, ACF memory off | ETTh1, traffic | 96 | 2024-2028 | 10 |
| shared memory on | `MambaACF`, variable mixer off, train-only peak memory | ETTh1, traffic | 96 | 2024-2028 | 10 |
| variable memory off | `MambaACFVariable`, memory off | ETTh1, traffic | 96 | 2024-2028 | 10 |
| variable memory on | `MambaACFVariable`, train-only peak memory | ETTh1, traffic | 96 | 2024-2028 | 10 |
| relation null | `MambaACFVariableAutoCon`, lambda `0.0` | ETTh1, traffic | 96 | 2024-2028 | 10 |
| relation | `MambaACFVariableAutoCon`, lambda `0.03` | ETTh1, traffic | 96 | 2024-2028 | 10 |

总计 **60 个作业**。新 seed 集合不重复已完成的 2021-2023 screen；所有臂固定 `d_model=32/d_ff=64`、一层 Mamba、`seq_len=96`、batch 32、最多 8 epoch、patience 3、cosine scheduler、raw train-only ACF source。该预算是为了精确重放候选 screen，因而仍是 exploratory，不是 50-epoch confirmation。

六臂是最低可解释矩阵：shared/variable 两种 trunk 分离 variable branch，memory off/on 分离 ACF forecast memory，AutoCon `lambda=0`/`.03` 才能提供同一 executable AutoCon 架构内唯一的 relation 增量对照。外部 `Mamba2`、PatchTST 等不放进这个最小矩阵；它们只能在后续新计划中作为 current-tree、同预算的外部锚点。

## 统计门槛

- 推断单位是每个 dataset 内的五个 seed-level paired contrast；窗口和通道不是独立样本，也不跨数据集伪造 n。
- 主性能比较是 `variable_relation_h96` 对 `shared_memory_on_h96`，但它是组合性能比较，不是单独的 variable 或 relation 因果比较。
- 机制主比较是 `variable_relation_h96 - variable_relation_null_h96`。辅助比较是 variable memory on/off、variable-vs-shared memory-on、shared memory on/off。
- 每个数据集必须报告所有 seed 的 scaled MSE/MAE、raw delta、log-MSE ratio、95% paired interval、胜负数和 exact paired permutation/Wilcoxon sensitivity。主性能和 relation 两个 family 分别对两个数据集做 Holm adjustment。
- 进入下一阶段的实用门槛是：主比较和 relation 比较在两个数据集都达到平均 MSE ratio `<=0.995`、至少 `3/5` seed 胜，且任一数据集平均伤害不得超过 `1%`。这只是升级规则，不等于显著性或可发表性证明。
- 任一缺失/重复坐标、非有限指标、ACF sidecar/effective-lag 不一致、source hash 漂移或 artifact hash 变化都会使对应 cell 无效；不能在看结果后补 seed、改 horizon、重选 lambda 或替换失败作业。

## 人工复核重点

1. 接受“当前 relation loss 不读取 variable-token forecast branch”的解释边界，并禁止把 lambda gain 描述成 variable-token ACF 表示学习。
2. 确认 2024-2028 是独立于已完成 screen 的新 seed 集合，且计划输出根目录为空、不会复用旧 artifacts。
3. 审核 raw ACF sidecar、`train_peaks_available` 的有效 lag、`relation_min_lag=96` 和 relation candidate/effective lag 的逐坐标一致性。
4. 在任何授权前完成参数量、GPU smoke、streaming equivalence、环境和当前 source tree hash 审计；资源测量不计入 60 个训练作业。

## 结论边界

本文件只冻结一个可审计的短矩阵，不能把尚未运行的计划称为结果。即使 60 个作业全部通过，也只能支持“在两个 hourly 数据集、H96、该实现和该预算下的变量-token/ACF/Relation 机制复现”。要讨论 Mamba2/PatchTST 优势、长 horizon、速度或论文发表性，必须先通过本矩阵，再另行复核并冻结完整 confirmation protocol。
