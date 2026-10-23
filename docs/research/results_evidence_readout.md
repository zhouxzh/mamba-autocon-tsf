# Mamba / AutoCon 结果证据分级速查

**用途：** 为人工复核区分已验证的 ETT 统计、完整但后验的高维复核表和新架构 pilot。除明确标为 matched-seed 的统计外，这里的相对变化是各 cell 均值的 `(实验模型 MSE / 对照 MSE - 1) * 100%`；负数表示误差下降。它不产生新的训练结果，也不替代对应工件。

> **Provenance 更正（2026-09-03）：** [`../reports/confirmatory-M-fresh.md`](../reports/confirmatory-M-fresh.md) 的 ETTh1 M 表来自旧的或不同 provenance snapshot，与当前冻结的 [`corrected/confirmatory_M/report/summary.csv`](../../artifacts/paper_final/corrected/confirmatory_M/report/summary.csv) 不一致。涉及 corrected ETT-M 的当前引用必须以 CSV、对应 `phase_validation.json` 和 `statistics/paper_statistics.json` 为准；旧报告只保留历史诊断用途，不能混入本页的 current-tree 或 corrected 结论。

## 2026-09-01 最新增量

### 2026-09-02 当前代码 fair screen

当前代码的变量-token screen 已完成 30/30 作业，五个 phase validator 均为
`valid=true`，独立 sidecar replay 为 30/30 contract/metric pass、drift 0。
同一 `MambaACFVariableAutoCon` 架构的 relation null 对照为 `lambda=.03 -
lambda=0`：MSE 平均相对变化 `-0.979%`，5/6 配对点改善（ETTh1 2/3，Traffic
3/3），描述性 paired t `p=0.0848`。memory on/off 的 MSE 变化为 `-1.168%`
（6/6），rank-2 horizon calibration 相对 relation 仅 `-0.025%`（3/6）。
这些是 H96、每数据集 3 seed、8 epoch 的探索性结果。

与独立 42-job fair baseline screen 的描述性均值比较时，必须区分两组
provenance。该 fair screen 的 shared calibrated arm 为 ETTh1 `0.393062`、Traffic
`0.562918`，相对 PatchTST 分别高 `+3.207%`、`+1.329%`；这组数字不能标成
variable-token candidate。当前 30-job variable-token screen 的 calibrated arm 为
ETTh1 `0.385372`、Traffic `0.550932`，相对同一独立 PatchTST 均值分别为
`+1.188%`、`-0.829%`，也只是跨 provenance 的描述性上下文。因此当前证据不能支持
“所有数据集精度不低于基线”。variable calibrated candidate 的 synthetic median
latency 为 ETTh1 `1.557 ms`、Traffic `1.669 ms`；只有 Traffic 在该输入形状
下快于 PatchTST，不能外推为端到端速度优势。

当前核心状态页见 [`mamba_current_status.md`](mamba_current_status.md)，机制与数据流复核见 [`autocon_mamba_acf_core_review.md`](autocon_mamba_acf_core_review.md)；[`mamba_acf_core_status_20260901.md`](mamba_acf_core_status_20260901.md) 仅是历史快照。新增的 relation null control 在同一 MambaACF variable-token AutoCon、同一 train-only sidecar、同一 lag-pair sampler 和同一预算下比较 `lambda=0` 与 `0.03`，12/12 作业通过 phase validator，配对 MSE delta 为 `-0.732%`（n=6，p=`0.0454`），MAE delta 为 `-0.821%`（p=`0.0980`）。随后完成的三臂 channel-gate screen 为 18/18 成功：ordinary low-rank gate 相对 no-gate 的 MSE delta 为 `-0.452%`（Traffic `-0.752%`），ACF-conditioned gate 相对 ordinary gate 为 `+0.347%`。最新的 current-tree raw relation-target screen 也已完成 12/12：soft05 相对 hard 的 pooled MSE `+0.146%`，Traffic 为 `+0.279%`（0/3 seed 改善），按注册停止规则不扩展。每个数据集只有 3 个 seed；因此这些都是探索性机制证据，不是普适或确认性结论。

该 null control 和 gate screen 的 hard-selected/global 关系目标平均只覆盖约 `2.977%` 的 pair，active-anchor 为 `91.447%`；其中 `76.063%` 的 high-ACF candidate 排除率只适用于 `residual` source 的 null-control 工件。当前 raw target screen 的独立 telemetry 为 hard `78.47%`、soft05 `66.62%`，但预测误差未改善。新的 conditional high-ACF exclusion 与 signed sign diagnostics 已写入 `loss_logs.json`。null-control sidecar 隔离回放为 `12/12` contract pass、`12/12` metric match；gate screen replay 为 `18/18`，target screen replay 为 `12/12`，均只证明加载路径，不增加训练样本量。

最新的两个探索性筛选也已完成。low-rank head screen 为 `12/12` 成功、两个 phase validator 均通过：rank-16 相对 full head 的 MSE 为 `+1.095%`（`0/6` 更低），仅 head 参数从 `18,624` 降到 `6,336`，不支持精度/速度兼顾。relation-coverage screen 为 `18/18` 成功、三个 validator 均通过：balanced pairing 将两条 effective lag 的 exposure 调平，但 MSE 为 `+0.239%`；active-anchor normalization 的 MSE 为 `+0.013%`。两份读数都来自短预算 provenance snapshot，p 值为未校正描述性配对检验，不能作为确认性证据。

## 证据层级

| 层级 | 工件 | 可支持的结论 | 不能支持的结论 |
| --- | --- | --- | --- |
| 已验证但有限 | `artifacts/paper_final/corrected/confirmatory_{S,M}/statistics/paper_statistics.json` | ETT 上的 seed-paired AutoCon 消融方向与校正后统计 | 普适提升、正式高维主表或 SOTA |
| 完整高维后验复核 | [`highdim-confirmatory-M-v3.md`](../reports/highdim-confirmatory-M-v3.md) | Electricity 与 Traffic 上 Mamba 的数据集依赖性；`_v3` confirmatory 覆盖 270/270 | 原始 launch-time 正式确认或 publication-ready 主张；selection sidecar 为 `publication_ready=false` |
| 探索性筛选 | `artifacts/mamba_acf_pilot_20260831/` 与 `artifacts/mamba_acf_selector_replication_20260831/` | 新结构实现是否可运行、下一轮候选的优先级；selector 的 3-seed 配对方向 | ACF、AutoCon 或 Mamba 的稳定精度收益；selector 仍只有 Traffic H96、3 epochs |
| 评估合同回放 | [`mamba_acf_sidecar_replay_audit.md`](mamba_acf_sidecar_replay_audit.md) 与 [`standalone replay`](../../artifacts/exploratory/mamba_acf_sidecar_standalone_replay_20260831_audit/replay_report.md) | 当前 loader 能在 `is_training=0` 下加载历史 checkpoint/sidecar，并在冻结容差内重现 18 个历史指标 | 新的训练证据、current-tree 重训结果、统计显著性或发表结论 |
| 三臂 channel-gate screen | [`mamba_acf_channel_gate_screen_readout_20260901.md`](mamba_acf_channel_gate_screen_readout_20260901.md) 与 [`analysis.json`](../../artifacts/paper_final/mamba_acf_channel_gate_screen_20260901/analysis.json) | 普通低秩 gate 是 Traffic 的候选跨变量组件；ACF 条件化在当前 sampler/预算下没有增益 | 普适提升、严格速度优势、ACF gate 的确认性收益 |
| Gate 资源/回放 | [`channel-gate replay`](../../artifacts/exploratory/mamba_acf_channel_gate_replay_20260901/replay_report.md) 与 [`resource directory`](../../artifacts/exploratory/mamba_acf_channel_gate_resources_20260901/) | 18/18 评估重放一致；可量化参数量和 synthetic latency 代价 | 端到端训练速度或真实部署吞吐主张 |
| 低秩 head screen | [`mamba_acf_low_rank_head_screen_readout_20260901.md`](mamba_acf_low_rank_head_screen_readout_20260901.md) | rank-16 的 head 参数合同和 paired 退化读数；full-rank 应保留 | 压缩后的精度/速度优势；无 standalone replay |
| Relation-coverage screen | [`mamba_acf_relation_coverage_screen_readout_20260901.md`](mamba_acf_relation_coverage_screen_readout_20260901.md) | 实际 lag exposure、active-anchor scale 与 paired coverage 对照 | balanced/normalization 的精度或普适机制收益 |
| Relation-target screen | [`mamba_acf_relation_target_screen_readout_20260901.md`](mamba_acf_relation_target_screen_readout_20260901.md) 与 [`target replay`](../../artifacts/exploratory/mamba_acf_relation_target_screen_replay_20260901/replay_report.md) | current-tree raw sidecar 下 hard/soft05 target support、独立 high-ACF telemetry 和 12/12 加载回放 | soft target 的确认性精度收益、false-negative 因果证明或普适结论 |

`corrected` 的统计以 seed 为配对单位，450/450 任务通过验证。其 selection binding 仍是 `posthoc_verified`，因此即使有校正后的显著 cell，也应表述为已验证的受限消融证据，而不是预注册的普适主张。

此外，当前代码审计发现这些快照早于 legacy ACF mask、AutoCon projection
contract 和 CI channel-wise global relation 修正。plain Mamba 的预测行不受
这些辅助目标变化影响；所有 AutoCon 行必须按新协议重新选择和训练，旧表只
作历史诊断，不能直接作为当前方法的增益估计。

高维 `_v3` 的 selection/confirmatory validator 分别记录 `378/378` 与 `270/270` 最终去重的 `ok` 作业；confirmatory 的 280 条原始 manifest 尝试中有 10 条路径名过长失败，均由同一 job ID 的后续成功记录替代。其 `prelaunch_replay_of_posthoc_attestation` 绑定只证明 replay 和输入完整性，不会把 `posthoc_verified`、`publication_ready=false` 的 selection 追溯性地变成 launch-time confirmatory。

## 已验证的 ETT 消融

| phase | 比较 | MSE 改善 cell | 平均相对变化 | Holm/FDR 显著 cell | 可读结论 |
| --- | --- | ---: | ---: | ---: | --- |
| corrected S | Mamba1AutoCon - Mamba1 | 4/10 | +1.242% | 0/0 | 当前不支持 Mamba-1 上的收益 |
| corrected S | Mamba2AutoCon - Mamba2 | 8/10 | -4.996% | 0/0 | 方向积极，但 10-cell 校正后未显著 |
| corrected M | Mamba1AutoCon - Mamba1 | 8/10 | -0.769% | 0/0 | 小且异质的方向性收益 |
| corrected M | Mamba2AutoCon - Mamba2 | 7/10 | -2.381% | 1/1 | 当前最强的 AutoCon-on-Mamba 证据，但仍非普适结论 |

这里不能用 [`strict-evidence-status.md`](../reports/strict-evidence-status.md) 的旧汇总替代 `corrected` 结论：该页明确对应局部 logits 修正前的冻结 pilot，与 corrected 工件不可混表。

## 冻结高维读数（cell-mean ratio，仅作描述性聚合）

历史无 relation-loss 的 Mamba-2 encoder + compatible-head arm 相对于 PatchTST 的 MSE 变化如下：

| 数据集 | H96 | H336 | H720 | 读法 |
| --- | ---: | ---: | ---: | --- |
| Electricity | -0.715% | +1.695% | -4.455% | 接近或在两个 horizon 更低 |
| Traffic | +29.205% | +20.611% | +19.833% | 三个 horizon 均明显更差 |

同一历史表中，Mamba-2 加历史 AutoCon 的 cell-mean ratio 变化为 Electricity `-0.442%/-0.481%/-0.015%`，Traffic `+0.226%/-0.051%/-0.242%`。方向小且不一致，不能据此说 AutoCon 在高维数据上稳定提升 Mamba；论文统计应采用同一 dataset/horizon/seed 的配对估计，见核心复核页。

因此“Mamba 编码器是否适合”的答案是条件性的：它可作为高效时间编码器候选，但现有结果不支持带兼容预测头的这一路径作为所有多变量数据集的默认预测骨干。Traffic 的结果尤其要求单独处理跨变量和局部片段结构。

## 新 MambaACF pilot

对照是同预算的 `plain shared temporal MambaACF`（MSE `0.600947`）。该配置会记录 train-only ACF prior，但关闭 memory 与 relation objective 时 prior 不进入 forecast 分支，因此可作为 forecasting control。

| 变体 | MSE | 相对 plain | 证据判断 |
| --- | ---: | ---: | --- |
| MambaACF1 plain | 0.608056 | +1.183% | 此单次预算下 Mamba-1 不占优 |
| variable-token | 0.587315 | -2.268% | 值得进入正式消融 |
| causal ACF memory | 0.596415 | -0.754% | 值得复验，不足以主张收益 |
| ACF relation loss | 0.606501 | +0.924% | 当前配置未显示收益 |
| variable-token + relation | 0.590902 | -1.672% | 仍差于 variable-only `+0.611%` |
| patch-token（旧 all-candidate pilot） | 0.611913 | +1.825% | 旧配置不能解释为有效 contrastive learning |
| causal embedding + memory | 0.615312 | +2.391% | 该组合在 pilot 中变差 |

新增的可用性加权 memory 复核固定了同一模型、训练合同和 seed，仅将 lag 选择改为 train-only peak strength 乘合法 forecast-step fraction。解析出的 lag 为 `[96,120,72]`：

| seed | selector MSE | fixed-memory MSE | plain MSE | selector vs fixed | fixed vs plain |
|---:|---:|---:|---:|---:|---:|
| 2021 | 0.586256 | 0.596419 | 0.600948 | 1.704% | 0.754% |
| 2022 | 0.590390 | 0.607324 | 0.611094 | 2.788% | 0.617% |
| 2023 | 0.587594 | 0.599909 | 0.605031 | 2.053% | 0.847% |
| mean | 0.588080 | 0.601217 | 0.605691 | 2.182% | 0.739% |

固定 memory 相对 plain 有 `0.739%` 的平均改善，selector 在固定 memory 上再改善 `2.182%`。所有 seed 的方向一致，但 `n=3`、仅一个高维数据集和 H96，按正式协议的双侧 exact Wilcoxon 为 `p=0.25`；这是下一轮正式消融的候选证据，不是确认性显著结果。完整配置和日志在 [`artifacts/mamba_acf_selector_replication_20260831/`](../../artifacts/mamba_acf_selector_replication_20260831/)。

pilot 中 `ACFRelationLoss` 的有效关系覆盖约为 16%，这是先调 sampler/lag 设计和损失权重、再判断其价值的理由，而不是把一次负结果解释成 ACF 无用。这个数字需要拆开报告：未使用 selected-lag mask 时，约 `15.9%` 是 dense global 候选覆盖、active-anchor 约 `99.9%`；使用 selected-lag mask 后，候选覆盖约 `3.1%`、active-anchor 约 `92.6%`。local scope 只在窗口内部位置查距离，`T=96` 且 lags 为 `[96,120,72]` 时，lag 96/120 对 local loss 没有有效 pair，只有 lag 72 能贡献部分 pair。正式日志还应报告 local/global active pair 和每个 lag 的计数；当前贪心 sampler 的一次审计计数为 `[89,101,81]`，并不保证 lag 均衡。

最新 patch-token 修正 pilot 打开了 `hard_selected + mamba_acf_relation_lag_mask=true`，在 ETTh1/Traffic 的单 seed、validation-only、4 epoch 配对中，patch-on 相对 off 的 best validation loss 分别下降 `2.79%/3.24%`；global contrastive-active anchors 约 `87.0%/84.1%`，degenerate positives 约 `5.2%/8.2%`。该结果说明 patch 分支值得多 seed 复验，但训练 wall time 增加 `6.47%/13.57%`，synthetic prediction latency 增加 `40.49%/65.88%`，因此没有速度优势证据，也不能进入论文主表。完整读数见 [`mamba_patch_token_contrastive_pilot_readout_20260907.md`](mamba_patch_token_contrastive_pilot_readout_20260907.md)。

逐步 ACF memory gate 的独立重跑已通过 CUDA 输出合同和完整测试，Traffic/H96/seed2021 的 3-epoch MSE 为 `0.585388`。它是实现级和候选筛选证据；它没有增加 seed、数据集或 horizon 覆盖，故仍归入 exploratory pilot。

## 当前代码 v3 的公平复跑与资源边界

最近的实现级复跑固定 Traffic/H96、`d_model=32`、Mamba-2 `d_state=16/d_conv=4/expand=2`、3 epochs、seed 2021，并检查 gate 与静态 ACF memory 的单独作用。该批次仍是单 seed/短预算：

| 变体 | MSE | MAE | 相对同批 PatchTST MSE |
| --- | ---: | ---: | ---: |
| PatchTST | 0.565537 | 0.358836 | -- |
| MambaACF gate-only | 0.566947 | 0.384248 | +0.249% |
| MambaACF memory-only | **0.549255** | 0.378875 | -2.879% |
| MambaACF gate + memory | 0.552341 | 0.377871 | -2.333% |

memory-only 是这批的最低 MSE，但 gate + memory 没有超过 memory-only；不能把一次单 seed 结果写成 ACF 的确认性收益。日志在 [`extension_pilot_20260831/traffic_h96`](../../artifacts/extension_pilot_20260831/traffic_h96/)。它也不能与上面的三 seed selector 表或旧的 `MambaACF` `.539017` 工件混合，后者使用了不同的代码/资源批次。

同一硬件上的 device-resident synthetic 资源读数（Traffic/C862/H96/B8）为：plain Mamba-2 `0.7208 ms`、PatchTST `1.0537 ms`、MambaACF memory-only `1.3871 ms`、gate + memory `1.6655 ms`；峰值增量分别为 `23.10/133.90/25.86/25.86 MB`。静态 cache 的价值目前是降低 memory 临时张量和显存压力，不是降低延迟；gate 还增加约 113 个 ACF adapter 参数和额外计算。MambaACF 的计时使用 20/100 次 warmup/timed，两个旧基线使用 5/20，正式比较前需统一协议。原始记录在 [`resource_extension`](../../artifacts/resource_extension/)。

资源 JSON 的参数计数是在 dead-parameter 修正前生成的历史快照。当前 shared
plain/no-memory、memory-only、gate+memory 的 active trainable count 约为
`145,387/160,760/183,294`；旧记录仍保留以保证可追溯性，不能与修正后的计数
混用。预测 pilot 的数值不因这项冻结而改变，但论文前必须用统一版本重跑参数、
显存和吞吐。

历史扩展 pilot 还存在 plain `weight_decay=0`、AutoCon 默认 `1e-5` 以及不同 scales 的配置差异。
当前 optimizer 已过滤冻结参数，但这些旧结果仍不能作为严格的正则化、容量或 ACF 归因对照。

## 直接决策

1. 正式实验优先保留 `plain`、`variable-token`、`train_peaks_available causal ACF memory`；patch-token 作为已修正 relation 合同下的结构候选，先做小规模多 seed 复验；relation loss 仍作为需要覆盖率消融的次级候选。
2. 不能把 ACF 高峰直接写成预测收益。训练集 ACF 与 seasonal fallback 只说明可利用的历史结构；其收益仍会受可用 horizon、相位变化和跨变量状态影响。
3. 进入论文主张前，至少需要在 Electricity 与 Traffic 的固定 validation gate 下，对上述变体做多 seed 配对、Holm/FDR、端到端训练时间和峰值显存比较。

相关原始材料：[`mamba_current_status.md`](./mamba_current_status.md)、[`autocon_mamba_acf_core_review.md`](./autocon_mamba_acf_core_review.md)、[`corrected-evidence-status.md`](../reports/corrected-evidence-status.md)。
