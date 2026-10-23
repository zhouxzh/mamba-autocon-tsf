# Mamba / AutoCon 当前状态

当前总体发表准备度见 [`2026-10-01 update`](publication_readiness_update_20261001.md)；[`2026-09-30 audit`](publication_readiness_audit_20260930.md) 保留为历史审计。两者均区分已冻结 held-out 证据、validation-only 补充和未完成门槛，不将局部正结果提升为普适结论。

## 2026-10-03 Expansion release: four traffic/weather domains now have frozen evidence

在前述 residual-target、exact-head、forecast-affine ACF 预训练合同冻结后，新增 METR-LA 与 PEMS-BAY 两个目标域。源域只由 target few-shot prefix 的 ACF signature 选择，目标 primary 只由 few-shot 内部 calibration tail 选择；两域均选 NLinear + uniform。10% validation 三 seed 均值分别为 METR-LA `1.483978 -> 1.421429`（`-4.215%`）和 PEMS-BAY `2.923039 -> 2.632410`（`-9.943%`），均为 `3/3` 胜。复用冻结 source checkpoint 的 25% validation 仍为 `-2.755%` 与 `-10.679%`，均 `3/3` 胜。

随后打开正式扩展 held-out：METR-LA Hybrid 相对 NLinear `-3.814%`、相对 PatchTST `-5.964%`、相对 iTransformer `-10.596%`；PEMS-BAY 分别为 `-9.041%`、`-3.889%`、`-9.312%`，所有 paired comparison 均 `3/3`。六种 stress 条件在两个域均维持 `3/3` paired wins。严格审计见 [`expansion readout`](mambatsv3_residual_autocon_expansion_20261003.md) 和 [`expansion audit`](../../artifacts/paper_final/mambatsv3_residual_autocon_expansion_confirmatory_20261003/audit.json)。这将工作从两个域的 validation candidate 推进为四域、两种数据比例和强基线的可复核结果，但仍不支持所有 horizon 或所有数据集上的普遍领先主张。

部署验签仍显示 inference 不返回 AutoCon 表示、不计算 relation loss，且 `mamba_scan_count=1`；METR-LA/PEMS-BAY batch-8 median 分别为 `.328/.382 ms`。当前最重要的剩余一区门槛是 H192/H336 horizon transfer、full-data reference、扩展域 active/null 机制消融，以及统一的外部强基线协议；不能仅凭本节结果直接宣称一区录用级别。

扩展域 active/null 机制消融现已完成且为负：仅把 source forecast-affine ACF term 从 calibrated active 改为零，其余 source、seed、残差目标、exact head 和 target fine-tuning 全部保持一致。METR-LA active 相对 null `+0.282%`、PEMS-BAY `+1.036%`，两域都是 `0/3` active wins。故上述四域的正向 low-data/robustness/held-out 结果必须归因于 predictive residual pretraining + exact-head transfer，不能归因于当前 AutoCon affine loss；该 loss 停止扩展、作为负消融保留。一区方向应重新设计真正有正负样本判别力的 ACF contrastive pretraining objective，并要求先过 active/null gate 后才重新打开 held-out。完整数字见 [`expansion readout`](mambatsv3_residual_autocon_expansion_20261003.md)。

新的 forecast-level signed InfoNCE 机制筛选已实现。它在 train-only ACF peak lag 配对的预测未来段上使用显式批内负样本，负相关对在分子与分母中统一做符号翻转；同时修复 source subsampler 排序截断可能丢失配对成员的问题。pair-preserving active/null validation 结果在 PEMS-BAY 为 `-0.229%`、METR-LA 为 `-0.080%`，均 `3/3` active wins，active anchor coverage 分别约 `94.8--96.9%` 与 `91.1--95.8%`。这说明新目标有小但可重复的跨两域机制信号，但尚未达到重新打开 held-out 的强度；必须先在非交通目标域 replication。旧 affine loss 不得与此新结果混写。完整边界见 [`ACF contrastive repair screen`](mambatsv3_acf_contrastive_repair_screen_20261003.md)。

非交通 Weather replication 现已完成：H96 10%/25% active/null 分别 `-0.081%/-0.094%`，均 `2/3` active wins；在 late shift、missing channel 和 noise 的封闭 validation 条件下均不劣。新目标同时完成 PEMS-BAY H192/H336 三 seed extension，分别 `-1.395%/-1.956%`、均 `3/3` active wins。新 checkpoint 的 H96 和 H336 prediction-only 验签仍为一个 Mamba scan、无 relation loss、无 AutoCon representation 返回。这个机制证据比旧 affine loss 明显更完整，但仍只覆盖 active/null validation，且 Weather 效应较小；尚不可开启用于稿件主结论的 held-out release，下一门槛应是冻结的 target-primary vs active-H192/H336 比较及外部强基线。

该 H192/H336 门槛现已完成并进入 PEMS-BAY held-out release。冻结的 active H192/H336 三 seed 分别相对 NLinear `-10.172%/-8.449%`，相对 PatchTST `-3.391%/-3.380%`，相对 iTransformer `-8.459%/-11.739%`，所有比较 `3/3` 胜；PatchTST paired one-SE upper ratios `.9690/.9695`。两 horizon 的六类封闭 validation stress 对 active/null 均为 `3/3` 正向。严格读取器还验证 source selector、exact-head、zero-test validation artifact、test-reader `training_performed=false` 和所有 GPU guard event。见 [`ACF contrastive repair screen`](mambatsv3_acf_contrastive_repair_screen_20261003.md) 与 [`H192/H336 audit`](../../artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/readout.md)。

同时，Electricity 的 few-shot source selector（Traffic/METR-LA/Weather）、DLinear primary 和三 seed active/null 验证为 `+0.316%`、`0/3` active wins，active anchors 仅 `39--49%`。这必须保留为负转移边界：当前 InfoNCE 不是跨所有域的普适方案，论文应定位为 ACF-compatible traffic/periodic low-data transfer，而不是宣称所有多变量时间序列均改善。

Traffic H96 亦已完成 InfoNCE active/null：10% 为 `-0.127%`、`2/3` active wins，late shift 为 `-0.144%`、`3/3`，其余 stress 非劣但只有 `2/3`；25% 则变为 `+0.216%`、`0/3`。Traffic 只能作为低样本边界支持，不能参与当前正向 held-out release。与之相对，PEMS-BAY H192/H336 在相同冻结 source checkpoint 的 25% validation 中继续为 `-1.339%/-1.949%`、均 `3/3` active wins。这形成了清晰的作用域：方法在 PEMS-BAY periodic traffic 的长预测和少样本下强，跨 Electricity 和更高 Traffic data coverage 不保证改善。

METR-LA H192 独立 long-horizon release 现已通过严格审计。active H192 相对 NLinear/PatchTST/iTransformer/TimesNet 为 `-4.209%/-5.634%/-8.660%/-9.736%`，全为 `3/3`，upper ratios `.9607/.9475/.9164/.9057`；六类 closed stress 对 exact null 全为 `3/3` 正向。25% H192 active/null 继续为 `-0.554%`、`3/3`。这加上 PEMS-BAY H192/H336 提供两个独立交通域、三个长预测坐标的 held-out/validation 证据。TimesNet 在 PEMS-BAY H192/H336 的后加描述性 test extension 也落后 `7.613%/10.509%`、均 3/3；固定 batch-8 forward 下 active `.384/.402 ms`、TimesNet `.665/.649 ms`，候选低约 `42.3%/38.1%`，并保持单 Mamba scan、无 relation loss。TimesNet PEMS-BAY 不是原冻结 release 成员，稿件必须标注为 post-release extension。

PEMS-BAY 的后加 TimeMixer extension 已完成：validation/test active 相对 TimeMixer H192/H336 为 `-5.074%/-6.264%` 与 `-4.145%/-5.463%`，均 3/3。device-resident batch-8 H192/H336 TimeMixer `.516/.520 ms`，相对 active `.384/.402 ms` 高约 `34.3%/29.2%`；这是固定 shape 前向边界。投稿前的核心结果和所有负边界已浓缩到 [`ACF-InfoNCE core evidence`](mambatsv3_acf_infonce_core_evidence_20261003.md)。

Weather H192 已完成第三个独立 held-out release：active 相对 NLinear/PatchTST/iTransformer/TimesNet 为 `-3.189%/-5.338%/-10.926%/-6.283%`，全为 3/3；六类 closed stress 相对 exact null 也均 3/3 正向。其 25% active/null 为 `-0.108%`，但仅 1/3 win，因此 Weather 只能支撑 held-out accuracy 与低数据 robustness，不应被用于稳定的 25% 机制泛化主张。三域核心表与这一边界已更新到 [`ACF-InfoNCE core evidence`](mambatsv3_acf_infonce_core_evidence_20261003.md)。

PEMS-BAY H192/H336 后加 RLinear validation/test extension 已完成：active 相对 RLinear 为 `-14.273%/-13.046%` 与 `-13.461%/-12.593%`，均 3/3。固定 batch-8 前向中 RLinear `.065/.067 ms`，明显快于 active `.384/.402 ms`。因此候选的速度主张严格限于实测的非线性 TimesNet/TimeMixer 对照，不能写成相对线性模型的通用速度优势；RLinear 同样属于 post-release descriptive extension，不计入原冻结显著性主表。

对 Electricity 负转移的 sampler 覆盖解释已作一次隔离复验：round-robin policy 提升 active ACF anchor coverage 至 `51--57%`，但 active/null 更差 `+0.548%`、0/3（旧 legacy policy 为 `+0.316%`、0/3）。round-robin 仅保留为 `--relation-pair-policy round_robin` 实验选项，默认回到历史 `legacy` 合同，避免影响既有 audited release。结论是 Electricity 失败不是简单的 lag-pair coverage 问题；停止该域的后验调参并保留为强负边界。

PEMS-BAY H192 另补 full-target-prefix validation reference：使用 100% target train prefix、固定 128 sampled windows（不是 exhaustive full-data convergence），active 相对 NLinear/PatchTST/iTransformer/TimesNet 为 `-13.085%/-5.836%/-7.827%/-4.571%`，均 3/3。该结果说明 PEMS-BAY 的收益不只在少量前缀出现，但不可把此窗口采样协议写作一般的 full-data SOTA；完整限制已写入 [`ACF-InfoNCE core evidence`](mambatsv3_acf_infonce_core_evidence_20261003.md)。

## 2026-10-02 Low-data transfer: multi-source gain, AutoCon mechanism not established

新完成的 low-data transfer 探索保留现有单扫描 MambaTSv3 主干，只平均 PEMS03/04/07/08 源模型中通道数无关且 shape 相同的 core 参数；通道相关项与预测头均重新初始化。10% Traffic 开发条件从随机初始化 `.895753` 到 `.880226`（`-1.733%`），独立的 10% Weather 外部验证从 `4.469334` 到 `4.417528`（`-1.159%`），且未读取目标 test split。这保留为 validation-only 的低数据迁移基线，不能升级为 held-out 主结果。

关键机制消融固定四个源、H96/d24、源 seed、结构、平均规则及 Traffic target run，仅把源端 `AutoCon_lambda` 从 `0` 改为 `.001`。lambda-zero/active 为 `.900750/.900515`，仅 `-0.026%`。这个信号不够支持 AutoCon 因果贡献，故不得将多源迁移增益写为 AutoCon 效果，也停止当前 auxiliary-loss 方向的目标特化调参。完整边界与工件见 [`low-data core-transfer readout`](mambatsv3_core_transfer_lowdata_20261002.md)。

同日完成了更严格的八源 leave-Traffic-out 自监督重构。新实现将生产 MambaTSv3 的 patch embedding、Mamba encoder、final norm 与 AutoCon representation 直接作为共享预训练 core；它还实现了通道置换等变的 `mamba_ts_channel_agnostic_scan`，在同一个 Mamba block 上按通道 batch 化 patch sequence，并绕过通道 embedding/context。所有 source ACF 正负关系都只来自训练前缀，active global relation anchors 为 `74.2%--77.7%`，故不是空目标。结果仍明确失败：通道无关 topology 的 initial/AutoCon-pretrained Traffic 10% validation MSE 为 `.907492/.911449`（`+0.436%`）。masked reconstruction 和纯关系的其它三个已注册分支也均未越过初始核心。禁止 Weather gate、25%/鲁棒性/速度/held-out 扩展，也不能写成可迁移 AutoCon pretraining。完整停止记录见 [`shared-core pretraining stop`](mambatsv3_shared_core_pretraining_stop_20261002.md)。

随后对“预测对齐”作了独立修正：生产-core next-patch 分支使用独立的 `predictive_weight`，不再受 masked-reconstruction 权重影响；关系视图依次对最终预测 state、每通道最终 state、每通道 train-prefix peak mask 与物理 window origin 配对进行验证。修正后的三个八源 Traffic 10% 对照分别为 `.903894` vs `.895390`（`+.949%`）、`.912186` vs `.904790`（`+.817%`）、`.904734` vs `.900674`（`+.451%`）。最后一个分支还验证了真实物理 lag 配对和 active-anchor 归一化，仍未改善。故当前 ACF 正负 pair 自监督路线整体停止；不能将其列为 SCI-Q1 论文的新方法或开展后续鲁棒性/held-out。边界与无效早期 predictive 工件见同一 [`shared-core pretraining stop`](mambatsv3_shared_core_pretraining_stop_20261002.md)。

2026-10-03 的后续结构修复保留了预测对齐与直接 forecast-patch ACF pair，但将 source `P*d -> H` 预测头保存并只在 horizon 完全相同的 target 上精确恢复，禁止此前失败的 H24-to-H96 平铺。该 `exact-head` 机制在一个 Traffic 开发条件和一个 leave-Weather-out 独立门槛均改善 10% low-data validation（`.893987 -> .880236`，`4.447756 -> 4.302628`），并在 25% 条件继续改善。单 seed 的 validation 压力读数也在两个目标的后半验证段、10%/30% channel masking 和 5%/10% 噪声下保持优势。prediction-only GPU 测量显式 `return_repr=false`、无 relation loss、一个 Mamba block。该证据仍只有每域一颗 seed、validation-only，当前不能作为投稿结论；第二颗 fresh paired seed 与强基线矩阵是下一门槛。完整人工复核见 [`exact-head pretraining validation`](mambatsv3_exact_head_pretraining_validation_20261003.md)。

更新后的三 seed 与强基线验证保留了 exact-head predictive transfer 的低数据信号，但同时限定了结论。Traffic/Weather 三 seed 相对 new Mamba target head 平均为 `-0.916%/-2.259%`；然而 AutoCon lambda-zero 控制与 active 的差异仅 `+.015%/-.002%`。将 ACF forecast-pair 的加权份额校准到每 batch 预测损失的约 `8.3--9.9%`、并保持 `93.8--96.3%` pair coverage 后，Traffic仅 `-.014%`，Weather退化 `+.150%`。所以当前收益不能归因于 AutoCon，而是通道无关 core 加 exact horizon-matched predictive head transfer。DLinear-primary hybrid 在 Traffic相对DLinear三 seed全胜（均值 `-.290%`），但在Weather落后DLinear/NLinear `+.424%/+1.184%`；它不能形成跨域强基线主张。现阶段禁止使用该路线的 held-out 结果或写成一区论文创新，完整审计见 [`exact-head pretraining validation`](mambatsv3_exact_head_pretraining_validation_20261003.md)。

最后的 source-selection/primary-selection 审计进一步收紧了边界。只用 target few-shot train prefix 的 ACF signature 选择 top-3 sources、再用 inner calibration tail 选择 DLinear (Traffic) 或 NLinear (Weather)，在一个全新 end-to-end seed 上只有 `-.072%/-.016%` 改善，Mamba residual gates 接近零。由于物质效应消失，停止 source-count、gate、25% selected-source、stress和 held-out扩展；它不能成为 SCI-Q1 机制结果。详细结论见 [`exact-head pretraining validation`](mambatsv3_exact_head_pretraining_validation_20261003.md)。

后续修正将 source head 改为预测最后观测值的残差，使其与 target hybrid 实际部署的 Mamba correction 完全一致；并把 signed affine ACF trajectory loss 校准为 source next-segment loss 的约10%。此前用完整 target train prefix 选 source 的工件被降为 audit-only；替代选择器仅使用各 target 的 declared few-shot prefix。训练内 calibration tail 稳定选择 Traffic 的 DLinear 和 Weather 的 NLinear，target windows均选择uniform。三颗完全独立的验证 seed 在 Traffic/Weather H96 10% 均胜所选 primary：平均 `.806555 -> .794837` (`-1.453%`, 3/3) 与 `4.288621 -> 4.207431` (`-1.893%`, 3/3)，paired upper ratios `.9866/.9832`。一颗 fresh stress pair 在后半验证段、10%/30% channel missing、5%/10% noise均改善；prediction-only benchmark 确认无 relation loss、无表示返回、一条 Mamba scan。它已经是强 validation candidate，但没有 opened target test、没有预注册 held-out release、外部基线与stress仍需前置冻结，不能称为一区论文结果。详情见 [`residual-target validation`](mambatsv3_residual_autocon_validation_20261003.md)。

为预检 held-out runner，曾有一个 1 epoch/8 window 的 implementation smoke 产生 test metric。该工件 payload 已正确标识 `test_metrics_produced=true`，但不满足冻结 release plan 的训练预算，已明确排除出所有选择、统计和稿件结论；正式 confirmatory held-out 坐标尚未运行。详情见 [`held-out release plan`](mambatsv3_residual_autocon_heldout_plan_20261003.md)。

最新完整回归（2026-10-03）为 `867 passed, 45 warnings`。2026-10-01 的 `832 passed` 是此前冻结审计的历史记录；新增 DCRNN 缓存扩散实现、完整 DCRNN 矩阵读取器和资源验签器，以及稿件 claim-binding、数值一致性、C/E 完整性审计、MambaTS 可选 Mamba-2 mixer 合同、stride-24 默认路径等价审计、CPU-4 高维 held-out reader 和 LatentTemporalMamba validation reader 合同均已覆盖。Mamba-2 mixer 在高维 matched resource preflight 中比 Mamba-1 慢 `25.020%/26.655%`，因此已停止训练扩展，仅保留为负消融和可复核实现。

当前源码 MambaTS 表示路径另有一个只读 GPU 优化记录：[`mambats profile optimization`](mambats_profile_optimization_20261001.md)。它把固定 patch-to-time 线性插值改为预计算权重 GEMM，在 METR-LA H12 `B32,L96,H12,C207` shape 上 median latency 从 `.3252` 降到 `.3208 ms`，约 `-1.4%`；历史 frozen release 数字未修改。

2026-10-01 stride-24 高维候选的默认路径等价审计通过：当前工作树与 2026-09-29 immutable validation snapshot 的 Mamba-1 v3 参数和预测哈希完全一致，AutoCon representation 最大绝对差为 `3.5763e-7`（低于 `2e-6` FP32 插值等价容差）。这证明新加入的可选 Mamba-2 mixer 未改变 stride-24 held-out 草案所绑定的默认 Mamba-1 路径；审计见 [`stride-24 default-path equivalence`](../../artifacts/exploratory/mambatsv3_stride24_default_path_equivalence_20261001/readout.md)。

## 2026-10-01 CPU-4 高维 held-out release completed; latent temporal candidate remains validation-only

CPU-4 限制的 stride-24 relation-auxiliary Electricity/Traffic H96 release 已完成 `86/86` 作业，并通过 selection `6/6`、candidate `10/10`、baseline `70/70` 三个严格 phase validator。五个 fresh paired test seed 的 `14` 个注册单元中 `13/14` 通过 `1.02` one-sided non-inferiority gate；唯一失败是 Traffic/iTransformer，候选平均 MSE 高 `2.154%`、仅 `1/5` seed 胜出。该失败必须保留在论文表中，不能声称高维全基线普适胜出。机器读数见 [`CPU-4 held-out readout`](stride24_highdim_cpu4_heldout_readout_20261001.md)。

训练 GPU telemetry 峰值为 `57C`、`35%` fan 和 `301.2 W`；CPU-4 环境将 PyTorch intra-op 限制为 4、inter-op 限制为 1。当前 source-bound prediction-only benchmark 显示该候选在 Electricity/Traffic 的 median 为 `.3913/.8366 ms`，快于 PatchTST、TimesNet 和 Mamba2，两域也快于 Traffic iTransformer；但仍慢于 DLinear/NLinear/RLinear 和 Electricity iTransformer。因此速度结论仍是指定形状、指定比较器的结论。

新的 `LatentTemporalMambaAutoCon` 高维资源预检（包含返回的 AutoCon representation）为 `.3763/.5038 ms`，分别比 stride-24 reference `.4255/.8727 ms` 快约 `12%/42%`。但其 CPU-4 60-job validation-only 矩阵已完成且失败：active 相对同构 lambda-zero 在 Electricity/Traffic 分别为 `+10.186%/+6.343%`（均 `0/3`），相对最强外部基线为 `+19.407%/+26.131%`。AutoCon loss 非零且表示/预测共享梯度路径已验证，因此这不是失活分支。停止该候选的 held-out、lambda、宽度和解码器调参；完整记录见 [`latent temporal validation stop`](latent_temporal_mamba_highdim_validation_20261001.md)。

## 2026-09-30 DCRNN 完整 validation-only 基线：候选在两个交通域均胜出

## 2026-10-01 GraphWaveNet held-out extension stopped by thermal protection

## 2026-10-02 Cross-domain AutoCon pretraining line: implementation and smoke only

The new `MambaTSPretrain` line replaces small auxiliary-loss tuning with a channel-count agnostic shared patch encoder and exactly one temporal Mamba-1 scan. It pools channel patch embeddings before scanning, so checkpoint parameter shapes do not depend on sensor count. Pretraining uses masked patch reconstruction plus train-only signed ACF AutoCon relations; fine-tuning/inference disables reconstruction and relation-loss work. The local guarded smoke completed one shared checkpoint across ETTh1 (7 channels), Weather (21), and PEMS03 (358), and a 10% ETTh1 fine-tuning smoke restored compatible encoder keys while keeping the original validation boundary after the target training prefix. A reconstruction-only `lambda=0` control skips relation-loss construction entirely. This is implementation/shape evidence only, not transfer accuracy. The registered low-data/shift/missing-sensor protocol is [`mambats_autocon_pretraining_protocol_20261002.md`](mambats_autocon_pretraining_protocol_20261002.md); its 898-unit matrix remains unauthorized.

The first one-seed Traffic 10% direction pilot revealed an invalid reconstruction mask: the encoder still saw patches whose errors were masked only at aggregation. It is audit-only. A corrected channelwise shared-Mamba pilot replaced selected patch tokens before the scan, preserved per-channel temporal states, excluded Traffic from sources, and used the same target prefix/validation block. Random initialization, masked reconstruction, and masked reconstruction plus AutoCon reached `.963972/.983536/.985084` validation MSE; both pretrained arms were worse by `2.029%/2.190%`. A final one-scan DeepSets channel-context variant then used continuous ACF lag-pair batches, yielding `94.5--96.7%` active global relation anchors. It improved AutoCon relative to reconstruction-only (`.980755` versus `.983376`) but remained `+0.985%` above random initialization (`.971181`). All pilots completed under the GPU guard. The pretraining candidate therefore fails empirically; the 898-unit matrix remains stopped/unauthorized and no post-hoc tuning is allowed. See [`pretraining Traffic pilot`](mambats_autocon_pretraining_traffic_pilot_20261002.md).

The final prediction-aligned variant used 72 observed points to predict the next 24-point patch with Smooth L1 and the same active ACF lag-pair mechanism. Its predictive-only/AutoCon values were `.993730/.992696`, both worse than random initialization `.967596` by `2.701%/2.594%`, even though `95.7--96.7%` of global relation anchors were active. This rejects the remaining goal-alignment explanation; the channel-count agnostic pretraining line is stopped and excluded from the paper's positive claims.

The final physical-interval conditioned pilot added ETTh1/Weather/PEMS03 sampling intervals (`60/10/5` minutes) and target Traffic `60` minutes. Relation anchors remained active (`95.7--96.7%`), but predictive-only/AutoCon validation MSE was `.974532/.976465` versus random initialization `.970218` at 10% Traffic. This rejects the remaining time-scale explanation; the 898-unit pretraining matrix is stopped and no further post-hoc search is authorized.

在一次性授权下启动了冻结的 GraphWaveNet H12 held-out extension。selection `6/6` 和 candidate test `10/10` 正常完成，但第一个 GraphWaveNet baseline 作业持续达到 `74--75C`、`67--68%` fan、约 `510--528W`。为避免达到 `78C` hard guard，进程组在 `75C` 主动终止；GPU 随后回落至 `49C` idle。GraphWaveNet `0/10`、TimesNet/Mamba2 `0/20` 未完成，因此禁止读取或报告 candidate test metrics。该计划已标为 `stopped_gpu_thermal_protection`，完整记录见 [`GraphWaveNet thermal stop`](metro_pemsbay_graphwavenet_h12_thermal_stop_20261001.md)。

在缓存输入扩散和 batched-GEMM 优化后的 DCRNN 实现上，完成 METR-LA/PEMS-BAY H12 的 `12/12`
validation-only 矩阵，三颗 fresh paired seeds、同一 source snapshot、相同 20 epoch 预算与测试关闭。
严格读数见 [`optimized DCRNN readout`](../../artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_readout_20260930/readout.md)。

| Dataset | Candidate MSE | DCRNN MSE | Candidate vs DCRNN | Wins |
| --- | ---: | ---: | ---: | ---: |
| METR-LA | `.273251` | `.288547` | `-5.287%` | `3/3` |
| PEMS-BAY | `.296305` | `.408946` | `-27.444%` | `3/3` |

这补齐了此前 DCRNN 单 seed 预检的精度缺口；DCRNN 仍是本地 canonical-style 实现而不是外部逐位复现。
同一 METR-LA 固定 shape `B32,L96,H12,C207,d24` 的 manifest-bound prediction-only 资源为候选 `.3252 ms`、优化后 DCRNN `9.8622 ms`，候选低约 `2932.51%`，参数为 `29,430/14,268`；这是固定 GPU shape 的前向边界，不是端到端训练吞吐结论。

## 2026-09-30 DCRNN 基线实现优化：缓存单步输入扩散，保持语义不变

对本地 canonical-style DCRNN 做了一个只影响基线实现的工程优化：在每个 recurrent time step 中，正/反图扩散的输入项在 gate 与 candidate 两条路径之间复用；原先重复计算的输入扩散被消除，state diffusion 仍按 gate/candidate 的不同 state 各自计算。新增逐元素 reference regression 验证优化前公式与优化后输出一致，并通过 shape、finite、gradient 测试。

在 METR-LA 固定 `B=32,L=96,H=12,C=207,d=24`、RTX 5090 D、FP32、25 warm-up/100 timed 的 prediction-only 测量中，历史 DCRNN 中位延迟为 `15.6454 ms`，缓存版本为 `13.7850 ms`，约降低 `11.9%`。这只是基线代码的中间优化记录，不能被写成候选 Mamba 的速度提升；完整优化版 DCRNN 结果见顶部 `12/12` 矩阵。

## 2026-09-30 冻结 held-out 证据复核：三组 release 与 52 行主表保持一致

对 PEMS H12、METR-LA/PEMS-BAY H12、Weather H96/H336/H720 三组既有 held-out release 进行了只读
重审计：计划哈希、授权状态、phase validator、manifest 覆盖、summary/test 行数和 `1.02` 非劣界均通过。
PEMS `28/28`、METR-LA/PEMS-BAY `12/12`、Weather `12/12` 比较单元全部保持非劣，candidate paired wins
分别为 `140/140`、`60/60`、`60/60`。新生成的 52 行主表与 2026-09-28 版本逐字一致，CSV SHA-256 均为
`0cf177189542e16ab1d6f2f04922156af2988dfc572d53170ab0dd66aebf384d`。详见
[`2026-09-30 release audit`](../../artifacts/exploratory/paper_release_bundle_audit_20260930/audit.md) 和
[`52-row main table`](../../artifacts/exploratory/paper_main_table_20260930/main_table.md)。

本复核不重跑训练、不重算测试指标，亦不扩大三组 release 的适用范围；它只确认论文主表目前仍具备可追溯性。

## 2026-09-30 AGCRN 自适应图基线：补齐精度结果并确认效率边界

此前只有资源预检的 AGCRN 精度缺口已补齐：METR-LA/PEMS-BAY H12 完成 `12/12` validation-only 配对作业，三颗 fresh seeds、同一 immutable source snapshot、测试关闭。严格读数见 [`AGCRN validation readout`](../../artifacts/exploratory/metro_pemsbay_agcrn_mambatsv3_h12_readout_20260930/readout.md)。

| Dataset | Candidate MSE | AGCRN MSE | Candidate vs AGCRN | Wins |
| --- | ---: | ---: | ---: | ---: |
| METR-LA | `.273466` | `.273197` | `+0.103%` | `1/3` |
| PEMS-BAY | `.296674` | `.288887` | `+2.695%` | `0/3` |

候选在 METR-LA 与 AGCRN 基本持平，但在 PEMS-BAY 落后 `2.695%`，不能宣称相对自适应图模型普遍优越。完整 validation job 时间为候选 `27.7/86.6 s`、AGCRN `1076.6/1697.8 s`；这是训练 job 成本而不是 prediction-only 推理延迟。独立的 manifest-bound RTX 5090 D 固定 shape `B32,L96,H12` 前向验签在 METR-LA 上为候选 `.3240 ms`、AGCRN `18.2249 ms`，候选低 `5524.41%`，参数 `29,430` 对 `72,258`；此结论仅限固定 GPU shape。AGCRN 是本地 canonical-style 实现，不是外部代码逐位复现；本轮作用是补齐精度基线和保留负边界。

## 2026-09-29 Solar H96 variable-axis Mamba 对 TimeMixer：非交通验证集的精度-效率正向证据

在已有 Solar variable-axis Mamba 的 fusion-on/off 机制矩阵基础上，冻结相同紧凑 variable-axis
Mamba-1、`d_model=64`、prediction-connected representation fusion 和 relation loss `lambda=.0001`
合同，并以三颗新 paired seeds 与 TimeMixer 进行 `6/6` 个 validation-only 作业的同源码比较。严格读数见
[`Solar TimeMixer matrix`](../../artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_h96_readout_20260929/readout.md)。

| Horizon | Variable-axis Mamba MSE | TimeMixer MSE | Candidate vs TimeMixer | Wins |
| --- | ---: | ---: | ---: | ---: |
| H96 | `.138005` | `.192720` | `-28.385%` | `3/3` |

在同一 immutable source snapshot、串行 RTX 5090 D、FP32、device-resident `B=32,L=96`、25 warm-up/
100 timed prediction-only 测量中，候选 `.2916 ms`，TimeMixer `.6755 ms`，候选快 `56.83%`，参数为
`69,763` 对 `183,577`。这是这个固定输入形状和比较器下的资源结论；完整验签见
[`Solar TimeMixer resources`](../../artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_resource_attestation_20260929/readout.md)。

三条候选训练日志的 relation loss 有限、非零并进入训练目标，最大恒等式残差 `2.75e-10`，但其平均加权
份额仅 `.053%`，且本轮固定 fusion 扩展并未重新检验 relation-null。因此可将结果作为 variable-axis
architecture 加 prediction-connected representation fusion 的 validation-only 正向证据，不能单独归因于
AutoCon relation loss，更不能提升为 held-out 或普适结论；审计见
[`Solar AutoCon audit`](../../artifacts/exploratory/nlinear_variable_mamba_solar_timemixer_h96_training_audit_20260929/readout.md)。

## 2026-09-29 ETTh2 长预测 TimeMixer 扩展：资源领先但精度失败，停止该分支

在 ETTh2 H96 相对 TimeMixer 的正向结果后，严格冻结相同 stride-24、d32、relation-auxiliary-only
MambaTSv3AutoCon 合同，不作 horizon 特化，完成 H336/H720、三颗 fresh paired seeds 的 `12/12`
validation-only 作业。严格读数见 [`ETTh2 long TimeMixer matrix`](../../artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_long_h336_h720_readout_20260929/readout.md)。

| Horizon | Candidate MSE | TimeMixer MSE | Candidate vs TimeMixer | Wins |
| --- | ---: | ---: | ---: | ---: |
| H336 | `.372600` | `.357728` | `+4.157%` | `0/3` |
| H720 | `.618063` | `.609351` | `+1.438%` | `0/3` |

H336 明显违反预注册的 2% 非劣界，H720 虽在 2% 内但无任何 paired seed 胜出；因此不把 H96
的局部成功推广到 ETTh2 长预测，也不再进行 lambda、patch、宽度或训练预算搜索。候选的 AutoCon
对比损失在两种 horizon 均有限非零，并实际进入训练目标：H336/H720 分别下降 `2.502%/2.271%`，
最大恒等式残差 `2.45e-09`，但这不改变精度停止结论；审计见 [`ETTh2 long AutoCon audit`](../../artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_long_training_audit_20260929/readout/readout.md)。

同一 immutable source snapshot、串行 RTX 5090 D、FP32、device-resident `B=32,L=96` 的
prediction-only 测量中，候选 H336/H720 为 `.2713/.2816 ms`，TimeMixer 为 `.4969/.4885 ms`，
即快 `45.40%/42.34%`，且参数更少。这仅证明该失败坐标的固定 shape 推理成本较低，不能抵消
精度失败或成为普适速度主张；完整资源见 [`ETTh2 long TimeMixer resources`](../../artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_long_resource_20260929/readout/readout.md)。

## 2026-09-29 ETTh2 TimeMixer 扩展：候选击败多尺度 MLP，但仍不是强基线冠军

为补齐 ETTh2 上的多尺度 MLP 对照，冻结与 ETTm2 TimeMixer 扩展相同的 stride-24、d32、
relation-auxiliary-only MambaTSv3AutoCon 合同，完成 ETTh2 H96、三颗 fresh paired seeds 的 `6/6`
validation-only 作业。严格读数见 [`ETTh2 TimeMixer matrix`](../../artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_h96_readout_20260929/readout.md)。

候选均值 MSE 为 `.223213`，TimeMixer 为 `.229752`，相对改善 `2.837%`，三颗种子均胜出。
这补充了一个跨 ETT 频率域的 TimeMixer 正向对照，但 ETTh2 上候选仍落后已有 RLinear/PatchTST/
iTransformer 强基线约 `10.90%/9.84%/5.91%`，因此不能写成跨模型普遍精度领先；该结果应与 ETTh2
负边界一并报告。

同一 immutable source snapshot、串行 RTX 5090 D、FP32、device-resident `B=32,L=96` 的
prediction-only 测量显示候选 `0.2772 ms`、TimeMixer `0.4977 ms`，候选快 `44.30%`；参数量为
`39,544` 对 `70,135`。这是固定输入形状下相对 TimeMixer 的资源结论，不是端到端吞吐或普适速度主张，
详见 [`ETTh2 TimeMixer resources`](../../artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_resource_20260929/readout/readout.md)。

三次候选训练日志的 AutoCon 对比损失均有限非零，平均下降 `2.290%`，加权占比 `.729%`，
目标恒等式最大残差 `1.21e-09`。这证明 AutoCon 表示目标确实参与了训练，但没有 active/null 对照，
不能将候选相对 TimeMixer 的差异归因于 AutoCon 的独立因果效应；审计见 [`ETTh2 AutoCon audit`](../../artifacts/exploratory/mambatsv3_stride24_etth2_timemixer_training_audit_20260929/readout/readout.md)。

## 2026-09-29 ETTm2 TimeMixer 强基线扩展：三种 horizon 均同时优于精度与前向速度

为补齐多尺度 MLP 基线，冻结 stride-24 relation-auxiliary-only MambaTSv3 和 TimeMixer 的公共训练合同，
完成 ETTm2 H96/H336/H720、三颗 fresh paired seeds 的 `18/18` validation-only 作业。严格读数见
[`ETTm2 TimeMixer matrix`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_timemixer_h96_h336_h720_readout_20260929/readout.md)。

| Horizon | Candidate MSE | TimeMixer MSE | Candidate vs TimeMixer | Wins |
| --- | ---: | ---: | ---: | ---: |
| H96 | `.125868` | `.142550` | `-11.319%` | `3/3` |
| H336 | `.220017` | `.266009` | `-15.356%` | `3/3` |
| H720 | `.290159` | `.301022` | `-3.320%` | `2/3` |

同一 immutable snapshot 下、串行 RTX 5090 D、FP32、device-resident `B=32,L=96`、25 warm-up/100 timed
prediction-only 测量显示候选为 `.2741/.2773/.2760 ms`，TimeMixer 为 `.4979/.5035/.4862 ms`，即候选在
H96/H336/H720 分别快 `44.94%/44.93%/43.24%`，且参数量为 `39,544/93,784/180,568`，低于 TimeMixer
的 `70,167/140,247/252,375`。这是针对 TimeMixer 的固定 shape 速度结论，不是端到端吞吐或普适速度主张；
见 [`TimeMixer resource attestation`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_timemixer_resource_20260929/readout/readout.md)。

候选的九个训练日志均有有限非零的 AutoCon contrastive loss，H96/H336/H720 平均下降
`1.430%/1.714%/1.822%`，最大 `train_loss = mse_loss + .001 * autocon_loss` 残差为 `1.14e-09`。
这证明本轮精度-速度比较确实保留了 AutoCon 训练期表示学习，但不能据此将全部差异归因于对比损失。
审计见 [`TimeMixer AutoCon audit`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_timemixer_training_audit_20260929/readout.md)。

随后完成 ETTh2 H96 的精确 active/null 机制控制：唯一变化为 `AutoCon_lambda=.001` 对 `0`，
使用三颗新的 paired seeds，共 `6/6` 个 validation-only 作业。active MSE `.221769`，lambda-zero
为 `.221790`，active 相对 null 改善 `0.010%`，三颗 seed 全胜，最大单 seed 退化为负，预注册局部门槛
通过。该独立信号非常小，只能说明 AutoCon 在此坐标有可测的局部作用，不能解释候选相对 TimeMixer
的全部优势，也不能写成普遍因果结论。严格读数见 [`ETTh2 AutoCon mechanism`](../../artifacts/exploratory/mambatsv3_stride24_etth2_autocon_mechanism_h96_readout_20260929/readout.md)。

## 2026-09-29 ETTm2 H336/H720 扩展：未超过非劣界，但未击败 PatchTST

为检验 H96 成功是否能延续到长预测，冻结 d32 stride-24 relation-auxiliary-only MambaTSv3 配置，
不做 horizon 特化调参，完成 ETTm2 H336/H720、七个基线、三颗 fresh paired seeds 的 `48/48`
validation-only 矩阵。严格读数见 [`ETTm2 long matrix`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_long_h336_h720_readout_20260929/readout.md)。

| Horizon | Candidate MSE | Strongest baseline | Candidate vs strongest | Wins |
| --- | ---: | --- | ---: | ---: |
| H336 | `.219028` | PatchTST `.216659` | `+1.095%` | `0/3` |
| H720 | `.293225` | PatchTST `.287750` | `+1.913%` | `0/3` |

在 H336，候选仍胜 iTransformer、Mamba-1、TimesNet 和 Mamba-2，但略落后 DLinear/PatchTST；在
H720，仅稳定胜 iTransformer 与 Mamba-2，落后 DLinear、RLinear、PatchTST、Mamba-1 和 TimesNet。
两个 horizon 均未越过预注册的 2% 非劣边界，却没有任何 seed 击败 PatchTST，因此不能把 ETTm2
的 H96 正向结果泛化为长预测优势，也不启动长预测 held-out 或针对结果调参。

六个候选作业的只读训练审计确认对比损失有限非零：H336/H720 分别下降 `1.914%/1.602%`，加权占比
`.706%/.496%`，目标恒等式最大残差 `4.39e-10/1.01e-09`。这证明 AutoCon 表示学习参与了长预测训练，
但不能将此写成精度因果结论。详见 [`ETTm2 long AutoCon audit`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_long_h336_h720_training_audit_20260929/readout.md)。

## 2026-09-29 d16 轻量化预检：参数下降但固定 shape 更慢，停止训练

为优化当前 ETTm2 stride-24 候选的推理成本，先对 `d_model=16,d_ff=32,d_state=8,d_conv=2,expand=1`
的 AutoCon MambaTSv3 做 resource-only 预检。相同 RTX 5090 D、FP32、`B=32,L=96,H=96`、25
warm-up/100 timed 下，d16 只有 `19,272` 个可训练参数、峰值 allocated `9.78 MB`，但中位延迟
为 `.2835 ms`；当前 d32 候选为 `39,544` 参数、`10.74 MB`、`.2746 ms`。d16 反而慢约 `3.2%`，
说明该 shape 下 Mamba kernel 固定开销高于缩宽收益。

该预检没有训练、验证或测试工件。按预先停止规则，d16 不启动 ETTm2 validation matrix，也不进行
后续宽度/lambda 搜索。完整说明见 [`d16 preflight`](mambatsv3_stride24_ettm2_d16_resource_preflight_20260929.md)。

## 2026-09-29 stride-24 ETTm2 H96 扩展：第二个高频 ETT 域全胜，但速度仍不是线性基线级别

沿用 ETTm1 已冻结的 stride-24、relation-auxiliary-only MambaTSv3 配置，不做 ETTm2 特化调参，完成
`24/24` 个 validation-only 作业：三颗 fresh paired seeds `3401--3403`，候选与 DLinear、RLinear、
PatchTST、iTransformer、Mamba-1、TimesNet、Mamba-2 共用同一 immutable source snapshot。严格读数见
[`ETTm2 matrix`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_h96_readout_20260929/readout.md)。

| Baseline | Candidate vs baseline | Wins |
| --- | ---: | ---: |
| DLinear | `-0.924%` | `3/3` |
| RLinear | `-1.135%` | `3/3` |
| PatchTST | `-0.986%` | `3/3` |
| iTransformer | `-3.491%` | `3/3` |
| Mamba-1 | `-5.982%` | `3/3` |
| TimesNet | `-3.578%` | `3/3` |
| Mamba-2 | `-10.106%` | `3/3` |

候选均值 MSE 为 `.125532`。同一 RTX 5090 D、FP32、`B=32,L=96,H=96`、25 warm-up/100 timed 的
prediction-only 测量中，候选 `.2746 ms`，PatchTST/iTransformer/Mamba-1/TimesNet/Mamba-2 为
`.2665/.1563/.4165/2.8330/.7054 ms`，DLinear/RLinear 为 `.0945/.0660 ms`。因此 ETTm2 复现了
“快于多数非线性基线、慢于线性基线”的资源边界，不能支持普遍速度优势。

训练日志审计确认 AutoCon loss 有限且非零，平均下降 `1.302%`，`train_loss = mse_loss + .001 *
autocon_loss` 最大残差 `4.48e-10`。这证明对比表示目标实际进入训练，但没有 ETTm2 active/null 对照，
所以不能将相对基线的精度差异归因于 AutoCon 单独作用。详见 [`ETTm2 resource attestation`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_resource_20260929/readout/readout.md) 和 [`ETTm2 AutoCon audit`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_training_audit_20260929/readout.md)。

随后完成同配置的 ETTm2 active/null 机制控制，唯一改变为 `AutoCon_lambda=.001` 对 `0`，三颗
fresh seeds `3411--3413`，共 `6/6` 个 validation-only 作业。active MSE `.124977`，lambda-zero
为 `.125106`，active 相对 null 改善 `0.102%`、`2/3` seed 胜出，最大单 seed 退化 `0.038%`，
预注册局部门槛通过。该结果支持 ETTm2 上一个小幅的局部辅助目标信号，但幅度不足以支持普遍
AutoCon 因果主张；严格读数见 [`ETTm2 mechanism readout`](../../artifacts/exploratory/mambatsv3_stride24_ettm2_autocon_mechanism_h96_readout_20260929/readout.md)。

## 2026-09-29 ETTm1 stride-24 AutoCon active/null 精确控制：目标参与但独立预测增益近零

在上一节 ETTm1 七基线矩阵基础上，进一步完成同一 stride-24 MambaTSv3、同一切分/预算、三颗
paired fresh seeds `3401--3403` 的 `6/6` validation-only active/null 对照；唯一差异是
`AutoCon_lambda=.001` 与 `0`，prediction fusion 两臂均关闭。严格读数见
[`ETTm1 mechanism readout`](../../artifacts/exploratory/mambatsv3_stride24_ettm1_autocon_mechanism_h96_readout_20260929/readout.md)。

| Arm | Validation MSE | Active vs null | Wins |
| --- | ---: | ---: | ---: |
| AutoCon active (`lambda=.001`) | `.385314` | `-0.004%` | `2/3` |
| Lambda-zero control | `.385329` | reference | - |

active 臂的原始对比损失为有限非零值，且 `effective_lambda=.001`；null 臂仍计算并记录原始损失，
但 `effective_lambda=0`，所以对比目标不进入 `train_loss`。这排除了“AutoCon 没有执行”的解释，
但 `0.004%` 的局部差异远小于可作为方法贡献的幅度。因此论文应把它写成 ETTm1 上的精确负/近零
机制消融，不能写成 AutoCon 独立提高预测精度的证据，也不应因此启动新的 lambda 搜索或测试释放。

## 2026-09-29 stride-24 ETTm1 H96 扩展：新增跨频率域验证成功，但仍不是测试结论

为检验高维/周期性结果能否迁移到高频 ETT 数据，冻结现有 stride-24 relation-auxiliary-only
候选，不做 ETTm1 专门调参，完成 `24/24` 个 validation-only 作业：ETTm1 H96、三颗 fresh
paired seeds `3391--3393`，候选与 DLinear、RLinear、PatchTST、iTransformer、Mamba-1、TimesNet、
Mamba-2 共用同一源码快照、数据切分、训练预算和 `patch_len=stride=24`。严格读数见
[`ETTm1 matrix`](../../artifacts/exploratory/mambatsv3_stride24_ettm1_h96_readout_20260929/readout.md)。

| Baseline | Candidate vs baseline | Wins |
| --- | ---: | ---: |
| DLinear | `-5.561%` | `3/3` |
| RLinear | `-5.904%` | `3/3` |
| PatchTST | `-3.389%` | `3/3` |
| iTransformer | `-2.500%` | `3/3` |
| Mamba-1 | `-8.255%` | `3/3` |
| TimesNet | `-13.007%` | `3/3` |
| Mamba-2 | `-9.048%` | `3/3` |

同一 RTX 5090 D、FP32、`B=32,L=96,H=96`、25 warm-up/100 timed 的 manifest-bound
prediction-only 测量中，候选为 `.2855 ms`，PatchTST/iTransformer/Mamba-1/TimesNet/Mamba-2 为
`.2712/.1579/.4241/2.7528/.7090 ms`，但 DLinear/RLinear 为 `.1013/.0674 ms`。因此该结果只支持
ETTm1 H96 上相对命名非线性基线的固定 shape 速度-精度补充，不支持普遍速度优势。资源详见
[`ETTm1 resource attestation`](../../artifacts/exploratory/mambatsv3_stride24_ettm1_resource_20260929/readout/readout.md)。

训练日志审计确认三次候选作业的 AutoCon loss 有限且非零，平均下降 `2.760%`，`train_loss = mse_loss
+ .001 * autocon_loss` 最大残差 `6.08e-10`；这证明目标参与训练，但不能将 ETTm1 的差异归因于
AutoCon 单独作用。见 [`ETTm1 AutoCon audit`](../../artifacts/exploratory/mambatsv3_stride24_ettm1_training_audit_20260929/readout.md)。

## 2026-09-29 轻量 MambaTS v1 的 train-only AutoCon 控制：零额外推理成本，但未带来精度收益

为隔离“AutoCon 训练目标”与 v3 共享上下文/结构变化，完成同一轻量 `MambaTS` v1、stride-24、H96 的严格
AutoCon on/off 矩阵：ETTh1、Weather、Electricity、Traffic，各三颗 paired fresh seed，`24/24`
validation-only 作业均成功、共享 immutable source snapshot，且没有测试工件。候选仅在训练时启用
`AutoCon_lambda=.001`；推理调用显式 `return_repr=false`，因而不返回或计算辅助表示。严格读数见
[`AutoCon on/off accuracy`](../../artifacts/exploratory/mambats_autocon_vs_plain_stride24_h96_readout_20260929/readout.md)。

| Dataset | AutoCon MSE | Plain MambaTS MSE | AutoCon vs plain | Wins |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | `.698067` | `.695794` | `+0.328%` | `2/3` |
| Weather | `.486394` | `.482320` | `+0.852%` | `2/3` |
| Electricity | `.182232` | `.182122` | `+0.060%` | `1/3` |
| Traffic | `.396213` | `.396018` | `+0.050%` | `1/3` |

训练日志审计确认 12 个候选作业的 contrastive loss 均为有限非零值，`train_loss = mse_loss + .001 *
autocon_loss` 的最大残差为 `1.68e-09`；详细值见 [`training audit`](../../artifacts/exploratory/mambats_autocon_vs_plain_stride24_h96_training_audit_20260929/readout.md)。
同一 RTX 5090 D、FP32、`B=32,L=96,H=96`、25 warm-up/100 timed 的 manifest-bound prediction-only
资源审计中，候选与 plain MambaTS 的中位延迟分别为 `.2229/.2199/.3301/.7342 ms` 与
`.2203/.2200/.3297/.7334 ms`，最大差异为 `+1.2%`；见 [`resource attestation`](../../artifacts/exploratory/mambats_autocon_vs_plain_stride24_resource_20260929/readout/readout.md)。

因此，这条实验证明训练期表征约束可与 prediction-only 轻量推理解耦，但**不**支持“AutoCon 在 plain
MambaTS v1 上独立提高精度”的主张。它应作为负消融和方法边界，而不能进入正向主结果表或触发 held-out
release。

## 2026-09-29 stride-24 完整六类基线复核：高维精度通过，速度优势仍有边界

补齐 TimesNet 和 Mamba-2 后，stride-24 relation-auxiliary 候选的同源码高维矩阵达到 `42/42`
个 validation-only 作业、12 个 paired comparison cells。Electricity/Traffic 上相对 DLinear、
PatchTST、iTransformer、Mamba-1、TimesNet、Mamba-2 均为 `3/3` seed 胜出；候选均值分别为
`.163764/.384616`，相对最强 comparator（TimesNet/iTransformer）为 `-3.438%/-4.332%`。
完整读数见 [`stride-24 full matrix`](../../artifacts/exploratory/mambatsv3_stride24_full_baseline_h96_validation_20260929/readout/readout.md)。

manifest-bound RTX 5090 D fixed-shape resource attestation 给出 candidate `.3971/.8379 ms`，
相对 PatchTST、TimesNet、Mamba-2 两域均更快；Electricity 还快于 Mamba-1 `.4604 ms`，Traffic
还快于 iTransformer `.9157 ms`。但 candidate 仍慢于 DLinear `.1140/.2333 ms`，Electricity 慢于
iTransformer `.2964 ms`，Traffic 慢于 Mamba-1 `.6627 ms`。因此当前可支持“高维 H96、固定 shape、
相对指定非线性基线的精度-前向 Pareto”，不能支持普遍速度优势或端到端吞吐结论。资源详见
[`stride-24 full resource attestation`](../../artifacts/exploratory/mambatsv3_stride24_full_baseline_resource_20260929/readout/readout.md)。

## 2026-09-29 stride-24 跨域复核：Weather 全胜，ETTh1 保留失败边界

同一 stride-24、relation-auxiliary-only 配置在 Weather/ETTh1 H96 的 `48/48` 个
validation-only 作业均成功，覆盖 DLinear、RLinear、PatchTST、iTransformer、Mamba-1、TimesNet 与
Mamba-2，三颗配对 seed。Weather 候选 MSE 为 `.444975`，对七类基线均为 `3/3` 胜出；相对最强
TimesNet `.463439` 为 `-3.977%`。ETTh1 候选 `.699699` 只优于 Mamba-1、TimesNet 与 Mamba-2，
而相对 DLinear/RLinear/PatchTST/iTransformer 为 `+0.344%/+0.031%/+1.611%/+2.425%`，不能选择性
忽略。完整结果见 [`stride-24 cross-domain matrix`](../../artifacts/exploratory/mambatsv3_stride24_crossdomain_h96_readout_20260929/readout.md)。

manifest-bound RTX 5090 D fixed-shape resource 审计（FP32、`B=32,L=96,H=96`、25 warm-up/100 timed）
给出 ETTh1/Weather 候选 `.2817/.2755 ms`。两域均快于 Mamba-1、TimesNet 与 Mamba-2，Weather 也略快于
PatchTST；但候选慢于 DLinear、RLinear 和 iTransformer，且 ETTh1 略慢于 PatchTST。因此只可将 Weather
作为补充的 scoped accuracy-latency observation；ETTh1 是明确精度失败边界。详见
[`stride-24 cross-domain resource attestation`](../../artifacts/exploratory/mambatsv3_stride24_crossdomain_resource_20260929/readout/readout.md)。

## 2026-09-29 stride-24 ETTh1 表示门控 blend：资源预检停止

为避免重复训练一个已知会牺牲速度的低维修复分支，对 `stride-24` 单扫描骨干先做了与既有
ETTh1 相同输入形状的 GPU 前向预检。候选保留 `AutoCon_lambda=.001`，仅启用已实现的
AutoCon representation-gated persistence blend（`bias=-4`），不增加第二次 Mamba scan。RTX 5090 D、
FP32、`B=32,L=96,H=96,C=7`、25 warm-up/100 timed 下，blend 为 `.3472 ms`、`45,785` 参数；完全相同
显式 control 为 `.2806 ms`、`39,544` 参数，即 blend 增加 `23.8%` 延迟。后者与已审计的 PatchTST
`.2676 ms` 已接近，而 blend 更慢，因此不满足本项目“精度不退化且更快”的架构筛选门槛。该分支未启动
任何训练、验证或测试集作业；只保留资源预检记录
[`blend`](../../artifacts/exploratory/mambatsv3_stride24_blend_etth1_resource_preflight_20260929/model_resources.json) 与
[`control`](../../artifacts/exploratory/mambatsv3_stride24_control_etth1_resource_preflight_20260929/model_resources.json)。

## 2026-09-29 stride-24 AutoCon 候选与 plain MambaTS 官方结构对照

为补齐与公开 MambaTS 架构的直接比较，冻结并完成了 `24/24` 个 validation-only 作业：
四个数据集（ETTh1、Weather、Electricity、Traffic）、H96、三颗 fresh paired seeds，候选为
stride-24 `MambaTSv3AutoCon`，对照为 plain local `MambaTS` v1。后者保留公开 MambaTS 的 patch-token/
variable-aware scan 高层思路，但明确不是官方仓库的 bitwise reproduction。严格读数见
[`MambaTS comparison`](../../artifacts/exploratory/mambatsv3_stride24_official_mambats_baseline_h96_readout_20260929/readout.md)。

| Dataset | AutoCon-v3 MSE | Plain MambaTS MSE | Candidate delta | Wins |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | `.701401` | `.694153` | `+1.047%` | `0/3` |
| Weather | `.442215` | `.482679` | `-8.376%` | `3/3` |
| Electricity | `.164678` | `.182222` | `-9.625%` | `3/3` |
| Traffic | `.382755` | `.396599` | `-3.490%` | `3/3` |

这组结果支持更窄的结论：AutoCon relation auxiliary 与 v3 的共享上下文在高维/周期性域相对 plain
MambaTS 有稳定验证收益，但 ETTh1 仍失败，不能宣称跨域普适改进。训练日志审计还确认 12 个候选作业
的 `train_loss = mse_loss + lambda * autocon_loss` 恒等式残差不超过 `9.65e-10`，AutoCon loss 在四个
数据集均下降，说明对比表示目标确实参与训练；这不构成其独立因果贡献证明。

同一 RTX 5090 D fixed-shape prediction-only 资源审计显示候选相对 plain MambaTS 更慢：ETTh1/Weather/
Electricity/Traffic 分别为 `.2778/.2781/.3926/.8383 ms` 对 `.2177/.2171/.3300/.7329 ms`，增幅
`27.6%/28.1%/19.0%/14.4%`。因此当前速度主张必须限定为相对指定的 PatchTST、TimesNet、Mamba2 等
比较器，不能写成“比 MambaTS 或所有 Mamba 结构更快”。资源审计见
[`MambaTS resources`](../../artifacts/exploratory/mambatsv3_stride24_official_mambats_baseline_resource_20260929/readout/readout.md)。

候选训练项的独立审计见 [`AutoCon training audit`](../../artifacts/exploratory/mambatsv3_stride24_official_mambats_autocon_training_audit_20260929/readout.md)。

## 2026-09-29 stride-24 relation-auxiliary MambaTSv3：高维验证集的精度-速度 Pareto 候选

为直接处理 v3 相对 plain Mamba-1 的速度劣势，冻结了 `30/30` 个 GPU validation-only 作业：
将 `L=96` 划分为四个不重叠的 `24x24` patch，保留单次 Mamba-1 scan、AutoCon
`lambda=.001` 和关闭 prediction fusion；在 Electricity/Traffic H96 上与 DLinear、PatchTST、
iTransformer、纯 Mamba-1 使用同预算、同 seed `3361--3363` 对比。严格 reader 核验完整覆盖、
统一 source snapshot 和零测试工件，读数见 [`stride-24 accuracy readout`](../../artifacts/exploratory/mambatsv3_stride24_relation_auxiliary_h96_validation_20260929/readout/readout.md)。

| Dataset | Candidate MSE | Strongest registered baseline | Candidate vs strongest | All paired comparisons |
| --- | ---: | --- | ---: | ---: |
| Electricity H96 | `.163764` | Mamba-1 `.177947` | `-7.960%` | `4/4`, each `3/3` |
| Traffic H96 | `.384616` | iTransformer `.402049` | `-4.332%` | `4/4`, each `3/3` |

同一 validation-manifest-bound RTX 5090 D prediction-only 协议（FP32、`B=32,L=96,H=96`、
25 warm-up/100 timed）下，candidate 为 `.3971/.8379 ms`：两数据集均快于 PatchTST
`.6406/1.6233 ms`；Electricity 快于 Mamba-1 `.4604 ms`，Traffic 快于 iTransformer `.9157 ms`。
但仍慢于 DLinear `.1140/.2333 ms`，且分别慢于 Electricity iTransformer `.2964 ms` 和 Traffic
Mamba-1 `.6627 ms`。资源记录见 [`stride-24 resource attestation`](../../artifacts/exploratory/mambatsv3_stride24_relation_auxiliary_h96_resource_20260929/readout/readout.md)。

这是当前最接近“精度不低于已注册强基线且相对指定非线性基线更快”的高维 validation-only 候选，
但未覆盖 TimesNet/Mamba2、跨域、长 horizon 或 held-out；不可替代现有主模型、不可写成普适速度或
AutoCon 独立因果结论，也不授权测试 release。

## 2026-09-29 纯 Mamba-1 基线补充：候选在验证集描述性比较中更低

为补齐架构对照，新增纯 `Mamba1` 的 Electricity/Traffic H96 validation-only 基线，6/6 个 GPU
作业全部成功，使用 seeds `3351--3353`；候选复用既有 relation-auxiliary-only 工件（seeds
`3291--3293`）。两组种子不同，因此严格 reader 只报告独立新种子的均值，不报告 paired wins。
读数见 [`Mamba-1 readout`](../../artifacts/exploratory/mambatsv3_mamba1_baseline_highdim_h96_validation_20260929/readout/readout.md)。

| Dataset | Candidate MSE | Plain Mamba-1 MSE | Candidate vs Mamba-1 |
| --- | ---: | ---: | ---: |
| Electricity H96 | `.158235` | `.183244` | `-13.648%` |
| Traffic H96 | `.372924` | `.489545` | `-23.822%` |

该结果只能作为同源码、验证集、独立 fresh-seed 的补充基线证据；它不是配对比较、held-out 结果，
也不能单独证明 AutoCon relation loss 的因果贡献。Mamba-1 已进入基线登记，但没有回写既有
高维 held-out 草案，避免改变已冻结计划哈希。

同一组已训练配置的 source-bound RTX 5090 D prediction-only 测量显示，候选并没有速度优势：
Electricity 为 `.8011 ms`，纯 Mamba-1 为 `.4469 ms`（候选慢 `79.2%`）；Traffic 为 `2.2067 ms`
对 `.6451 ms`（候选慢 `242.0%`）。候选峰值显存为 `247.6/649.8 MB`，纯 Mamba-1 为
`54.0/128.2 MB`。该资源边界严格绑定两个 validation manifest 和共享 source tree，测量使用
`B=32,L=96,H=96`、25 warm-up/100 timed、合成 device-resident 输入，不含 relation loss 或
表示返回。详见 [`Mamba-1 resource attestation`](../../artifacts/exploratory/mambatsv3_relation_auxiliary_mamba1_resource_h96_20260929/readout/readout.md)。

因此当前论文不能把候选表述为“比纯 Mamba-1 更快”；更准确的定位是候选在部分基线精度上更强，
但 v3 预测路径相对 plain Mamba-1 存在明确的固定形状效率劣势。

## 2026-09-29 Solar H96 relation-auxiliary-only fresh-seed复核：相对多数基线通过，但仍输 iTransformer

针对旧 Solar fusion 候选输给 iTransformer 的边界，新增冻结的 `24/24` GPU validation-only 矩阵：
relation-auxiliary-only MambaTSv3 候选与 DLinear、NLinear、RLinear、PatchTST、TimesNet、iTransformer、
Mamba2，fresh seeds `3341--3343`，同一 source snapshot
`1e3976666f8971abe3bf884f2635bbaf57f522d327201f183647544a666a63a5`。严格读数见
[`Solar relation-auxiliary readout`](../../artifacts/exploratory/mambatsv3_relation_auxiliary_solar_h96_validation_20260929/readout/readout.md)。

| Baseline | Candidate vs baseline | Wins |
| --- | ---: | ---: |
| DLinear / NLinear / RLinear | `-32.383% / -32.552% / -32.645%` | `3/3` each |
| PatchTST / TimesNet | `-4.167% / -14.323%` | `3/3` each |
| Mamba2 | `-23.304%` | `3/3` |
| iTransformer | `+5.839%` | `0/3` |

候选均值为 `.166694`。因此 relation-auxiliary-only 路线在 Solar 上改善了对多数基线的验证差距，
但仍无法通过最强 iTransformer 门槛；Solar 必须作为高维非交通失败边界保留，不能进入普适优势或
held-out 主结果声明。

## 2026-09-29 relation-auxiliary-only 的 AutoCon 表示学习训练审计

为避免将 `AutoCon=True` 仅当作配置标签，对高维、跨域与 Solar 矩阵共 `15` 个候选作业进行只读训练日志审计。
审计器逐作业核验：训练期返回表示的 AutoCon 合同、`lambda=.001`、有限且非零的 contrastive loss、
以及逐 epoch 的 `train_loss = mse_loss + lambda * autocon_loss` 恒等式；最大残差不超过 `1.29e-09`，
且没有测试工件。机器读数见 [`AutoCon training audit`](../../artifacts/exploratory/relation_auxiliary_autocon_training_audit_20260929/readout/readout.md)。

| Dataset | First / last AutoCon loss | Relative change | Weighted objective share |
| --- | ---: | ---: | ---: |
| Electricity | `2.234786 / 2.144404` | `-4.044%` | `1.536%` |
| Traffic | `1.923620 / 1.875499` | `-2.502%` | `.719%` |
| Weather | `1.931406 / 1.889970` | `-2.145%` | `.517%` |
| ETTh1 | `2.168218 / 2.128356` | `-1.838%` | `.665%` |
| Solar | `2.247841 / 2.112414` | `-6.025%` | `1.205%` |

该结果证明 train-only AutoCon contrastive representation objective 实际进入梯度优化，且不是零权重或
日志占位；但是 loss 下降本身不证明其导致预测精度提升。由于 highdim mechanism matrix 中 prediction
fusion 失败，表示仍保持为训练期辅助约束，不据此主张预测路径的 representation fusion 增益。

## 2026-09-29 Weather / ETTh1 跨域 relation-auxiliary-only 复核：成功域稳定，低维失败边界复现

为检验高维候选是否只是 Electricity/Traffic 的选择性结果，新增冻结的 `36/36` GPU
validation-only 矩阵：Weather（成功域）与 ETTh1（低维困难域），候选为 MambaTSv3 v3
保留 AutoCon relation loss、关闭 prediction-connected fusion；同时重新训练 RLinear、PatchTST、
TimesNet、iTransformer 和 Mamba2，三颗 fresh seeds `3331--3333`。所有作业共享 source snapshot
`1e3976666f8971abe3bf884f2635bbaf57f522d327201f183647544a666a63a5`，没有测试工件。严格读数见
[`cross-domain readout`](../../artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/readout/readout.md)。

| Dataset | Candidate MSE | Strongest baseline | Candidate vs strongest | Wins vs 5 baselines |
| --- | ---: | --- | ---: | ---: |
| Weather H96 | `.445005` | TimesNet `.454056` | `-1.99%` | `5/5`, each `3/3` |
| ETTh1 H96 | `.709807` | iTransformer `.682973` | `+3.93%` | `2/5` |

Weather 上候选相对 RLinear、PatchTST、TimesNet、iTransformer、Mamba2 均为 `3/3` 胜出；ETTh1
仍只优于 TimesNet 和 Mamba2，落后 RLinear `1.50%`、PatchTST `3.17%` 和 iTransformer `3.93%`。
这复现了低维失败边界，支持“高维/周期性域具有竞争力”的限定叙事，而不支持跨数据集普适优势。

同一 cross-domain matrix 的 manifest-bound RTX 5090 D resource audit 显示，Weather/ETTh1 H96 的候选
median 均为 `.2758/.2751 ms`，快于 PatchTST `.2858/.2827 ms`、TimesNet `3.1347/2.9577 ms` 和 Mamba2
`.7158/.7047 ms`，但慢于 RLinear `.0648/.0659 ms` 与 iTransformer `.1535/.1560 ms`。Weather 同时通过
精度筛选，故可作为独立的 fixed-shape accuracy/latency Pareto 补充；ETTh1 的精度失败边界仍优先于其
速度结果。资源细节见 [`cross-domain resource readout`](mambatsv3_relation_auxiliary_crossdomain_h96_resource_20260929.md)。

## 2026-09-28 Electricity / Traffic H96 relation-auxiliary-only 确认：精度与前向路径同时改善

根据紧邻的机制矩阵，冻结新的 MambaTSv3 v3 候选：保留 `AutoCon_lambda=.001`，关闭
prediction-connected representation fusion。只新增 `6/6` 个候选 GPU validation-only 作业，并与已完成、
同源码快照、同 seed 的 `42` 个强基线工件通过严格 reader 合并；没有测试指标、预测数组或 streaming
工件。读数见 [`完整 matrix`](../../artifacts/exploratory/mambatsv3_highdim_relation_auxiliary_only_h96_validation_20260928/readout/readout.md)。

| Dataset | Candidate MSE | Strongest baseline | Candidate vs strongest | Wins | vs prior fusion candidate |
| --- | ---: | --- | ---: | ---: | ---: |
| Electricity H96 | `.158235` | TimesNet `.168558` | `-6.117%` | `3/3` | `-1.798%`, `3/3` |
| Traffic H96 | `.372924` | iTransformer `.401117` | `-7.029%` | `3/3` | `-1.379%`, `3/3` |

在同一 RTX 5090 D、FP32、`B=32,L=96,H=96`、25 warm-up/100 timed 的 prediction-only 协议中，
该候选在 Electricity/Traffic 的 median 为 `.8011/2.2052 ms`，相对 prior fusion 路径
`.9231/2.3551 ms` 分别低 `13.2%/6.4%`，参数为 `76,753/103,262`。资源细节见
[`resource readout`](mambatsv3_highdim_relation_auxiliary_only_h96_resource_20260928.md)。

这使高维候选的证据更一致：AutoCon objective 作为辅助训练约束可保留，prediction-connected fusion
不应在此高维配置中启用。该结论仍是 validation-only，且只支持相对 PatchTST/TimesNet 的指定 shape
前向延迟优势；它不覆盖线性、iTransformer 或 Mamba2，也不授权 held-out release。

据此已冻结 86-job 高维 held-out 草案 [`relation-auxiliary-only test draft`](mambatsv3_relation_auxiliary_highdim_test_release_draft_20260928.json)，
包含 6 个 validation selection、10 个候选测试和 70 个七类基线测试作业，confirmation seeds 为
`3311--3315`。草案已通过 `validate_only`，但 `execution.authorized=false`；在人工批准精确计划哈希前不启动任何测试作业。
当前草案 SHA-256：`6539139c30f9ceaba869e0d1afb8005623b653a86938273b697977a0e4d02614`。
对应严格 reader 为 [`analyze_mambatsv3_relation_auxiliary_highdim_test_release.py`](../../scripts/analyze_mambatsv3_relation_auxiliary_highdim_test_release.py)，
它会校验 launch sidecar、计划哈希、immutable source、全部 80 个 confirmation 坐标、streaming aggregation 和配对非劣界。

## 2026-09-28 Electricity / Traffic H96 AutoCon 机制矩阵：relation loss 小幅通过，prediction fusion 失败

为区分 AutoCon 的两个作用路径，新增冻结的 `18/18` GPU validation-only 精确配对矩阵：
Electricity/Traffic H96，各覆盖 `active_fusion`、`fusion_off` 与 `lambda_zero`，三颗 fresh seeds
`3301--3303`，同一当前源码快照 `1e3976666f8971abe3bf884f2635bbaf57f522d327201f183647544a666a63a5`。
严格读数见 [`机制矩阵 readout`](../../artifacts/exploratory/mambatsv3_highdim_autocon_mechanism_h96_validation_20260928/readout/readout.md)。

| Dataset | Contrast | Active MSE | Control MSE | Active vs control | Wins | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Electricity H96 | active vs fusion-off | `.160631` | `.158788` | `+1.160%` | `0/3` | FAIL |
| Electricity H96 | active vs lambda-zero | `.160631` | `.161240` | `-0.376%` | `2/3` | PASS |
| Traffic H96 | active vs fusion-off | `.377001` | `.372956` | `+1.081%` | `0/3` | FAIL |
| Traffic H96 | active vs lambda-zero | `.377001` | `.377883` | `-0.236%` | `2/3` | PASS |

该结果支持一个更窄但可复核的机制结论：在这两个高维域，训练期 relation loss 相对
`lambda=0` 有小幅验证收益；将 AutoCon 表示显式接入预测路径却分别退化约 `1.1%`，因此
不能宣称 representation fusion 已提升预测。该矩阵没有测试指标、预测数组或 streaming 工件；
relation loss 的收益仍属于 validation-only 机制证据，不是 held-out 因果结论。

## 2026-09-28 Solar H96 变量轴 Mamba：结构与表示融合通过，relation loss 因果门失败

为同时审计高维 Solar 的准确率、效率和 AutoCon 机制，将紧凑的
`NLinearVariableMambaAutoCon` 放入新的当前源码 `27/27` GPU validation-only 矩阵：三个 exact
paired variable-axis Mamba 臂（active、prediction-connected fusion-off、relation-null）、DLinear、
NLinear、PatchTST、TimesNet、iTransformer 与单独标识的 temporal `MambaTSv3AutoCon` context。所有
作业使用 Solar AL、`L=96,H=96,C=137`、同一 chronological split、8 epoch 预算、fresh seeds
`3281--3283` 和 `skip_test=true`；五个 phase validator 均通过，统一 source snapshot
`1e3976666f8971abe3bf884f2635bbaf57f522d327201f183647544a666a63a5`，没有测试工件。严格读数见
[`Solar variable-axis readout`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928/readout/readout.md)。

| Arm / baseline | Mean validation MSE | Active relative delta | Active wins |
| --- | ---: | ---: | ---: |
| Variable-axis Mamba, active | `.138581` | - | - |
| Same graph, fusion off | `.140531` | `-1.39%` | `3/3` |
| Same graph, relation-null | `.138323` | `+0.19%` | `1/3` |
| DLinear / NLinear | `.211087` / `.251588` | `-34.35%` / `-44.92%` | `3/3` / `3/3` |
| PatchTST / TimesNet | `.174179` / `.188706` | `-20.44%` / `-26.56%` | `3/3` / `3/3` |
| iTransformer | `.153402` | `-9.66%` | `3/3` |
| Temporal MambaTSv3AutoCon context | `.167120` | `-17.08%` | `3/3` |

这提供了两个必须分开的结论。第一，prediction-connected AutoCon representation fusion 是此
variable-axis Mamba 结构的稳定正消融：active 相对 fusion-off 三个 seed 全胜。第二，训练期 ACF
contrastive relation loss 不能被宣称为精度来源：active 相对 exact relation-null 平均退化 `.19%`，仅
`1/3` 胜出，与此前 Solar H336/H720 的负机制复核一致。它仍作为 AutoCon 的可审计训练项留在代码中，
但不作为该 Solar 成绩的因果解释，也不开放测试 release。

在固定 RTX 5090 D、FP32、device-resident `B=32,L=96,H=96,C=137`、25 warm-up/100 timed 的
prediction-plus-representation 测量中，active 为 `69,763` 参数、median `.2874 ms`、peak allocated
`31.86 MB`；同形状 iTransformer 为 `79,648` 参数、`.3172 ms`、`23.09 MB`，即 active 快约
`9.4%`，但显存较高。DLinear 为 `18,624` 参数、`.0679 ms`，因此候选不具备相对所有基线的速度优势。
relation-null 资源为 `.2939 ms`、同样 `31.86 MB`。这些只是在固定 shape 上相对 iTransformer/PatchTST
的 Pareto 证据，不是端到端吞吐或通用效率结论；原始资源记录见 [`active`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928/resource_active)、[`null`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928/resource_relation_null)、[`iTransformer`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928/resource_itransformer) 与 [`DLinear`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h96_mechanism_strong_baseline_20260928/resource_dlinear)。

## 2026-09-28 Solar H96 当前源码强基线矩阵：候选接近但不及 iTransformer，且无速度优势

为避免只保留交通与 Weather 的成功坐标，新增冻结的 `18/18` GPU validation-only SolarEnergy H96 矩阵：MambaTSv3AutoCon prediction-connected fusion 与 DLinear、NLinear、PatchTST、TimesNet、iTransformer，三颗 fresh seeds `3271--3273`。候选与 baseline phase 均通过严格 validator（分别 `3/3`、`15/15`），共享一个 source snapshot，所有配置为 `skip_test=true`；未产生或读取测试指标、预测或 streaming metrics。机器读数见 [`Solar strict readout`](../../artifacts/exploratory/mambatsv3_solar_strong_baseline_h96_validation_20260928/readout/readout.md)。

| Model | Mean validation MSE | Candidate relative delta | Candidate wins |
| --- | ---: | ---: | ---: |
| MambaTSv3AutoCon | `.167468` | - | - |
| DLinear | `.247239` | `-32.26%` | `3/3` |
| NLinear | `.247554` | `-32.35%` | `3/3` |
| PatchTST | `.175801` | `-4.74%` | `3/3` |
| TimesNet | `.204652` | `-18.17%` | `3/3` |
| iTransformer | `.162222` | `+3.23%` | `0/3` |

固定 RTX 5090 D、FP32、device-resident `B=32,L=96,H=96,C=137` 的 prediction-only 测量中，候选显式包含辅助 AutoCon representation 返回路径，median 为 `.5313 ms`、peak allocated `110.2 MB`；iTransformer 为 `.1903 ms`、`18.2 MB`。因此 Solar 不仅未通过相对最强 iTransformer 的精度门，也不通过速度/显存门。此坐标应作为论文的高维非交通失败边界；不可从候选对四个其他 baseline 的胜出选择性推出 Solar 优势，也不授权任何测试 release。资源记录见 [`candidate`](../../artifacts/exploratory/mambatsv3_solar_strong_baseline_h96_validation_20260928/resources_candidate) 与 [`iTransformer`](../../artifacts/exploratory/mambatsv3_solar_strong_baseline_h96_validation_20260928/resources_itransformer)。

## 2026-09-28 AGCRN 自适应图循环基线：资源预检历史（已由 2026-09-30 精度矩阵补齐）

为补齐固定图卷积、扩散循环与 Graph WaveNet 之外的自适应图循环比较器，新增本地 canonical-style
AGCRN：可学习节点嵌入生成自适应图，节点特异 Chebyshev 图卷积参数化 GRU gate，并以最后状态直接
输出多步预测。实现、注册与图基线合同测试通过（相关测试 `146 passed`）。

但在 METR-LA、RTX 5090 D、FP32、`B=32,L=96,H=12,C=207,d=24` 的 device-resident
prediction-only 预检中，AGCRN 中位延迟为 `17.955 ms`、参数 `72,258`，约为同形状
MambaTSv3AutoCon（`.3268 ms`）的 `55x`。其递归图卷积成本也超过已经停止的 DCRNN
(`15.645 ms`)；该预检本身不能作为精度结论。后续 2026-09-30 已完成同源 AGCRN validation-only
精度矩阵，顶部状态记录其与候选的全部 paired 结果；本节只保留资源负证据。资源记录见
[`AGCRN preflight`](../../artifacts/exploratory/agcrn_metr_resource_preflight_20260928/model_resources.json)。

## 2026-09-28 MambaTSv3 可逆仿射归一化筛选：局部通过，不升级为主模型

在现有 MambaTSv3AutoCon 的 MeanStd 窗口归一化后加入零初始化、每通道 scale/shift 的可逆仿射适配器。
该分支不增加 Mamba scan，初始状态与 exact v3 严格等价；18 个 GPU validation-only 作业全部成功，所有作业
`skip_test=true`，未生成测试指标或预测。严格读数见
[`reversible affine readout`](../../artifacts/exploratory/mambatsv3_revin_affine_screen_20260928/readout.md)。

| Dataset | Affine mean | Exact v3 mean | Relative change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 H96 | `.701327` | `.703421` | `-0.298%` | `3/3` |
| ETTh2 H96 | `.220173` | `.219776` | `+0.181%` | `1/3` |
| ETTm2 H96 | `.125942` | `.126053` | `-0.089%` | `3/3` |

按预注册门槛，ETTh1 与 ETTm2 通过、ETTh2 失败，因此只能作为轻量归一化消融或局部迁移证据，不能表述为普适精度提升或 AutoCon 独立因果效果。
固定 RTX 5090 D、FP32、`B=32,L=96,H=96,C=7` 的 prediction-plus-representation 测量中，exact v3 为
`0.3334 ms/61,016 params`，affine 为 `0.3467 ms/61,030 params`；仅增加 14 个参数，但延迟约增加 `3.99%`。
该结果适合放在消融/效率附录，不应替换已经冻结的 held-out 主结果。

## 2026-09-28 DCRNN-style 图扩散循环基线：早期单 seed 预检（已被 2026-09-30 完整矩阵替代）

为避免交通论文只比较卷积式图基线，新增了本地 canonical-style DCRNN：正/反图扩散嵌入
门控循环单元，并使用同一份 SHA-256 固定图契约。模型注册、图哈希/形状/梯度合同和 GPU
preflight 均通过（相关测试 `145 passed`）。但 METR-LA、RTX 5090 D、FP32、
`B=32,L=96,H=12,d=24` 的 device-resident prediction-only 前向中位延迟为 `15.645 ms`，
远高于同形状候选的 `.3268 ms`。完整 PEMS-BAY seed `3081` validation-only 作业又得到
DCRNN `.378567`，同一 seed 的候选为 `.293547`（DCRNN 高 `28.96%`），且 DCRNN 单作业
耗时 `1412.5 s`。该早期结果已被 2026-09-30 的优化版 DCRNN 完整 `12/12` validation-only 矩阵替代；本节只保留为优化前历史记录，不再作为当前三 seed baseline 表或论文精度结论。当前完整结果见顶部状态与对应 readout。

## 2026-09-28 GraphWaveNet held-out 扩展：草案已冻结，等待人工批准

GraphWaveNet 已在 METR-LA/PEMS-BAY H12 完成三 seed validation-only 比较，候选分别改善
`5.324%/4.635%`；为避免将验证集结果直接写入论文主表，已独立冻结 `46` 个作业的 held-out
草案 [`GraphWaveNet test draft`](metro_pemsbay_graphwavenet_mambatsv3_h12_test_release_draft_20260928.json)。
它包含 6 个候选 selection 预检、候选/GraphWaveNet 各 10 个 held-out 作业，以及 TimesNet/Mamba2
各 10 个 context 作业，所有 confirmation seed 固定为 `3101--3105`。GraphWaveNet 与 Mamba2 是
Bonferroni-adjusted `1.02` 非劣界的主要比较器；TimesNet 仅作描述性比较。严格 reader
[`analyze_metro_pemsbay_graphwavenet_mambatsv3_h12_test_release.py`](../../scripts/analyze_metro_pemsbay_graphwavenet_mambatsv3_h12_test_release.py)
会校验 phase、source snapshot、launch sidecar、每个 GraphWaveNet 图哈希和 streaming 指标。
该草案已通过 `--validate_only`，SHA-256 为
`60746691f98ad557d970619da68cd14c219714ee6259498834a7b7d909c2a71e`，但当前
`execution.authorized=false`；不会启动其中任何测试作业。

## 2026-09-28 METR-LA / PEMS-BAY Graph WaveNet 强图基线扩充：候选精度与前向资源均占优

为补齐交通实验中缺少的自适应图基线，新增了 Graph WaveNet-style 本地实现：因果膨胀时间门、
官方图契约的正/反扩散、二阶图卷积和可学习自适应邻接。它不是外部仓库的逐位复现，也不接收
AutoCon。冻结的 `12/12` GPU validation-only 作业覆盖 METR-LA/PEMS-BAY H12、候选
`MambaTSv3AutoCon` 和 GraphWaveNet、三颗 fresh seeds `3071--3073`；两个 phase validator
均 `valid=true`，所有作业 `skip_test=true`，没有测试工件。严格读数见
[`GraphWaveNet readout`](../../artifacts/exploratory/metro_pemsbay_graphwavenet_mambatsv3_h12_validation_20260928/readout/readout.md)。

| Dataset | Candidate MSE | GraphWaveNet MSE | Candidate vs GraphWaveNet | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| METR-LA H12 | `.275380` | `.290875` | `-5.324%` | `3/3` |
| PEMS-BAY H12 | `.297072` | `.311860` | `-4.635%` | `3/3` |

同形状 RTX 5090 D、FP32、`B=32,L=96,H=12` 的 prediction-only 测量（候选显式包含辅助表示
返回路径）为：METR-LA 候选 `.3268 ms`/`38.6 MB`，GraphWaveNet `6.4537 ms`/`1061.1 MB`；
PEMS-BAY 候选 `.3503 ms`/`56.7 MB`，GraphWaveNet `12.7647 ms`/`1666.7 MB`。这支持在这两个
固定交通 shape 上相对该强图比较器的资源优势，但不等同于端到端训练吞吐，也不能外推到所有
图模型或设备。资源记录见 [`candidate`](../../artifacts/exploratory/metro_pemsbay_graphwavenet_resource_candidate_20260928)
和 [`GraphWaveNet`](../../artifacts/exploratory/metro_pemsbay_graphwavenet_resource_baseline_20260928)。

## 2026-09-28 ETTh2/ETTm2 AutoCon 线性混合专家迁移：ETTm2 通过，ETTh2 失败

将 ETTh1 已冻结的 blend 配置（`bias=-4`、`AutoCon lambda=.001`、v3 backbone）原样迁移到
ETTh2/ETTm2，完成 `24/24` GPU validation-only 作业：blend、exact v3 control、RLinear 和
PatchTST，两个 phase validator 均 `valid=true`，单一 source snapshot，所有作业 `skip_test=true`。
严格读数见 [`blend transfer readout`](../../artifacts/exploratory/mambatsv3_ett2_ettm2_linear_blend_transfer_20260928/readout/readout.md)。

| Dataset | Blend | Exact v3 | Blend vs v3 | Strongest baseline | Blend vs strongest | Gate |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| ETTh2 H96 | `.227770` | `.232939` | `-2.22%`, 3/3 | RLinear `.211189` | `+7.85%` | FAIL |
| ETTm2 H96 | `.125751` | `.126747` | `-0.79%`, 2/3 | PatchTST `.125272` | `+0.38%` | PASS |

这说明 blend 的作用不是普适性修复：它能在 ETTm2 接近 PatchTST 并改善 v3，但无法解决 ETTh2
的强线性趋势缺口。ETTm2 结果可作为验证集 transfer evidence，ETTh2 必须作为失败边界保留；
两者都不能直接替代 held-out 测试主表。

在同一 RTX 5090 D、FP32、`B=32,L=96,H=96,C=7` 的 device-resident prediction-only
测量中，且将候选辅助表示返回路径计入，ETTm2 blend/v3 control/PatchTST/RLinear 的 median
latency 分别为 `.3851/.3492/.2869/.0660 ms`，参数量为 `76,928/70,687/46,080/9,326`。因此
ETTm2 的局部精度通过不构成速度优势：blend 比 control 慢约 `10.3%`，也慢于两个线性/patch
比较器。资源记录见 [`AutoCon resource`](../../artifacts/exploratory/mambatsv3_ett2_ettm2_linear_blend_resource_autocon_20260928)
和 [`baseline resource`](../../artifacts/exploratory/mambatsv3_ett2_ettm2_linear_blend_resource_baselines_20260928)。

## 2026-09-28 ETTh2/ETTm2 当前源码强基线扩展：完成但未通过普适精度门槛

为补齐 ETT 变体数据集，冻结了当前源码下的 `36/36` GPU validation-only 矩阵：
`MambaTSv3AutoCon`、RLinear、PatchTST、TimesNet、iTransformer、Mamba2，覆盖 ETTh2 和
ETTm2 H96、三颗 fresh seeds `3191--3193`。两个 phase validator 均 `valid=true`，共享
source snapshot，所有作业 `skip_test=true`，没有测试工件。严格读数见
[`ETTh2/ETTm2 readout`](../../artifacts/exploratory/mambatsv3_ett2_ettm2_strong_baseline_h96_20260928/readout/readout.md)。

| Dataset | Candidate | RLinear | PatchTST | TimesNet | iTransformer | Mamba2 | Candidate vs strongest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 H96 | `.234257` | `.211236` | `.213264` | `.250826` | `.221188` | `.234957` | `+10.90%` vs RLinear, 0/3 |
| ETTm2 H96 | `.127253` | `.126873` | `.125730` | `.132367` | `.130233` | `.134280` | `+1.21%` vs PatchTST, 0/3 |

候选在 ETTh2 仅优于 TimesNet、接近 Mamba2，但明显落后 RLinear/PatchTST/iTransformer；在
ETTm2 对 RLinear 基本持平，却仍落后 PatchTST。两套变体因此不能支持跨 ETT 普适优势或测试
释放，只作为新增数据集的完整边界证据。当前结果把论文定位进一步收窄为：候选对高维多变量域
和交通预测更具优势，低维 ETT 系列仍受线性/patch 模型限制。

## 2026-09-28 ETTh1 当前源码 AutoCon 线性混合专家：精度 gate 通过，速度需限定

针对上一节当前源码矩阵中 ETTh1 落后 PatchTST/RLinear 的失败边界，冻结了一个 12-job 小矩阵：
AutoCon 表示驱动的 MambaTSv3/persistence blend、无 blend 的精确 v3 控制、RLinear 和 PatchTST，
统一 ETTh1 H96、三颗 fresh seeds `3161--3163`、训练预算和数据切分。三个 phase validator 均
`valid=true`，共享 source snapshot，所有作业 `skip_test=true`，没有测试工件。严格读数见
[`ETTh1 blend readout`](../../artifacts/exploratory/mambatsv3_etth1_linear_blend_current_20260928/readout/readout.md)。

| Arm | Mean validation MSE | SD | Mean job sec |
| --- | ---: | ---: | ---: |
| AutoCon Mamba/persistence blend | `.692450` | `.001852` | `8.275` |
| Exact MambaTSv3AutoCon control | `.714235` | `.009820` | `8.393` |
| RLinear | `.699013` | `.002408` | `4.446` |
| PatchTST | `.684233` | `.003791` | `8.156` |

Blend 相对 exact v3 改善 `3.050%` 且 `3/3` seed 胜出；相对最强 PatchTST 仅差 `1.201%`，通过
预注册 accuracy gate。这只证明 ETTh1 的局部精度修复，不证明 AutoCon 的独立因果作用，也不
自动授权测试集发布。

同形状 RTX 5090 D、FP32、`B=32,L=96,H=96,C=7` 的 prediction-only 资源测量将候选的辅助
表示路径显式计入：blend median `0.3770 ms`、v3 control `0.3492 ms`、PatchTST `0.2803 ms`、
RLinear `0.0660 ms`。因此 blend 相对 v3 约慢 `8.0%`，不能宣称 ETTh1 上比所有基线更快；速度
优势仍限于已有高维交通形状中相对 TimesNet/Mamba2/STGCN 的固定设备测量。原始资源记录见
[`AutoCon resource`](../../artifacts/exploratory/mambatsv3_etth1_linear_blend_resource_autocon_20260928)
和 [`baseline resource`](../../artifacts/exploratory/mambatsv3_etth1_linear_blend_resource_baselines_20260928)。

基于该 validation gate，已冻结一个但未授权的 23-job ETTh1 H96 held-out 草案，包含 3 个
validation reproducibility jobs、5 个 blend test、5 个 exact-v3 control test 和 10 个
RLinear/PatchTST test jobs。草案 [`ETTh1 test draft`](mambatsv3_etth1_linear_blend_test_release_draft_20260928.json)
当前 `execution.authorized=false`；在人工批准精确 SHA-256
`6e44a927ef88b31fdae79912d50825c5fae98cadb52ae7edc14972dd89ea12b2` 前不会启动任何测试作业。

## 2026-09-28 当前源码强基线配对矩阵：72/72 验证作业完成

为避免把新增线性基线与历史不同源码快照混合，冻结了同一当前源码、同一数据划分、同一优化器和
同一三颗随机种子的 H96 配对矩阵：`MambaTSv3AutoCon` 候选，以及 `RLinear`、`PatchTST`、
`TimesNet`、`iTransformer`、`Mamba2` 五个基线，覆盖 ETTh1、Weather、Electricity、Traffic。
候选使用 prediction-connected AutoCon 表示融合；基线不接收候选专属开关。候选 `12/12`、基线
`60/60` 作业全部成功，两个 phase validator 均为 `valid=true`，共用 source snapshot，所有作业
均 `skip_test=true`，没有读取或生成测试指标和预测数组。冻结计划、授权和严格读取器分别见
[`plan`](mambatsv3_rlinear_strong_baseline_h96_validation_20260928.json)、
[`launch`](mambatsv3_rlinear_strong_baseline_h96_validation_20260928_launch_authorization.json) 和
[`readout`](../../artifacts/exploratory/mambatsv3_rlinear_strong_baseline_h96_validation_20260928/readout/readout.md)。

| Dataset | Candidate | RLinear | PatchTST | TimesNet | iTransformer | Mamba2 | Candidate vs strongest baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 H96 | `.710786` | `.700129` | `.682066` | `.761902` | `.684833` | `.740456` | `+4.21%` vs PatchTST, 0/3 |
| Weather H96 | `.445807` | `.537030` | `.493160` | `.464808` | `.481362` | `.474707` | `-4.09%` vs TimesNet, 3/3 |
| Electricity H96 | `.162095` | `.201675` | `.192983` | `.170309` | `.187114` | `.176276` | `-4.82%` vs TimesNet, 3/3 |
| Traffic H96 | `.373581` | `.506212` | `.412443` | `.430078` | `.401491` | `.488010` | `-6.95%` vs iTransformer, 3/3 |

### 同日补齐线性与同源码八臂高维矩阵

为把 DLinear/NLinear 纳入高维结论而不混合早期源码快照，随后完成了两份新的 validation-only
冻结计划：候选、DLinear、NLinear 共 `18/18` jobs，随后 RLinear、PatchTST、TimesNet、
iTransformer、Mamba2 共 `30/30` jobs。两个 artifact root 的 immutable source hash 都为
`1e3976666f8971abe3bf884f2635bbaf57f522d327201f183647544a666a63a5`。专用 reader 在聚合前
核验 phase validity、48 条精确坐标、全部 paired seeds、candidate/baseline isolation 和零测试工件。

| Dataset | Candidate | Strongest baseline | Candidate vs strongest | Seven baseline paired cells |
| --- | ---: | --- | ---: | ---: |
| Electricity H96 | `.161140` | TimesNet `.168558` | `-4.393%` | `7/7` cells, each `3/3` wins |
| Traffic H96 | `.378149` | iTransformer `.401117` | `-5.727%` | `7/7` cells, each `3/3` wins |

完整 machine readout 见 [`full high-dimensional matrix`](../../artifacts/exploratory/mambatsv3_highdim_full_baseline_matrix_h96_validation_20260928/readout/readout.md)。这是当前最完整的 Electricity/Traffic H96 validation 筛选证据，但没有产生 held-out 指标，不能提升为测试主表或 relation-loss 因果主张。

对应的 RTX 5090 D、FP32、`B32,L96,H96` prediction-only 资源束显示候选在 Electricity/Traffic
为 `.9231/2.3551 ms`，快于 PatchTST `1.2594/3.9465 ms` 和 TimesNet `2.7163/2.8633 ms`，但慢于
DLinear、NLinear、RLinear、iTransformer 和 Mamba2。候选模型的 prediction-connected fusion 已在实际
预测路径中执行；relation loss 不属于推理路径且未计入。完整边界和参数量见
[`resource readout`](mambatsv3_highdim_full_baseline_matrix_h96_resource_20260928.md)。因此可报告两类
非线性 comparator 的固定 shape 延迟优势，不能报告普适速度优势。

这组当前源码证据把模型定位得更准确：候选在 Weather、Electricity、Traffic 上相对每个注册基线
均为 `3/3` seed 胜出，改善范围 `4.09%--26.20%`；但 ETTh1 同时落后 RLinear `1.52%`、
PatchTST `4.21%` 和 iTransformer `3.79%`。因此 ETTh1 不能被纳入候选的普适优越性主张，
下一轮架构工作应针对低维趋势/线性竞争单独设计并重新验证，而不是选择性删除该失败坐标。上述
数字是 validation-only 结果，不能替代 `artifacts/paper_final/` 中的 held-out 主表。

## 2026-09-28 RLinear 当前源码基线扩展：18/18 验证作业完成

为避免把低维任务上的线性强基线遗漏在比较表之外，新增独立 `RLinear` 实现：每个输入窗口
执行带可学习通道仿射的 RevIN，再用共享时间线性映射预测 horizon；没有 AutoCon 分支，因而
不会把归一化收益误报成对比学习收益。该结构是对 RLinear 高层方法的独立重实现，不复制外部
代码。计划覆盖 ETTh1、Weather、Exchange-Rate、Electricity、Traffic 的 H96，以及 ILI 的
H24；三颗 fresh seeds `3131--3133`，共 `18/18` GPU validation-only 作业。两个 phase
validator 均 `valid=true`，同一 source snapshot，所有作业 `skip_test=true`，无测试指标或
预测工件。冻结计划、授权与严格读出分别见 [`plan`](rlinear_current_baseline_extension_20260928.json)、
[`launch`](rlinear_current_baseline_extension_20260928_launch_authorization.json) 和
[`readout`](../../artifacts/exploratory/rlinear_current_baseline_extension_20260928/readout/readout.md)。

| Dataset | Horizon | RLinear mean validation MSE | SD | Mean job sec |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 96 | `.701193` | `.002000` | `4.695` |
| Weather | 96 | `.536918` | `.000075` | `16.009` |
| Exchange-Rate | 96 | `.267169` | `.000174` | `3.975` |
| Electricity | 96 | `.201683` | `.000001` | `31.589` |
| Traffic | 96 | `.506214` | `.000030` | `37.298` |
| ILI | 24 | `.264098` | `.001487` | `2.565` |

这些数字是当前源码快照下的 RLinear 基线参考，不能与历史 Mamba 候选做跨快照 paired delta；
它们的主要作用是补齐论文的归一化线性下界，并提醒当前 Mamba+AutoCon 候选在低维/分布变化域
必须同时面对强线性模型。ETTh1 H96 的同形状独立 GPU prediction-only 资源为 `9,326` 参数、
median `.0674 ms`、peak allocated `8.90 MB`，只能作为 RLinear 的资源记录，不能据此宣称候选
普遍更快。资源工件见 [`RLinear ETTh1 resource`](../../artifacts/exploratory/rlinear_resource_etth1_h96_20260928)。

## 2026-09-28 METR-LA / PEMS-BAY 同形状资源复测：候选延迟占优，显存并非全面占优

为使交通实验的效率表不再混用不同日期、不同 warm-up 次数的读数，重新以同一资源工具在
RTX 5090 D、FP32、device-resident synthetic `B=32,L=96,H=12` 下测量候选、STGCN、
TimesNet 和 Mamba2；每个配置为 30 次 warm-up 和 100 次 timed forwards。候选固定为
`MambaTSv3AutoCon` 的 prediction-connected fusion，并显式保留 AutoCon 辅助表示返回；
比较器均只计模型默认预测前向。没有加载训练、验证或测试样本，故这些是资源测量而非新的
预测精度结果。

| 数据集 | 模型 | 参数量 | median / p95 forward ms | peak allocated MB |
| --- | --- | ---: | ---: | ---: |
| METR-LA | MambaTSv3AutoCon | 33,198 | `.355 / .363` | `56.9` |
| METR-LA | Mamba2 | 35,848 | `.678 / .688` | `31.1` |
| METR-LA | TimesNet | 690,099 | `2.005 / 2.018` | `45.2` |
| METR-LA | STGCN | 32,004 | `3.661 / 3.689` | `1039.2` |
| PEMS-BAY | MambaTSv3AutoCon | 43,818 | `.414 / .426` | `85.2` |
| PEMS-BAY | Mamba2 | 50,126 | `.673 / .689` | `34.7` |
| PEMS-BAY | TimesNet | 701,545 | `2.183 / 2.198` | `49.6` |
| PEMS-BAY | STGCN | 32,004 | `6.198 / 6.222` | `1622.6` |

这支持一个形状限定、可进入论文效率表的表述：在这两个 H12 交通输入形状上，候选即使
保留 AutoCon 表示，仍快于本地 STGCN-style、TimesNet 与 Mamba2 实现，且参数少于后两者。
它不支持“对所有基线都更快”或“显存更低”：候选显存高于 Mamba2/TimesNet，只显著低于
STGCN。所有原始硬件、软件版本、输入协议与 source hash 在资源工件中：
[`METR-LA`](../../artifacts/exploratory/resource_bundle_metr_h12_20260928)、
[`PEMS-BAY`](../../artifacts/exploratory/resource_bundle_pemsbay_h12_20260928)。
这些资源结果不改变 STGCN/TimesNet/Mamba2 测试精度尚未发布的边界。

## 2026-09-28 METR-LA / PEMS-BAY 强基线测试扩展：已冻结草案，等待人工批准

现有 validation-only 结果显示，候选在 METR-LA/PEMS-BAY H12 上相对本地 STGCN-style、
TimesNet 和 Mamba2 分别取得 `-5.488%/-19.346%`、`-27.711%/-36.677%` 和
`-7.018%/-14.747%` 的验证 MSE 差异；这些数字不能替代测试结果。为补齐论文中最重要的
图时空和现代 Mamba 比较，已写入一个未授权的 46-job held-out 草案：
[`test-release draft`](metro_pemsbay_mambatsv3_strong_baseline_extension_test_release_draft_20260928.json)。

草案包含 6 个候选 validation 预检查、10 个候选测试、10 个 STGCN 测试、20 个
TimesNet/Mamba2 测试，统一 H12、`L=96`、5 个 paired seeds、streaming metric 和
数据/图 SHA-256 契约。候选才启用冻结的 AutoCon prediction-connected fusion；STGCN
只接收 adjacency，TimesNet/Mamba2 不接收候选专属开关。STGCN 与 Mamba2 被注册为主要非劣
比较器，采用 Bonferroni-adjusted one-sided interval；TimesNet 仅作描述性扩展。

该计划当前 `execution.authorized=false`、`training_started=false`，测试输出目录不存在。
在人工批准精确计划 SHA-256 `f0e6eb3990dd76bae429cd1c6a2bf9ce8d956c6c6c1f953170b6c4ef51565319` 前，
不会启动任何 held-out job。

## 2026-09-28 Solar H720 forecast lag-pair 独立复核：机制 gate 失败，停止该损失分支

Solar H336 上，训练期 signed ACF forecast lag-pair consistency 在三个 exact paired
seed 中均降低 validation MSE；原协议明确规定下一步必须在不变的 architecture、weight、
data split、optimizer 和 epoch budget 下进行独立 horizon 复核。新 H720 screen 固定
`NLinearVariableMambaAutoCon`、prediction-connected representation fusion、原始训练集
ACF sidecar、encoder relation objective、`AutoCon_lambda=.0001`，唯一处理差异为 forecast
pair term `off/0` 对 `on/.01`。两 arm 均为三 fresh seed `2041--2043`，所有六项 GPU
作业为 `skip_test=true`，无测试工件；两个 phase validator 均为 `valid=true`。

| Horizon | Term off MSE | Term on MSE | On vs off | paired wins | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Solar H336（既有） | `.153843` | `.153613` | `-0.1495%` | `3/3` | 局部通过 |
| Solar H720（独立） | `.153945` | `.154117` | `+0.1114%` | `1/3` | 失败 |

H720 treatment 的每个 seed 都具有正的 pair loss、pair count（约 `14.38`）和 active-anchor
fraction（约 `89.6%`），故失败不是由监督项未执行造成。完成作业的平均 wall time 为 control
约 `48.51 s`、treatment 约 `53.94 s`，训练增加约 `11.2%`；该项只在训练中计算，不改变预测
前向图，不能将此训练时间用于推理速度主张。严格只读读数见 [`H720 readout`](../../artifacts/exploratory/nlinear_variable_mamba_solar_h720_forecast_pair_confirmation_20260928/readout/readout.md)，冻结计划见 [`plan`](nlinear_variable_mamba_solar_h720_forecast_pair_confirmation_20260928.json)。

按预注册停止规则，该 forecast-pair loss 不会再做 weight search、架构联动调节或 held-out
test release。Solar 的高精度 variable-axis Mamba 结果可保留为一个 architecture baseline，
但不能作为满足“AutoCon 训练机制在多 horizon 稳定改善预测”的论文候选。

## 2026-09-28 STGCN 图时空基线补充：验证集候选胜出，但不改变测试发布边界

此前 METR-LA/PEMS-BAY 的 held-out 表是明确的非图设置，缺少图时空参照。新增
`models/STGCN.py`：纯 PyTorch 的 STGCN-style baseline，采用时间 GLU、一次官方
row-normalized directed graph propagation 和共享 horizon head；图从已有安全 NPZ
契约读取并校验 SHA-256。它参考 STGCN 的公开图时空卷积思想，但不宣称逐位复现公开实现。

冻结的 12/12 GPU validation-only 作业覆盖 METR-LA/PEMS-BAY、H12、fresh seeds
`3071--3073`；candidate 和 STGCN 使用相同时间切分、`L=96`、优化器、batch、epochs
与 early stopping，且所有作业 `skip_test=true`。两个 phase validator 均 `valid=true`，
共享源码快照，不存在测试工件。机器读数见 [`STGCN validation readout`](../../artifacts/exploratory/metro_pemsbay_stgcn_mambatsv3_h12_validation_20260928/readout/readout.md)。

| 数据集 | Candidate validation MSE | STGCN validation MSE | Candidate vs STGCN | paired wins | 候选/STGCN job sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| METR-LA | `.275380` | `.291373` | `-5.488%` | `3/3` | `29.4/139.8` |
| PEMS-BAY | `.297072` | `.368358` | `-19.346%` | `3/3` | `89.4/283.5` |

同一 RTX 5090 D、FP32、device-resident `B=32,L=96,H=12` 的独立 prediction-only
资源读数，candidate 为 `.419/.554 ms`（METR-LA/PEMS-BAY），STGCN 为
`3.665/6.183 ms`；candidate 参数量 `31,422/42,042`，STGCN 都为 `32,004`。
这是命名图基线的形状限定资源对比，不是通用速度结论。STGCN 的训练耗时也更高，不能把
candidate 的完整 job time 解释成硬件无关吞吐量。

该矩阵弥补了交通实验缺少图时空 baseline 的问题，但它仍是 validation-only，不能修改此前
METR-LA/PEMS-BAY held-out 发布；如要把“优于图时空 baseline”写进主表，必须先注册新的
独立 held-out 协议。

## 2026-09-28 AutoCon representation-gated bounded Mamba/linear blend：ETTh1 局部通过，跨域停止

针对 MambaTSv3 在低维 ETTh1、Exchange-Rate 和 ILI 坐标落后线性/patch 基线的问题，新增了一个有界凸 blend：AutoCon 的共享表示按预测 horizon 产生 gate，在 MambaTSv3 prediction-connected forecast 与线性专家之间插值。第一版直接复用 v3 已有 persistence projection，避免第二次 Mamba scan；模型、计划器、资源工具和严格契约测试均已覆盖。

冻结的 `45/45` GPU validation-only 矩阵覆盖 ETTh1/H96、Exchange-Rate/H96、ILI/H24，candidate、exact v3 control，以及 DLinear/NLinear/PatchTST 三个外部基线，fresh seeds `3011--3013`。三个 phase validator 均为 `valid=true`，同一 source snapshot，测试执行关闭且没有测试工件。严格读数见 [`bounded blend readout`](../../artifacts/exploratory/mambatsv3_repr_linear_blend_lowdim_h96_validation_20260928/readout/readout.md)，冻结计划见 [`plan`](mambatsv3_repr_linear_blend_lowdim_h96_validation_20260928.json) 和 [`launch authorization`](mambatsv3_repr_linear_blend_lowdim_h96_validation_20260928_launch_authorization.json)。

| 数据集 | Blend MSE | Exact v3 MSE | 相对 v3 | 最强外部基线 | 相对最强基线 | Gate |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| ETTh1 H96 | `.688516` | `.696795` | `-1.188%`, 3/3 | PatchTST `.687299` | `+0.177%` | 通过 |
| Exchange-Rate H96 | `.271632` | `.283192` | `-4.082%`, 3/3 | NLinear `.260772` | `+4.165%` | 失败 |
| ILI H24 | `.239056` | `.250316` | `-4.498%`, 3/3 | PatchTST `.227639` | `+5.015%` | 失败 |

因此该 blend 只在 ETTh1 达到“改善 exact v3 且不超过最强外部基线 2%”的局部 gate，未达到预注册的至少 2/3 数据集继续条件；不注册 held-out test，也不调整 blend bias 或 AutoCon lambda。

同形状 RTX 5090 D prediction-plus-representation 资源复测中，复用 persistence 的 blend median `.372320 ms`，同命令 v3 `.343440 ms`，慢约 `8.4%`，在效率预算内。进一步的真正低秩 NLinear blend 在 rank-16/rank-4 下分别慢约 `17.9%/17.6%`，均在训练前停止；资源工件见 [`rank-16`](../../artifacts/exploratory/mambatsv3_repr_nlinear_blend_etth1_resource_20260928/model_resources.csv) 和 [`rank-4`](../../artifacts/exploratory/mambatsv3_repr_nlinear_blend_rank4_etth1_resource_20260928/model_resources.csv)。这说明当前失败域需要更强的线性表达力，但增加线性投影又破坏了 v3 的速度预算，暂不继续扩大架构搜索。

## 2026-09-28 METR-LA / PEMS-BAY TimesNet 与 Mamba2 基线扩充：验证集通过，测试仍冻结

为补齐交通实验中缺少的现代基线，冻结了已有的 `MambaTSv3AutoCon` prediction-connected
fusion 配置，并新增 TimesNet 与 Mamba2。两数据集均使用 H12、`seq_len=96`、相同的
时间顺序切分、训练预算和 `2971--2973` 三个 fresh seeds；全部作业使用 GPU，且
`skip_test=true`。18/18 作业完成，两个严格 phase validator 均 `valid=true`，共享一个
源码快照，没有生成测试工件。机器读数见
[`TimesNet/Mamba2 validation readout`](../../artifacts/exploratory/metro_pemsbay_mambatsv3_timesnet_mamba2_h12_validation_20260928/readout/readout.md)。

| 数据集 | 候选验证 MSE | TimesNet | Mamba2 |
| --- | ---: | ---: | ---: |
| METR-LA | 0.272997 | 0.377712 (`-27.711%`) | 0.293620 (`-7.018%`) |
| PEMS-BAY | 0.295474 | 0.466643 (`-36.677%`) | 0.346596 (`-14.747%`) |

候选在四个配对单元中均为 `3/3` seed 胜出。这一结果增强了 METR-LA/PEMS-BAY 的验证集
基线覆盖，但不能修改此前 held-out 测试发布，也不能将验证 MSE 直接写入论文主表。
下一步若要形成新的测试主表，必须重新注册冻结的 TimesNet/Mamba2 对照与独立测试预算；
在此之前，本扩充只作为 validation-only 支撑证据。

## 2026-09-27 MambaTSv12 two-Delta screen：精度候选通过部分坐标，速度目标失败

参考公开 `ms-Mamba` 的高层思想，本仓库独立实现了 `MambaTSv12`：同一 patch-token 序列经过两个不同 Delta 初始化范围的 Mamba-1 分支，分支状态以初始等权、可学习 softmax 融合；融合状态同时进入预测头和 AutoCon 表征头。没有复制外部仓库代码，也没有将外部结果当成本地结果。实现与合同测试见 [`models/MambaTS.py`](../../models/MambaTS.py)、[`models/MambaTSv12.py`](../../models/MambaTSv12.py)。

冻结的 `54/54` GPU validation-only screen 覆盖 ETTh1 H96、Weather H96、PEMS07 H12，包含 v12、同预算 v3、PatchTST、TimeMixer、iTransformer、Mamba2；六个 phase validator 均 `valid=true`，没有测试工件。严格读数见 [`v12 readout`](../../artifacts/exploratory/mambatsv12_multiscale_autocon_h96_validation_20260927/readout/readout.md)。v12 相对 v3 的平均验证 MSE 变化为 ETTh1 `-0.554%`（`2/3`）、Weather `-0.294%`（`2/3`）、PEMS07 `-0.956%`（`3/3`）。相对强外部基线，Weather H96 和 PEMS07 H12 通过；ETTh1 H96 仍比 PatchTST/iTransformer 高约 `3.325%/3.559%`，因此不能宣称跨域普适改进。

独立 RTX 5090 D 资源测量（PEMS07 `B=32,L=96,C=883,H=12`、同宽度）显示 v12 `0.882 ms`、`160.7 MB`，一尺度 v3 `0.605 ms`、`153.6 MB`；v12 约慢 `46%`。因此 v12 只能作为多尺度精度候选和消融，不能作为“比基线计算更快”的论文主模型。其 AutoCon prediction-fusion 开关还需要单独 off/on 配对验证，之后才决定是否注册 held-out 测试。

随后完成的 `18/18` v12 fusion off/on validation control 也全部通过严格校验。PEMS07 H12 的 on/off 为 `.058982/.060022`，改善 `1.729%`、`3/3` 胜出，满足局部归因 gate；ETTh1 H96 为 `.710640/.710185`（退化 `0.066%`、`1/3`），Weather H96 为 `.447635/.447085`（退化 `0.126%`、`2/3`），均未通过预注册 gate。完整读数见 [`v12 fusion control`](../../artifacts/exploratory/mambatsv12_autocon_fusion_exact_control_20260927/readout/readout.md)。因此 v12 不能声称 AutoCon fusion 在跨域稳定有效；若进入论文，只能报告 PEMS07 的局部机制结果，并同时报告失败坐标。

## 2026-09-28 MambaTSv13 one-scan light multi-scale screen: PEMS07 local pass, cross-domain and speed gates fail

`MambaTSv13` keeps the MambaTSv3 single Mamba-1 scan, then applies kernel-3 and kernel-5 depthwise filters to the observed patch states. A zero-initialized ReZero residual sends the learned multi-scale state to both the forecast head and the AutoCon representation head. This is an independent lightweight design, not a copy of external code. The model, run-time contract, runner registry and resource tool are covered by `167` focused tests. During the first launch a stale run-time fusion whitelist rejected v13 before training; that invalid batch is retained separately for audit. The whitelist was fixed, regression-tested, and the registered rerun used new output roots and a new immutable sidecar.

The rerun completed `54/54` validation-only GPU jobs on ETTh1 H96, Weather H96 and PEMS07 H12, against same-budget MambaTSv3AutoCon, PatchTST, TimeMixer, iTransformer and Mamba2, with three fresh paired seeds. All six phase validators have `valid=true`, one common source snapshot, and no test artifacts. The strict machine readout is [`v13 readout`](../../artifacts/exploratory/mambatsv13_lite_multiscale_autocon_h96_validation_20260928_rerun/readout/readout.md). v13 improves only PEMS07 against v3: `.059748/.060025` (`-0.461%`, `2/3` wins), while beating each registered external baseline there. It is worse than v3 on ETTh1 (`+0.686%`, `1/3`) and Weather (`+1.344%`, `1/3`); ETTh1 is also `+3.391%/+2.277%/+3.766%` above PatchTST/TimeMixer/iTransformer. This violates the registered v3 and external-baseline stopping rules, so v13 must not receive a held-out test release or be claimed as a cross-domain architecture improvement.

The independent same-shape RTX 5090 D benchmark on PEMS07 (`B=32,L=96,C=883,H=12`) reports v13 `66,529` parameters, `0.679 ms` mean forward time and `153.6 MB`; v3 has `66,398` parameters, `0.605 ms` and the same memory. Thus v13 is about `12.2%` slower than v3, so its PEMS validation gain cannot support an efficiency claim. It may be retained only as a local ablation/negative-transfer result, not as the paper's main model.

## 2026-09-28 MambaTSv14 coarse-patch local-expert resource preflight: stop before training

Inspired by the global/local separation used in SST-style forecasting architectures, `MambaTSv14` was implemented as a deliberately constrained alternative: a single shared Mamba-1 scan over four full-coverage `24x24` patches, plus two zero-initialized depthwise observed-window filters. Both the forecast and AutoCon representation consume this same filtered state; the filters read only the input window, so there is no target leakage. The first forward is functionally nested in the corresponding v3 path because the local ReZero scale starts at zero. Model, CLI, plan runner and resource registry coverage increased to `170` focused tests.

The resource gate rejects this route before any accuracy run. On RTX 5090 D with PEMS07 `B=32,L=96,C=883,H=12`, FP32, device-resident prediction plus the auxiliary representation path, v14 has `73,465` parameters, `0.766 ms` mean (`0.766 ms` median) and `195.2 MB` allocated. An immediate same-command v3 replication has `66,398` parameters, `0.629 ms` mean (`0.629 ms` median) and `153.8 MB`. Thus v14 is `21.7%` slower and uses more memory; it cannot satisfy the required efficiency objective even if a later validation MSE improved. No training, validation, or test job is authorized for v14. Resource artifacts are [`v14`](../../artifacts/exploratory/mambatsv14_pems07_resource_20260928/model_resources.csv) and [`same-command v3`](../../artifacts/exploratory/mambatsv14_v3_same_command_resource_20260928/model_resources.csv).

## 2026-09-28 MambaTSv15 time-conditioned single-scan preflight: stop before training

借鉴 TimePro 类结构把时间上下文作为状态条件的高层思想，`MambaTSv15` 在 v3 的一个
Mamba-1 scan 前，对历史 `x_mark_enc` 的 calendar feature 按 patch 池化并投影到 patch
token。一个零初始化 ReZero 标量控制注入，故初始化时含/不含时间标记的预测严格一致；
条件化后的编码状态仍是预测头与 AutoCon 表示头唯一的共同输入。它不读未来目标，也不增加
第二个 selective scan。模型、runtime、计划器、资源注册与 zero-scale 性质均有契约测试。

资源门槛仍然否决该候选。在 RTX 5090 D、ETTh1 `B=32,L=96,C=7,H=96`、FP32、相同
`d=24` 设置下，含 AutoCon representation 的 v15 有 `35,656` 个参数，mean/median
前向为 `.3894/.3859 ms`；立即同命令复测 v3 为 `35,511` 参数和 `.3390/.3382 ms`。
v15 的中位延迟约高 `14.1%`，超过预设的 `<=10%` 上限。因为新增仅 145 参数，主要成本是
小 token shape 上额外的 marker pooling/projection 调度，而非模型容量。故没有授权 v15
的训练、验证或测试。资源工件为 [`v15`](../../artifacts/exploratory/mambatsv15_etth1_resource_20260928/model_resources.csv) 与 [`same-command v3`](../../artifacts/exploratory/mambatsv15_v3_same_command_etth1_resource_20260928/model_resources.csv)。

## 2026-09-28 MambaTSv16 anchor-conditioned preflight: stop before training

为消除 v15 的滑动窗口池化开销，v16 只通过预注册 patch 起始索引读取历史
`x_mark_enc`，再使用相同零初始化 ReZero 条件投影。它仍然是单 Mamba-1 scan，且条件化
state 同时供预测和 AutoCon 表示使用。契约测试确认 patch 起始索引、输出形状和零初始化
梯度路径有效。

但在同一 RTX 5090 D、ETTh1 `B=32,L=96,C=7,H=96`、FP32、包含 AutoCon 表示路径的
资源协议下，v16 为 `35,656` 参数、median `.3759 ms`；同命令 v3 为 `35,511` 参数、
median `.3407 ms`。v16 仍慢约 `10.3%`，略超过预设 `<=10%` 门槛。因此不启动 v16 训练、
验证或测试，避免把微小资源差异扩展成无界架构搜索。资源工件为 [`v16`](../../artifacts/exploratory/mambatsv16_etth1_resource_20260928/model_resources.csv) 与 [`same-command v3`](../../artifacts/exploratory/mambatsv16_v3_same_command_etth1_resource_20260928/model_resources.csv)。

## 2026-09-27 Cross-domain H96 fusion audit: three branches stop, fusion mechanism not general

Two fresh-seed validation-only matrices have completed on RTX 5090 D. The first contains `96/96` jobs: MambaTSv3 AutoCon prediction-connected fusion against DLinear, NLinear, PatchTST, TimeMixer, iTransformer, Mamba2, and TimeMachine-style MambaTM1 on ETTh1/Weather/Exchange-Rate H96 and ILI H48. All four strict phase validators are valid; no test artifact was generated. The strict external-baseline readout is [`cross-domain fusion validation`](../../artifacts/exploratory/mambatsv3_autocon_fusion_crossdomain_h96_validation_20260927/readout/readout.md).

The candidate is strong only on Weather H96: MSE `.446518`, improving `18.127%/17.965%/9.545%/3.424%/7.471%/6.971%/7.459%` relative to DLinear/NLinear/PatchTST/TimeMixer/iTransformer/Mamba2/MambaTM1, with `3/3` wins in every cell. It fails the registered external baseline gate elsewhere: ETTh1 `.706591` is `+3.170%` above PatchTST `.684875`; Exchange-Rate `.286903` is `+10.502%` above PatchTST `.259636`; ILI `.241198` is `+14.471%` above NLinear `.210715`. These three branches are stopped: no test release, post-hoc width/lambda tuning, or selective reporting.

A second `24/24` job exact off/on control then isolated only `mamba_ts_repr_forecast_fusion`, with identical MambaTSv3 AutoCon model, lambda, data, seed, and budget. All eight phase validators are valid and no test artifact exists. The strict paired readout is [`exact fusion control`](../../artifacts/exploratory/mambatsv3_autocon_fusion_exact_crossdomain_h96_control_20260927/readout/readout.md): ETTh1 on/off `.711011/.711958` (`-0.131%`, `2/3`, gate true); Weather `.449469/.446304` (`+0.724%`, `1/3`, gate false); Exchange `.282449/.280192` (`+0.811%`, `1/3`, gate false); ILI `.237189/.238308` (`-0.438%`, `2/3`, but one seed `+2.586%`, gate false). Hence the direct representation-fusion switch is not a stable cross-domain contributor. The traffic H12 fusion result remains a restricted traffic finding, not a general mechanism claim.

The current defensible paper nucleus is therefore the held-out four-PEMS H12 and METR-LA/PEMS-BAY H12 evidence plus Weather-only confirmation, with explicit negative transfer evidence. It is not yet a high-level general forecasting paper: a new architecture must be selected before observing test metrics, pass exact mechanism controls and strong baseline gates on more than one domain, and separately meet named-comparator speed gates.

## 2026-09-27 Official-style MambaTS VPT local-search audit: stop

The local MambaTS v11 route was upgraded from its previous greedy decoder to a `python-tsp` local-search ATSP path following the public MambaTS VPT structure: random per-sample variable paths during training, detached prediction-loss transition costs, train-derived route recovery at evaluation, and restored channel order. The public MambaTS implementation identifies `models/MambaTS.py`, `utils/masking.py`, and its VAST mixer as the relevant architecture locations; this repository implements the route logic against its installed Mamba-1 API, so it is an adaptation rather than a bitwise reproduction.

The frozen `42/42` GPU validation-only preflight contains exact v11 greedy and local-search arms, a current v3 fusion reference, and PatchTST/TimeMixer/iTransformer/Mamba2 baselines on ETTh1/Weather H96, three fresh paired seeds each. All four phase validators are valid; all six v11 jobs per solver have nonzero transition coverage and count; no test artifact exists. Strict audit is [`v11 local-search readout`](../../artifacts/exploratory/mambatsv11_official_vpt_ls_h96_preflight_20260927/readout/readout.md).

Local search versus greedy is ETTh1 `.716456/.716827` (`-0.051%`, only `1/3` wins) and Weather `.448503/.447457` (`+0.233%`, `0/3` wins). ETTh1 local search is still `+4.389%` above PatchTST `.686335` and `+2.214%` above TimeMixer `.700938`; Weather is `-9.071%` versus PatchTST and `-2.231%` versus TimeMixer, but the solver itself is worse than its greedy control. The registered continuation rule required local search to improve greedy on at least `2/3` pairs in both datasets and remain within 2% of the strongest reference. It fails; stop VPT solver/lambda/width/route search and do not release any held-out test.

## 2026-09-27 MambaTSv3 prediction-connected AutoCon fusion H12：held-out confirmation通过

冻结的 `76/76` job release 已在 RTX 5090 D 完成：selection 6、fusion candidate 10、同源基线 60；三个 phase validator 均 `valid=true`，`failed_jobs=0`。候选和基线使用相同 METR-LA/PEMS-BAY H12 数据切分、训练预算和五个 fresh paired seeds（2921--2925）。严格读数见 [`held-out fusion readout`](../../artifacts/paper_final/metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927/readout/readout.md)，计划与审计入口见 [`test release plan`](metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_plan_20260927.json) 和 [`launch authorization`](metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927_launch_authorization.json)。

候选的 held-out streaming scaled MSE 为 METR-LA `.389206`、PEMS-BAY `.310211`。相对最强同源 plain `MambaTSv3`，METR-LA 改善 `2.257%`（`.389206` vs `.398196`），PEMS-BAY 改善 `1.816%`（`.310211` vs `.315941`）；两者均为 `5/5` paired wins，一侧 95% paired upper ratio 分别为 `0.9814` 和 `0.9864`，低于预注册的 `1.02` 非劣界。相对 TimeMixer 分别改善 `6.674%` 和 `16.579%`，同样通过非劣界；相对全部六类注册基线的 12 个数据集×模型单元均通过，且每个单元均为 `5/5` 候选胜出。

这使当前结果可以进入论文的主实验候选表：它支持“在非图 METR-LA/PEMS-BAY、H12、五 seed held-out 设定下，prediction-connected AutoCon fusion 不劣于并小幅优于同源 MambaTSv3”的范围明确结论。与此前 validation confirmation 相比，MambaTSv3 相对改善由 METR-LA `3.455%`、PEMS-BAY `1.554%` 变为 held-out `2.257%`、`1.816%`，方向一致；因此可以写作独立测试复现，而不是只报告选择集结果。

训练 elapsed time 不能当作推理速度：candidate 平均训练耗时约为 METR-LA `27.7 s`、PEMS-BAY `102.1 s`，高于 plain MambaTSv3 的 `22.1 s`、`52.5 s`。已有公平 representation-path benchmark 仅显示 fusion 相对 off 的前向表示路径开销约 `4.4%/5.2%`；论文中可以报告“小幅表示路径开销”，不能声称端到端训练或普遍推理更快。结果仍限于 H12、两个非图交通数据集和当前源码快照，不能外推到长 horizon、图建模或所有时间序列。

## 2026-09-27 MambaTSv3 prediction-connected AutoCon fusion H12：METR-LA/PEMS-BAY 配对验证通过，等待同源基线确认

在 Variable-Mamba selector 的 PEMS-BAY 主干失败后，切换到已在交通数据上更强的单次 shared-scan `MambaTSv3AutoCon`，只增加零初始化的 prediction-connected AutoCon representation residual；它不增加第二次 Mamba scan，且 AutoCon 对比表示与 forecast 使用同一编码状态。新鲜 seed `2811--2813` 的 12 个 validation-only GPU 作业全部完成，两个 phase validator 均 `valid=true`，没有测试工件。

METR-LA fusion off/on 平均最佳验证 MSE 为 `.284513/.275437`，改善 `3.190%`、`3/3` 配对胜出；PEMS-BAY 为 `.305198/.299235`，改善 `1.954%`、`3/3` 胜出。fusion-on 的 checkpoint scale 在 6/6 作业非零，说明不是开关未生效。严格机器读数见 [`MambaTSv3 fusion H12 readout`](../../artifacts/exploratory/metro_pemsbay_mambatsv3_autocon_repr_fusion_h12_screen_20260927/readout/readout.md)。

这是 validation gate 的通过，不是论文结果：当前还没有在同一源码快照下复跑 DLinear、NLinear、PatchTST、TimeMixer、iTransformer、plain MambaTSv3 的完整对照，也没有资源/速度结论，更没有开放测试集。下一步固定 fusion 结构与超参数，完成同源强基线和 RTX 5090 D exact-shape prediction-only 资源确认；若外部 gate 与效率预算同时通过，才注册 held-out confirmation。

## 2026-09-27 Variable-Mamba ACF selector H12：selector 归因通过，但 PEMS-BAY 主干失败，停止释放测试

新授权的 48 个 validation-only GPU 作业已在 RTX 5090 D 完成：selector off/on 各 6 个作业，外部基线 36 个作业；三个 phase validator 均为 `valid=true`，`failed_jobs=0`，没有测试指标或预测文件。严格 reader 见 [`selector H12 readout`](../../artifacts/exploratory/metro_pemsbay_nlinear_variable_mamba_acf_selector_h12_screen_20260927/readout/readout.md)。

METR-LA 的 selector off/on 平均验证 MSE 为 `.286945/.285681`，改善 `0.440%`、`3/3` 配对胜出；它与最强 MambaTSv3 基线 `.285636` 的差距仅 `+0.016%`，通过本轮外部 gate。PEMS-BAY 的 off/on 为 `.365535/.364161`，改善 `0.376%`、`3/3` 胜出，但最强 MambaTSv3 为 `.305239`，候选落后 `19.304%`，外部 gate 失败。因此 selector 机制本身有稳定的小幅正向归因，但 `NLinearVariableMamba` 主干不能承担高维交通预测；不注册该路线 held-out test，也不允许选择性报告 METR-LA 作为通用结果。

该结果直接改变下一步架构选择：保留 AutoCon 表示学习和 train-only ACF 证据，但将主干切换到已经在 PEMS-BAY H12 验证通过的单次 shared-scan `MambaTSv3AutoCon`，再把 ACF 记忆作为零初始化、有界的预测门接入，先做小规模 validation-only 一因素筛选。任何新分支必须同时超过 exact MambaTSv3 control，并在两个交通数据集相对最强基线不超过预注册 2% 差距，否则停止。

## 2026-09-27 Variable-Mamba ACF selector：METR-LA/PEMS-BAY H12 方案已冻结，等待本轮 GPU 授权

针对 Traffic/ETTh1 H96 中 selector 的正向信号，已冻结新的跨数据集 H12 validation-only 方案
[`metro_pemsbay_nlinear_variable_mamba_acf_selector_h12_screen_plan_20260927.json`](metro_pemsbay_nlinear_variable_mamba_acf_selector_h12_screen_plan_20260927.json)。方案包含 selector off/on 精确配对各 6 个作业，以及 DLinear、NLinear、PatchTST、TimeMixer、iTransformer、MambaTSv3 六类外部基线共 36 个作业，总计 48 个；所有作业 `skip_test=true`，使用 fresh seeds `2711--2713`，不产生测试指标。

运行器已通过 48-job 严格校验和 dry-run；METR-LA/PEMS-BAY CSV 与邻接文件 SHA-256 均与协议一致。候选的 CUDA 资源预检已在 RTX 5090 D 完成：`NLinearVariableMambaAutoCon`（B2、L96、H12、C207、prediction-only）参数 `11,263`，median latency `0.399 ms`，p95 `0.427 ms`，显存峰值约 `9.93 MB`，`benchmark_status=ok`。这只是模型初始化/前向资源证据，不是训练精度结果。

当前计划仍标记为 `authorized=false`，因此没有擅自启动新的 48-job 训练矩阵。此前图时空计划已完成但 graph-on 在 METR-LA/PEMS-BAY 均未通过配对 gate；本轮应先由人工确认是否批准上述新协议，再决定是否占用 GPU 进行完整计算。

## 2026-09-27 图时空 MambaTSv3：实现与资源预检完成，训练仍待人工批准

METR-LA 和 PEMS-BAY 现已各自绑定安全的 `adjacency.npz` 图合同：CSV 传感器列顺序与官方有向邻接逐项一致，METR-LA 为 207 节点、1,722 边，PEMS-BAY 为 325 节点、2,694 边。新 `MambaTSv3AutoCon` 候选仅在 observed seasonal window 上做一次稀疏、行归一化邻居传播，随后复用既有单次 Mamba scan；预测头与 AutoCon representation 仍消费同一编码状态。图门为 ReZero 初始化，因此 graph-off/on 在初始化时严格嵌套。`GraphDLinear` 同时加入为本地拓扑基线，使用同一图传播但不使用 Mamba 或 AutoCon；它不是 DCRNN 或 Graph WaveNet 的复现。

RTX 5090 D 的 B32、L96、H12、prediction-only 预检显示，图 Mamba 在 METR-LA/PEMS-BAY 的均值延迟为 `.297/.339 ms`，GraphDLinear 为 `.124/.161 ms`。图 Mamba 仍快于已记录的 PatchTST `.364/.497 ms` 和 TimeMixer `.986/1.330 ms`，但慢于 iTransformer `.222/.330 ms`，也慢于线性基线。因此当前只能保留“快于命名的部分非线性基线”的可检验假设，不能声称普遍速度领先。代码、图合同、计划命令和资源工具的全量回归为 `726 passed`。

随后使用真实 `mamba_ssm` CUDA（不是测试用的 IdentityMamba）完成了实现 smoke：METR-LA `B=2,C=207` 与 PEMS-BAY `B=2,C=325` 均在 RTX 5090 D 上输出 H12 预测和 `[B,96,48]` AutoCon representation，loss 有限，图 ReZero 门梯度非零（分别 `0.00602`、`0.00987`）。可复现脚本为 [`smoke_graph_mambatsv3_gpu.py`](../../scripts/smoke_graph_mambatsv3_gpu.py)，结果保存在 `artifacts/exploratory/metro_pemsbay_graph_spatial_preflight_20260927/real_cuda_graph_mambatsv3_smoke.json`。这只证明 CUDA 前向/反向和 AutoCon 共享路径有效，不包含真实数据训练或精度结论。

已冻结 [`graph-spatial validation plan`](metro_pemsbay_graph_spatial_mambatsv3_autocon_screen_plan_20260927.json)：两数据集、H12、三 seed、48 个 validation-only 作业，包含 Mamba 图 off/on 精确配对、GraphDLinear、DLinear/PatchTST/TimeMixer/iTransformer/plain MambaTSv3。运行器同时验证 CSV 哈希、NPZ 图哈希、schema、节点数、边数和每个传感器的 CSV-to-graph 顺序；篡改 METR-LA 的 1,722 条边登记会被直接拒绝。该计划已由用户批准并完成 GPU 运行；人工复核入口仍是 [`protocol`](metro_pemsbay_graph_spatial_mambatsv3_autocon_screen_protocol_20260927.md)，当前结论见下方图时空 H12 validation screen，测试集没有开放。

为批准后的读数增加了严格 reader [`analyze_graph_spatial_mambatsv3_screen.py`](../../scripts/analyze_graph_spatial_mambatsv3_screen.py)：它要求四个 phase 完整覆盖、`phase_validation.json` 有效、source snapshot 一致、无测试工件，并逐作业复核图路径/哈希，然后自动输出 graph-off/on 配对、GraphDLinear 归因、外部最强基线和预注册 gate。当前图计划尚无训练 artifact，因此 reader 还没有可报告的精度数值。

进一步的真实数据 CUDA smoke 只读取两个数据集的一个训练分割 batch，并执行一次
`MSE + .001 * AutoCon` 反向传播：METR-LA 与 PEMS-BAY 的 AutoCon loss 均非零
（`1.9064/1.9993`），单独对图 ReZero 门的梯度绝对值为
`.01602/.05926`，一步 Adam 更新后门均离开零。脚本
[`smoke_graph_mambatsv3_gpu.py`](../../scripts/smoke_graph_mambatsv3_gpu.py) 与
[`real-batch artifact`](../../artifacts/exploratory/metro_pemsbay_graph_spatial_preflight_20260927/real_cuda_graph_mambatsv3_real_batch_smoke.json)
证明图传播、预测头和 AutoCon representation 的同路梯度存在；它不跑 validation/test，
不保存 checkpoint，也绝不能作为精度读数。

### 图时空 H12 validation screen：两数据集均失败，停止（2026-09-27）

经用户批准后，冻结的 48 个 GPU validation-only 作业全部完成；graph-off/on、
GraphDLinear 和 30 个外部基线作业的严格 phase validator 均为 `valid=true`，
没有 `metrics.npy`、`original_metrics.npy` 或 streaming test 工件。严格 reader 的
48 行读数为：METR-LA graph-on/off `0.283260/0.282439`（`+0.290%`，`0/3` paired
wins），PEMS-BAY `0.306108/0.305724`（`+0.126%`，`0/3`）。这直接违反两个数据集都
至少 `-1%`、至少 `2/3` 胜出、且无退化的预注册 graph gate。

图 Mamba 均低于本地 GraphDLinear（`.304434/.441182`），也在最强外部同预算
plain MambaTSv3（`.286150/.307255`）的 2% 以内，但这不能抵消相对于严格 graph-off
对照的成对失败。因而停止图尺度、归一化、lambda 或 horizon 的事后搜索，且不释放
held-out test。完整机器读数为
[`graph screen readout`](../../artifacts/exploratory/metro_pemsbay_graph_spatial_mambatsv3_autocon_screen_20260927/readout/readout.md)。

## 2026-09-27 MambaTSv3 全局低秩通道状态混合器：METR-LA 三 seed 全退化，停止

为修复冻结 MambaTSv3AutoCon 在 METR-LA/H12 落后 PatchTST/TimeMixer 的跨变量缺口，在相同 `d24`、单 shared Mamba-1 scan、`patch_len=stride=24`、`AutoCon_lambda=.001` 配置上新增了仅读取当前观察窗口均值/RMS的 rank-4 全局通道状态 FiLM 模块。它不启动第二次变量轴 Mamba scan；其调制后的 shared state 同时进入预测头与 AutoCon representation head，且 ReZero 初始化使 on/off 首次前向严格嵌套。`36/36` 个 validation-only 作业成功，三个 phase validator 均有效，未生成任一 `metrics.npy` 测试文件。

METR-LA on/off 的平均最佳 validation MSE 为 `.288517/.281822`，即 treatment `+2.376%` 且 `0/3` paired seed 胜出；逐 seed 均退化，违反预先登记的 `>=1%` 改善、`>=2/3` 胜出与单 seed 不超过 `2%` 退化门槛。PEMS-BAY 为 `.304363/.305140`，即 `-0.255%`、`3/3` 胜出，但不能抵消 METR-LA 的失败。checkpoint 的 mixer ReZero scale 在两个数据集均离开零，确认该负结果不是开关未生效。资源预检中，METR-LA/PEMS-BAY control/on 为 `.249/.342 ms` 与 `.263/.336 ms`；on 仍快于同形状 PatchTST/TimeMixer，但慢于 plain MambaTSv3。因此停止该 mixer 的 rank/lambda/horizon 搜索与 held-out test，不得将其写为速度或精度改进。完整人工复核见 [`MambaTSv3 global channel mixer H12 screen`](metro_pemsbay_mambatsv3_global_channel_mixer_screen_20260927.md)。

GitHub 结构复核入口为 [`github_mamba_architecture_audit_20260923.md`](github_mamba_architecture_audit_20260923.md)：TimeMachine、S-Mamba/TimesMamba、MambaTS 均已有本地对应且已完成边界筛选；ms-Mamba 的内部 Delta 多尺度扫描保留为有严格精度/速度门槛的后续候选，而非立即扩展。

公开实现复核以官方仓库为准：MambaTS 的官方实现明确把 `models/MambaTS.py` 作为架构入口，并把 VPT/VAST 作为独立机制；Bi-Mamba4TS 官方实现采用双向 Mamba 与 series-relation-aware tokenization；S-Mamba、TimeMachine、DMamba 和 ms-Mamba 均有公开代码入口。本仓库的 `NLinearVariableMamba` 是针对当前 CUDA 环境和资源门槛的本地轻量候选，不等同于这些论文的官方复现；论文中必须分别标注“官方基线”和“本地结构改造”。

## 2026-09-27 Variable-Mamba ACF selector：Traffic H96 强基线筛选通过，但测试仍冻结

在前一轮基线命令因候选 relation-lag 参数错误继承而全部于参数校验时退出后，已用新 seed `2241--2243` 重跑一个隔离的 21-job validation-only replacement。候选 `NLinearVariableMambaAutoCon` 保持共享 NLinear 预测基座、一次 variable-axis Mamba-1 scan、prediction-connected AutoCon representation 和 train-only ACF memory；selector on/off 的唯一图差异是零初始化 rank-2 horizon + rank-1 channel memory selector。所有外部基线明确关闭 candidate-only relation 配置，所有作业均 `skip_test=true`。

严格 reader 对三 phase 的 21/21 完整作业、source snapshot、训练完成标记、无测试产物、train-only ACF sidecar、可访问 relation lag 和 selector checkpoint 均通过。selector off/on 的平均最佳验证 MSE 为 `.397010/.392141`，即 `-1.226%`，三个配对 seed 全部改善。相同短预算下最强外部基线为 iTransformer `.399676`，候选低 `1.885%`；PatchTST/TimesNet/NLinear/DLinear 的均值分别为 `.407419/.434185/.506532/.512167`。

这只说明该轻量 selector 在 Traffic/H96、三 seed、八 epoch 的冻结筛选中同时通过归因对照和同预算外部基线门槛。它不能证明 AutoCon relation loss 的独立效应，不能推出普适精度或速度优势，也不允许触及测试集。下一步必须保持结构和超参数冻结，在独立数据集或新 horizon 完成 validation confirmation，再决定是否注册 held-out test。完整人工复核入口见 [`Traffic H96 strong-baseline selector screen`](nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_20260927.md)。

同一冻结 selector 已在 ETTh1/H96、fresh seed `2251--2253` 的 21-job 同源强基线确认中再次通过。selector off/on 为 `.651147/.647344`，即 `-0.584%`，三个 seed 全部改善；外部最强基线为 DLinear `.650119`，候选低 `0.427%`。所有作业 `skip_test=true`，严格 reader 检查了完整覆盖、无测试产物、ACF sidecar、可访问 lag 和非零 selector checkpoint。ETTh1 的 exact-shape prediction-only 前向为 selector-on `.441 ms`、TimesNet `2.954 ms`、PatchTST `.291 ms`、iTransformer `.166 ms`、DLinear `.068 ms`、NLinear `.031 ms`，所以只能说其快于 TimesNet，不能声称低维速度优势。完整复核见 [`ETTh1 H96 strong-baseline selector confirmation`](nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_20260927.md)。

该 selector 在第三个预先登记的 Electricity/H96、fresh seed `2261--2263` 矩阵中触发停止规则。selector off/on 为 `.185067/.184745`，仅 `-0.174%` 但仍为 `3/3` 配对改善，非零 selector scale 和 train-only ACF sidecar 均通过；然而同预算 TimesNet 为 `.171796`，候选高 `7.537%`。其余 iTransformer/PatchTST/DLinear/NLinear 为 `.195824/.196422/.200868/.202235`。因此不能选择性使用 Traffic 与 ETTh1 的正向读数来声称跨数据集性能；停止此 selector 的调参、扩展和 held-out test release。完整负确认见 [`Electricity H96 strong-baseline selector confirmation`](nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_20260927.md)。

## 2026-09-27 MambaACF rank-4 channel-state mixer：Electricity H96 分支通过，relation 增益未确认

在 selector 于 Electricity 失败后，使用同一当前源码快照和 Electricity 321 通道合同，完成了新的 27-job validation-only factorial：五个外部基线（DLinear、NLinear、PatchTST、TimesNet、iTransformer）各三个 seed，以及 MambaACFAutoCon 的 mixer-off/on × relation-null/active 四个三 seed 单元。所有作业 `skip_test=true`、无测试指标、train-only raw ACF sidecar 完整且 source snapshot 为 `7755385b...e28a4`。

新模块是 `LowRankGlobalChannelStateMixer(rank=4)`：它读取观测窗口已有的共享 hidden summary，通过低秩 FiLM residual 调制共享状态；预测头和 AutoCon 表示消费同一个 post-mixer state，不增加第二条 Mamba scan，也不构造 dense `C x C` 矩阵。rank-4 预检只增加 `453` 个可训练参数；B32、T96、C321 的 prediction-only median 为 off `.883 ms`、on `.860 ms`，只能说明没有明显资源退化，不能形成速度优越性结论。

严格 reader 结果为：mixer-on + `lambda=.001` 的主候选平均验证 MSE `.155544`，相对 mixer-off active `.157668` 改善 `1.347%`，`2/3` 配对 seed 胜出；最强外部基线 TimesNet 为 `.169221`，候选低 `8.082%`。mixer-null 下 on 相对 off 改善 `1.349%`，说明主要信号来自窗口条件状态混合器。active relation 相对同一 mixer 的 null 为 `+0.012%`、`0/3` 胜出，但落在预注册的 `<=1%` 非劣门槛内。因此当前结论是：该 MambaACF channel-state mixer 候选通过本轮分支与外部基线门槛，可以注册独立多数据集确认；AutoCon relation 只能作为非劣训练目标保留，不能写成独立精度贡献。

完整人工复核入口见 [`MambaACF channel-state mixer Electricity H96`](mamba_acf_channel_mixer_electricity_h96_20260927.md)，机器读数见 [`strict readout`](mamba_acf_channel_mixer_electricity_h96_20260927_readout/readout.md)。

### 2026-09-27 跨数据集确认：高维局部通过，ETTh1 失败，停止通用分支

同一冻结的 rank-4 channel-state mixer 随后在 ETTh1/H96（seed `2281--2283`）和 Traffic/H96（seed `2291--2293`）各完成 27-job 独立 factorial，并保留 DLinear、NLinear、PatchTST、TimesNet、iTransformer 五个外部基线。Traffic 主候选为 `.396263`，相对 exact mixer-off active `.396838` 改善 `0.145%`、`2/3` seed 胜出，并低于最强 iTransformer `.413117` 的 `4.080%`；Electricity 仍为 `.155544`，低于 TimesNet `8.082%`。但 ETTh1 主候选 `.717523` 虽相对 mixer-off active `.718419` 改善 `0.125%`、`2/3` seed 胜出，仍高于最强 iTransformer `.686635` 的 `4.498%`，触发外部基线停止门槛。

relation active 相对完全相同 mixer-on null 的变化为 Electricity `+0.012%`、Traffic `+0.013%`、ETTh1 `+0.007%`，对应 seed 胜出为 `0/3`、`1/3`、`0/3`。它在三个坐标皆低于预注册 1% 非劣界，但没有独立预测增益。rank-4 mixer 每个 shape 仅多 `453` 参数，却在 ETTh1/Traffic prediction-only 前向增加约 `5.8%/5.4%` 延迟；因此既不具跨数据集精度非劣，也不满足通用速度主张。停止该 mixer 的 rank/lambda 搜索、长 horizon 与 held-out test；仅将 Electricity/Traffic 保留为高维、validation-only 机制证据。完整汇总见 [`three-dataset channel-state mixer confirmation`](mamba_acf_channel_mixer_multidataset_h96_20260927.md)。

## 2026-09-27 MambaTSv3 d24 高维 H12 迁移：Electricity/Traffic 均失败

为检验 PEMS H12 上最强的轻量 `MambaTSv3AutoCon` 配置能否迁移，冻结 `d_model=24,d_ff=48,patch_len=stride=24`、同一训练合同和 fresh seed `2301--2303`，完成 Electricity/Traffic 各 7 个外部基线、MambaTSv3 null 与 active 的 54-job validation-only 矩阵。54/54 作业成功，source snapshot 一致，未生成测试指标。

Electricity active/null 均值为 `.109479/.109536`，active 仅改善 `0.052%`、`1/3` seed 胜出；最强外部基线为 TimeMixer `.106383`，候选落后 `2.910%`。Traffic active/null 为 `.341378/.341351`，active 退化 `0.008%`、`0/3` seed 胜出；最强基线为 iTransformer `.324793`，候选落后 `5.106%`。两个数据集均违反预注册的 2% external gate，且 AutoCon 没有达到“均值改善且至少 2/3 胜出”的独立效应标准。因此停止该 d24 配置在高维数据集上的扩展和测试释放，不能把 PEMS H12 的局部结果写成通用高维结论。完整人工复核见 [`MambaTSv3 d24 high-dimensional transfer`](mambatsv3_d24_highdim_h12_transfer_20260927.md)。

## 2026-09-27 BiPatchMamba + AutoCon H12 筛选：结构局部有效，但加性表示修正失败

参考公开 Bi-Mamba+ 的双向 patch/variable-token 结构，新增 `BiPatchMamba` 与 `BiPatchMambaAutoCon` 的四数据集 72-job validation-only screen。ETTh1、Electricity、Traffic、Weather 各使用 DLinear、NLinear、PatchTST、TimeMixer、iTransformer、MambaTSv3 六个外部基线，以及 plain/null/active 三个 BiPatch 臂；72/72 作业成功，source snapshot 一致且没有测试指标。

Active MSE 分别为 `.320870/.122824/.343654/.248507`。相对 exact null 的变化为 `-0.493%/-2.311%/-1.192%/+0.829%`，但 active 相对 plain 的变化为 `+1.377%/+13.467%/+5.904%/+12.924%`，四个数据集全部退化。当前结论是：双向 patch 主干在 Traffic/Weather plain 行接近外部最强基线，但 AutoCon 的直接加性 forecast correction 破坏了预测路径；不能释放测试集或声称 BiPatch+AutoCon 通过精度门。下一步固定主干，改为有界零初始化 representation gate，再做小规模筛选。完整复核见 [`BiPatchMamba H12 transfer`](bipatchmamba_autocon_transfer_h12_screen_20260927.md)。

### 2026-09-27 有界 representation gate：消除大幅退化，但仍未胜过 plain

在不改变 BiPatchMamba 主干、训练预算、数据划分和 AutoCon 超参的条件下，新 24-job validation-only pilot 用有界、零初始化的乘性 gate 替代直接加性 correction。严格 reader 通过 24/24 作业、三臂 source snapshot 一致、正确的 gate 开关和无测试工件检查。active/null/plain 的平均验证 MSE 为 ETTh1 `.313071/.312793/.313147`、Electricity `.110758/.111020/.108040`、Traffic `.327827/.328119/.323963`、Weather `.223865/.224445/.219406`。

active gate 相对 null 在四个数据集都满足 1% 非劣界，且 Electricity、Weather 均为 2/2 seed 改善，说明 gate 比原加性 correction 稳定得多；但它相对 fresh plain 在 Electricity/Traffic/Weather 仍为 `+2.516%/+1.193%/+2.032%`，ETTh1 只有 `-0.024%`。因此该分支没有通过“结构优于 plain”的最低门槛，不释放测试集，不再进行 post-hoc gate/lambda 调参。它仅保留为一项负机制证据：限制表示修正幅度能降低伤害，但没有制造可投稿的跨数据集精度收益。完整人工复核见 [`BiPatchMamba representation-gate pilot`](bipatchmamba_repr_gate_pilot_h12_20260927.md)。

## 2026-09-27 独立交通数据扩展：MambaTSv3 AutoCon H12 通过 validation gate，待冻结测试

新增 METR-LA（207 个传感器、34,272 个 5 分钟步）和 PEMS-BAY（325 个传感器、52,116 个 5 分钟步）的 timestamp-first CSV numeric loader、SHA-256 数据合同及真实起点的日历时间特征后，完成冻结 d24 MambaTSv3AutoCon 的 32-job validation-only transfer。DLinear、NLinear、PatchTST、TimeMixer、iTransformer、plain MambaTSv3、exact lambda-zero null 和 active `lambda=.001` 均在每数据集两个 fresh seed 下完成；32/32 成功，source snapshot 一致，未生成测试指标。

METR-LA active/null/plain 为 `.301471/.301482/.302461`，强基线 TimeMixer `.296230`，active 相对其 `+1.769%`，仍处于 2% 门槛内。PEMS-BAY active/null/plain 为 `.328045/.327987/.329126`，active 比 plain 最强行低 `0.329%`。active 对 null 分别为 `-0.004%/+0.018%`，均满足 1% 非劣，但都只有 1/2 seed 改善，因此不能宣称 AutoCon 的独立精度贡献。

同形状 prediction-only 中位延迟的 active 为 `.244/.266 ms`，快于两数据集的 PatchTST `.363/.497 ms` 和 TimeMixer `.980/1.330 ms`；METR-LA 慢于 iTransformer，且两处均慢于线性基线。该 validation pass 只允许注册五个 fresh seed 的冻结 held-out confirmation，不能直接作为论文主表。完整人工复核见 [`METR-LA / PEMS-BAY transfer`](metro_pemsbay_mambatsv3_autocon_h12_screen_20260927.md)。

### Held-out confirmation result

随后完成冻结的 `76/76` job held-out release（selection 6、candidate 10、baselines 60），三 phase validator 全部通过。PEMS-BAY candidate 的 mean streaming scaled MSE 为 `.344955`，相对 PatchTST/TimeMixer/plain MambaTSv3/iTransformer 分别改善 `14.602%/7.999%/0.665%/3.840%`，全部通过一侧 1.02 非劣界。METR-LA candidate 为 `.432489`，相对 PatchTST `+1.248%`、TimeMixer `+2.900%`，对应 upper ratios `1.0203/1.0394`，主精度门失败；只对 DLinear、NLinear、iTransformer 和 plain MambaTSv3 非劣。

冻结后资源复测的 candidate median latency 为 METR-LA `.245 ms`、PEMS-BAY `.268 ms`；它在两处都快于 PatchTST/TimeMixer，但 METR-LA 慢于 iTransformer，且两处均不快于 DLinear/NLinear。由此当前证据只能支持“PEMS-BAY H12 的短期、非图交通预测结果”，不能支持两数据集通用主张或高水平论文完成。完整 release 见 [`METR-LA / PEMS-BAY held-out release`](metro_pemsbay_mambatsv3_d24_test_release_20260927.md)。

## 2026-09-26 PEMS H12 held-out release：四数据集精度与部分基线速度证据成立

冻结的 d24 full-coverage `MambaTSv3AutoCon` 在 PEMS03/04/07/08、H12、fresh
seed `2131--2135` 的 held-out streaming test 已完成。selection (`12/12`)、candidate
(`20/20`) 与七类 baseline (`140/140`) phase 均 `valid=true`，没有失败作业。候选的测试
MSE 为 `.069361/.080189/.063274/.083224`，对注册的主要基线 PatchTST/Mamba2 在每个数据集
均 `5/5` seed 胜出，且一侧 95% paired upper MSE ratio 均低于 `1.02`；对 TimeMixer 的
均值改善为 `3.14%/2.57%/5.47%/2.70%`。

同源 RTX 5090 D、FP32、B32、device-resident prediction-only 资源复测显示候选延迟为
`.303/.285/.530/.269 ms`，在四个 PEMS shape 都快于 PatchTST、TimesNet、TimeMixer 与
Mamba2；PEMS03/04/07 也快于 iTransformer。但候选始终慢于 DLinear/NLinear，且 PEMS08 慢于
iTransformer，故结论只能是“对命名的非线性比较器有更快前向”，绝不能写作通用速度优势。

这是仓库目前最强的可审计结果，但仍只是 PEMS H12；同一 d24 路线已在 H96 四数据集确认失败，
而 AutoCon active/null 的独立 effect 小且不稳定。因此它不足以支撑高水平通用时间序列论文，
只能构成一个范围明确的短期交通预测主结果。完整证据见
[`PEMS d24 held-out release`](pems_mambatsv3_d24_test_release_20260926.md)。

## 2026-09-27 MambaTS v7 H336/H720 fresh-seed screen：按停止规则关闭

此前完成但尚未汇总的 v7 长预测批次现已按原注册协议读出：ETTh1、ETTm1、Electricity、
Traffic，H336/H720，各 3 个 fresh seed；外部对照为同一源码快照下的 DLinear、PatchTST、
TimesNet、iTransformer。120 个作业全部完成，测试集没有生成或读取指标。

| 数据集 | H336 相对最强基线 | H720 相对最强基线 |
| --- | ---: | ---: |
| ETTh1 | `+6.395%` | `+2.667%` |
| ETTm1 | `-2.850%` | `-2.114%` |
| Electricity | `-1.697%` | `+3.616%` |
| Traffic | `-8.676%` | `-7.957%` |

正值表示 v7 的验证 MSE 更差。由于 8 个单元中有 3 个单元退化，达到预注册的“3 个单元
失败即停止”条件，停止 v7 的调参、额外基线和测试释放。v7 的 AutoCon/ACF telemetry
在所有作业均有限且非零，但这不能把跨数据集结果归因于 AutoCon；同样，v7 在 ETTh1、
ETTm1 和 Traffic 的长预测单元通常比最强基线更慢，不能宣称通用效率优势。完整汇总见
[`MambaTS v7 fresh-seed readout`](../../artifacts/exploratory/mambatsv7_multihorizon_fresh_seed_confirmation_20260925/readout/readout.md)。

截至目前，长预测方向没有一条同时满足“跨数据集精度不劣、AutoCon effect 可归因、推理更快”
的候选；下一步应先重新定义论文问题或引入新的、预注册的机制，而不是在 v7 上继续事后搜索。

## 2026-09-27 MambaTS v9 decomposed single-scan：ETTh1 H336 预检停止

基于公开 DMamba 的“季节项 Mamba、趋势项轻量预测”原则，新增本地 v9：仅对已观察的
MeanStd 窗口做移动平均分解，季节残差走一次现有 shared Mamba scan，趋势走共享线性头；
同一时间索引 AutoCon representation 通过 ReZero horizon gate 直接调制两路预测的差。
它不复现 DMamba 的 variable-direction encoder 或 MLP stack，也没有额外 Mamba scan、
注意力或未来信息。

ETTh1/H336、fresh seed `2191--2193` 的冻结三臂 validation-only 预检已完成：v9 null/
active/PatchTST 的平均最佳验证 MSE 为 `1.330833/1.330459/1.279328`。active 对 exact
null 的均值变化只有 `-0.028%`，仅 `1/3` seed 改善；相对 PatchTST 为 `+3.997%`，`0/3`
seed 胜出。九个作业零失败、跨 phase source hash 一致，且没有测试指标。由于同时违反
AutoCon 归因和精度门槛，停止 v9 的任何事后 kernel/lambda/宽度调参、资源测量、数据集扩展
与测试释放。完整记录见
[`MambaTS v9 ETTh1 H336 readout`](mambatsv9_decomposition_etth1_h336_preflight_readout_20260927.md)。

## 2026-09-27 MambaTS v10 VAST-lite：ETTh1 H336 预检停止

复核官方 MambaTS 实现后发现，官方 VPT/VAST 不只是训练期随机变量置换：它还从训练期的
transition-cost adjacency matrix 构造推理期变量顺序。为保持本项目的单扫描效率约束，v10 没有
宣称复现官方 ATSP solver，而是从训练 split 的 signed-ACF profile（固定 lag `24,48`）构造一个
确定性通道相似性顺序；训练和验证均在该顺序下执行同一条 shared Mamba scan，并在预测前还原原始
通道次序。AutoCon representation 仍通过零初始化 forecast residual 直接进入预测。

修正过一次“候选专属 flag 错传 PatchTST”导致的无效初始批次后，冻结 replacement plan 的
ETTh1/H336、fresh seed `2211--2213` validation-only 三臂试验完整完成。v10 null/active/
PatchTST 的平均最优验证 MSE 为 `1.333936/1.337945/1.276616`；active 相对 null 为 `+0.301%`
且 `0/3` seed 胜出，相对 PatchTST 为 `+4.804%` 且 `0/3` seed 胜出。9/9 作业成功、源快照一致、
所有作业 `skip_test=true`，严格审计没有发现测试指标；六个 v10 ACF sidecar 也完全一致。

因此同时违反 AutoCon 归因门槛与精度门槛，停止 v10 的排序/lag/lambda 调参、资源测量、
多数据集扩展和测试释放。该负结果不评价官方 learned VPT/VAST，只否定本地 fixed ACF-order
VAST-lite 近似在此坐标的有效性。完整证据见
[`MambaTS v10 ETTh1 H336 readout`](mambatsv10_vastlite_etth1_h336_preflight_readout_20260927.md)。

## 2026-09-27 MambaTS v11 learned VPT plus greedy VAST order：ETTh1 H336 预检停止

在 v10 的固定 ACF 排序失败后，v11 恢复官方 MambaTS VPT 的两个关键训练机制：每个样本各自
随机变量路径，以及由 detached 纯预测 MSE 更新的有向 transition-cost 矩阵。验证时不使用官方
外部 ATSP solver，而以确定性 greedy minimum-cost path 解码一个顺序；输入在同一 shared Mamba
scan 前按该顺序重排，输出随后还原。AutoCon representation 继续通过零初始化 forecast residual
直接进入预测。因此 v11 是机制更接近的本地适配，而非官方 VPT/VAST 复现。

ETTh1/H336、fresh seed `2221--2223` 的冻结三臂 validation-only 预检已完整完成。v11 null/
active/PatchTST 的平均最佳验证 MSE 为 `1.340283/1.342472/1.279776`；active 相对 exact null
为 `+0.163%`，仅 `1/3` seed 改善；相对 PatchTST 为 `+4.899%`，`0/3` seed 胜出。六个 v11
训练均记录 VPT coverage `1.0`、累计边数 `201061`，所以失败不是未执行 transition accounting
造成。9/9 作业成功、三 phase 源快照一致、所有作业 `skip_test=true`，严格审计没有发现测试指标。

由于同时违反 AutoCon 归因和精度门槛，停止 v11 的 VPT/decoder/lambda/预算调参、资源测量、
多数据集扩展和测试释放。该负结果仅否定本地 greedy-decoded VPT 适配在本坐标的有效性，不评价
使用官方 ATSP solver 的原始方法。完整记录见
[`MambaTS v11 ETTh1 H336 readout`](mambatsv11_vpt_etth1_h336_preflight_readout_20260927.md)。

## 2026-09-26 PEMS H=12 fresh same-snapshot：精度候选成立，效率门槛未成立

### 2026-09-26 d24 无间隙四-token MambaTS v3：四 PEMS H12 同时通过精度与速度门槛

保留完整历史覆盖的 `patch_len=24,stride=24`，将宽度固定为 `d_model=24,d_ff=48`。PEMS07 独立三 seed 筛选的 active/null 均值为 `.062087/.062084`，二者均比 TimeMixer `.065095` 低约 `4.62%`；active/null 差为 `+0.0048%`，所以结构收益不能归因于对比损失。四 PEMS 的 fresh seed `2131/2132/2133` 确认中，active MSE 为 `.063093/.080203/.062459/.093008`，相对每个冻结外部最强基线为 `-1.178%/-0.018%/-4.049%/+0.125%`，均处于预先登记的 `1%` 非劣范围。两 phase 均为 `valid=true`、12/12 作业、无测试指标。

串行 GPU prediction-only 基准（B32/L96/H12）中，candidate/TimeMixer 的中位延迟分别是 PEMS03 `.303/1.431 ms`、PEMS04 `.289/1.277 ms`、PEMS07 `.530/.599 ms`、PEMS08 `.268/.869 ms`。候选在四个 PEMS 形状均更快且参数更少；这些数据不包含训练、加载或对比损失时间。AutoCon active 相对 null 的变化为 `-0.015%/+0.007%/-0.061%/-0.075%`，表明它可作为共享表示学习和严格 null 消融保留，但其独立效应很小。该路线可进入 H96/长预测确认，完整审计见 [`d24 PEMS H12 readout`](pems_fullcover_p24_s24_d24_mambatsv3_autocon_h12_confirmation_readout_20260926.md)。

H96 fresh-seed 确认已完成并触发停止：active MSE 为 PEMS03 `.168134`、PEMS04 `.214917`、PEMS07 `.218978`、PEMS08 `.367357`。相对历史强 H96 参考分别为 `+3.610%/+9.757%/+1.452%/-2.160%`，其中 PEMS03 和 PEMS04 超过预设 `2%` 门槛；active/null 变化为 `+0.009%/-0.042%/-0.144%/+0.003%`，方向也不一致。两个 phase 均为 `valid=true`、12/12、无测试指标。即使 PEMS07 prediction-only 仍为 `.567 ms`，效率不能抵消跨数据集精度失败，故停止 d24 四-token路线，不进行同源强基线扩展或测试释放。详见 [`d24 PEMS H96 readout`](pems_fullcover_p24_s24_d24_mambatsv3_autocon_h96_confirmation_readout_20260926.md)。

另一条六-token无间隙路径（`patch_len=16,stride=16,d_model=16`）在同形状 PEMS07/H96 forward 为 `.683 ms`，快于 TimeMixer `2.630 ms`，但新的 three-seed active/null MSE 为 `.231868/.231743`。active 比历史 MambaTSv3 H96 参考 `.215844` 高 `7.424%`，也相对 null 退化 `0.054%`、仅 1/3 seed 改善。该路线不能通过精度门槛，停止于 PEMS07，不做四 PEMS 或测试释放。详见 [`six-token H96 readout`](pems07_mambatsv3_fullcover_p16_s16_d16_autocon_h96_probe_readout_20260926.md)。

最新 120-job validation-only screen 已完成：PEMS03/04/07/08、H=12、seed 2026/2027/2028，DLinear/NLinear/PatchTST/TimesNet/TimeMixer/iTransformer/Mamba2 与 MambaTSv3 plain、AutoCon null、AutoCon active 使用同一冻结源码与训练合同，四个 phase 均为完整且零失败。`MambaTSv3AutoCon` 的 mean validation MSE 分别为 `.060018/.074632/.059442/.088577`，相对每个数据集最佳外部基线分别改善约 `5.99%/6.96%/8.68%/4.64%`。AutoCon active 相对严格 null 的配对均值变化为 PEMS03 `-0.0243%`、PEMS04 `-0.0413%`、PEMS07 `-0.1390%`、PEMS08 `+0.0113%`，说明表示学习目标在前三个数据集有小幅信号，在 PEMS08 近似中性；本轮没有开启独立 `mamba_acf_relation_loss`，不能把结果归因于 ACF relation objective。

代表性 PEMS07 前向资源测量显示，候选 `2.189 ms`、峰值 allocated `647.8 MB`，慢于 TimeMixer `.601 ms`、Mamba2 `.753 ms`、iTransformer `.915 ms`，因此当前不能声称速度或显存优于基线。完整读出见 [`PEMS H=12 fresh readout`](pems_v3_autocon_h12_fresh_same_snapshot_readout_20260926.md)。该结果仍是验证集探索性证据，测试集、H=96 以上 horizon 和置信区间尚未完成，不能直接作为论文主表。

### 2026-09-26 效率结构筛选：稀疏 patch 与变量 token 均未同时通过精度门槛

### 2026-09-26 无间隙四-token MambaTS v3：速度与单坐标精度通过，AutoCon effect 不稳定

此前的 `patch_len=16, patch_stride=24` 虽然只有 4 个 patch token，却只覆盖 96 个历史点中的 64 个，存在 8 点间隙并丢弃末端 8 点。为排除这一结构性缺陷，冻结了 `patch_len=24, patch_stride=24, d_model=16, d_ff=32` 的 full-coverage 版本；它仍只有 4 个 token，但覆盖全部输入历史。PEMS07 `B32/L96/C883/H12` 的 prediction-only GPU 预检为 `27,587` 参数、`0.490 ms` mean（`0.489 ms` median）、`105.2 MB` peak allocated，比同协议 TimeMixer `0.601 ms` 快约 `18.5%`。

新 seed `2091/2092/2093` 的 validation-only active/null 配对均完成，两个 phase 均 `valid=true`、每 phase 3/3 作业、`skip_test=true`，没有测试指标。最佳纯预测 validation MSE 为：null `0.063536/0.063983/0.063578`，均值 `0.063699`；active (`lambda=.001`) `0.063499/0.064017/0.063626`，均值 `0.063714`。active 相对 null 的配对变化为 `+0.024%`，仅 `1/3` seed 改善，因此不能声称 AutoCon 带来稳定预测增益；但 active 和 null 都比历史同快照 TimeMixer `0.065095` 低约 `2.1%`。这条结果支持“无间隙低 token MambaTS v3 在 PEMS07 H12 同时满足局部精度和预测速度门槛”，不支持跨数据集或长 horizon 结论，也不能把架构收益归因于对比学习。

冻结计划、启动授权和 phase 工件见 [`full-coverage p24/s24 probe`](pems07_mambatsv3_fullcover_p24_s24_autocon_h12_probe_20260926.json)、[`launch sidecar`](pems07_mambatsv3_fullcover_p24_s24_autocon_h12_probe_20260926_launch_sidecar.json) 及 [`phase artifacts`](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p24_s24_autocon_h12_probe_20260926/)。下一步应使用新 seed 在 PEMS03/04/07/08 做确认，并把 active/null 作为独立消融；若 active 仍无稳定 effect，论文应将 AutoCon 定位为共享表示训练机制而非精度提升模块。

该四 PEMS 确认已完成，且按预注册规则停止：new seed `2101/2102/2103` 的 active MSE 为 PEMS03 `.064881`、PEMS04 `.082387`、PEMS07 `.064088`、PEMS08 `.093733`。相对同快照最强外部基线，分别为 `+1.623%/+2.705%/-1.546%/+0.905%`；PEMS03 和 PEMS04 超过每数据集 `1%` 非劣界限。active 相对严格 null 的变化为 `-0.109%/-0.012%/+0.116%/+0.002%`，方向不一致，不能声称 AutoCon 产生稳定精度增益。两个 phase 都是 `valid=true`、12/12 jobs、`skip_test=true`，没有测试指标。full-coverage 路线因此不能进入测试释放或论文主模型；完整表和审计入口见 [`full-coverage PEMS readout`](pems_fullcover_p24_s24_mambatsv3_autocon_h12_confirmation_readout_20260926.md)。

为了消除 MambaTSv3 的 `C x patch_count` 共享 scan 成本，完成了 token-sparse v3 的资源和四 PEMS 新 seed 验证。`patch_stride=24,d_model=16` 在 PEMS07/B32/L96/C883/H12 的 prediction-only 测量为 `.475 ms`、`105.2 MB`，快于 TimeMixer/Mamba2/iTransformer；但三 seed 四 PEMS active MSE 为 `.066334/.086068/.065736/.095609`，相对冻结最佳外部基线分别落后 `3.90%/7.29%/.98%/2.93%`，违反 PEMS03/04/08 的 1% 非劣门槛。active/null 的全坐标平均差仅 `-0.0148%`。更稀疏的 stride=32 也在 PEMS07 退化约 `3.6%`。因此停止稀疏 patch 路线，不做测试释放。

同时复测了预测连接的 `NLinearVariableMambaAutoCon`：变量 token 单次 Mamba-1 scan 的 PEMS07 前向为 `.339 ms`、`65.1 MB`、`6,379` 参数，但其当前 train-peak ACF relation objective 在 PEMS07 训练中出现 `Contrastive Anchors=0%`、`Degenerate Positive=100%`，且 null MSE 约 `.075`，比 TimeMixer 差超过 15%；按停止规则未把 active 阶段当作可用证据。完整审计见 [`Mamba efficiency architecture screen`](mamba_efficiency_architecture_screen_20260926.md)。这两个负结果说明当前不能通过单纯缩短 Mamba token 序列来同时满足精度和速度目标。

2026-09-26 的首次 `Mamba2AutoCon` 表征预测连接 PEMS07 H12 screen 训练已完成但被作废为不可解释的探索结果：新增分支最初构建在历史预测层之前，导致 fusion/control 的共享预测层初始化不一致。虽该批 fusion 验证 MSE 表面上优于 null，不能将差异归因于 fusion，也不用于选择、报告或测试释放。代码已改为在所有历史预测层之后创建可选分支，并新增共享初始化逐元素一致的回归测试。

### 2026-09-26 Mamba2 预测连接复跑与变量-token d64 结论

修正后，`Mamba2AutoCon` 在 PEMS07/H12 的三新种子 exact paired validation-only rerun 中，fusion/null 的平均 validation MSE 为 `.060710/.071304`，fusion 对 null 为 `-14.86%`，也低于同快照 TimeMixer `.065095`（`-6.74%`）。共享参数的逐元素初始化一致性和 fusion gate 的梯度连接均有回归测试；两 phase validator 均为 `valid=true`，且 `skip_test=true`。这说明“把 AutoCon representation 接入预测路径”在这个单数据集短预测期坐标有强信号，但还不能推广为跨数据集或论文结论。

同一 RTX 5090 D、PEMS07 `B32/L96/C883/H12` prediction-only 协议下，Mamba2 fusion 为 `.824 ms`（`217,386` 参数、`74.6 MB` peak allocated），而 TimeMixer 为 `.601 ms`。缩小 Mamba2 宽度没有降低该路径延迟。因此 Mamba2 fusion 通过此坐标的精度门槛、未通过“快于 TimeMixer”的效率门槛，停止作为效率论文主候选；完整计划为 [`Mamba2 paired rerun`](pems07_mamba2_repr_fusion_autocon_h12_paired_rerun_20260926.json)。

为排除变量-token 小模型的容量不足，新增了 d64/dff128 单向 Mamba-1 配对 screen，并把 `mamba_acf_relation_lag_mask=true` 固定到训练合同。两组各三个作业均 `valid=true`；relation telemetry 的 contrastive-active anchor 约为 `68.8%--69.7%`、degenerate-positive 约 `17.9%--18.4%`，不再是旧 screen 的零负样本退化。active/null 平均 validation MSE 为 `.067201/.067309`，active 的小幅变化为 `-0.16%`，但仍分别高于同快照 TimeMixer `.065095` 约 `3.24%/3.40%`，触发停止规则。其前向为 `.336 ms`、`56,155` 参数、`96.2 MB` peak allocated，确实快于 TimeMixer，却未达精度非劣；不得测试释放或作为论文主结果。计划和相位校验工件见 [`d64 relation paired screen`](pems07_nlinear_variable_mamba_d64_relation_paired_screen_20260926.json)。

## 2026-09-25 NLinearVariableMamba：结构有高维精度/速度信号，但 AutoCon 仍未产生可归因增益

### 冻结后的 Solar H336/H720 新 seed screen

最新的 16-job validation-only screen 已完成：候选 `NLinearVariableMambaAutoCon` 与 DLinear、iTransformer、PatchTST 在 H336/H720、seed 2022/2023 上使用同一训练合同。候选 validation MSE 在 4/4 个单元优于 iTransformer，均值为 H336 `0.154109` 对 `0.179891`（`-14.33%`），H720 `0.154078` 对 `0.180783`（`-14.77%`）。这证明候选结构的长 horizon 信号没有在新 seed 上消失，但仍是 validation-only，且没有 active/null 配对，不能归因于 AutoCon relation loss。

同一 RTX 5090 D prediction-only 资源协议（B32、30 warm-up/100 timed）下，候选 H336/H720 为 `0.29779/0.34960 ms` median；iTransformer 为 `0.31426/0.34021 ms`，PatchTST 为 `1.74606/1.85718 ms`，DLinear 为 `0.09218/0.11691 ms`。因此候选仅在 H336 略快于 iTransformer，H720 略慢，两个 horizon 都慢于 DLinear。完整读出见 [`Solar multihorizon confirmation readout`](nlinear_variable_mamba_repr_fusion_solar_multihorizon_confirmation_screen_readout_20260925.md)；不能把速度结论泛化为“超过所有基线”。

### Traffic H336/H720 复验与低秩 mixer 停止

冻结 Solar 候选在高维 Traffic（862 变量）H336/H720、seed 2022/2023 的 4/4 坐标都落后 iTransformer：候选均值 `.376081/.402033`，iTransformer `.368431/.397886`，分别为 `+2.08%/+1.04%`。这否决了该候选已有跨数据集精度结论，测试仍关闭。尝试的 ReZero 低秩跨变量 `C -> 8 -> C` 残差更差：相对原候选 H336/H720 为 `+3.31%/+1.85%`，4/4 坐标退化，停止 rank/初始化/深度搜索。低秩路径虽在 Traffic prediction-only 测量中快于 iTransformer（`.38200/.45706 ms` 对 `.96408/.97909 ms`），却不满足精度非劣。完整记录见 [`Traffic multihorizon readout`](nlinear_variable_mamba_traffic_multihorizon_readout_20260925.md)。

随后仅改变候选的窗口归一化做了 8-job MeanStd 消融：H336 均值由 LastVal `.376081` 降至 `.374894`（`-0.32%`），H720 升至 `.405335`（`+0.82%`），仍分别落后同源 iTransformer `1.75%/1.87%`。归一化不是稳定的 Traffic 解决方案，停止继续搜索；读出见 [`Traffic MeanStd normalization`](nlinear_variable_mamba_meanstd_traffic_multihorizon_readout_20260925.md)。

最后固定双向变量扫描并加入训练期 signed-ACF profile 的确定性通道相似性排序，进行同样的
Traffic H336/H720、seed 2022/2023、8-job validation-only 复验。排序使 H336 从 `.373792`
小幅降至 `.373396`（`-0.11%`），H720 从 `.399869` 变为 `.399875`（无改善）；相对同源码
iTransformer 的差距仍为 `+1.35%/+0.50%`，且候选 4/4 坐标均落后。因此停止 ACF 通道排序及
双向组合搜索，不进入额外数据集或 test release。ACF 只从训练 split 固定一次排序，验证期不动态
计算，但该对比同时保留 relation loss，不能归因于 AutoCon。完整审计见
[`Traffic bidirectional similarity readout`](nlinear_variable_mamba_bidir_similarity_traffic_multihorizon_readout_20260926.md)。

新增 `NLinearVariableMamba/NLinearVariableMambaAutoCon`：以 LastVal-NLinear 全变量线性层作为预测基座，将每个变量的完整历史窗口投影为一个 token，仅执行一次 Mamba-1 变量轴扫描，再用 ReZero 残差修正预测。AutoCon 表示由归一化历史与同一批变量 token 的动态双线性投影构成，因此对比损失反向更新变量 token、Mamba 状态和表示投影，而非一个与预测脱钩的辅助编码器。该实现是面向当前硬件约束的紧凑本地候选，不宣称等同于公开的 S-Mamba、RaMamba 或 MambaTS。

在 RTX 5090 D、FP32、50 warm-up/200 timed、device-resident prediction-only 协议下，训练同构的 active 网络 (`d_model=32,d_ff=64,d_state=8,d_conv=2,expand=1`) 在 Solar/C137/H96/B32 为 `31,553` 参数、`.19450 ms` 中位延迟、`.20323 ms` p95；Traffic/C862/H96/B8 为 `54,753` 参数、`.18898 ms`、`.19920 ms`。工件见 [`Solar active resource`](../../artifacts/exploratory/nlinear_variable_mamba_preflight_20260925/resource_solar_active/model_resources.json) 与 [`Traffic active resource`](../../artifacts/exploratory/nlinear_variable_mamba_preflight_20260925/resource_traffic_active/model_resources.json)。资源工具尚未暴露 `residual_init`；它只改变一个标量的初值，不改变参数量或预测前向算子。前向期间关闭表示返回，但模型仍含完整 AutoCon 参数；因此这里是部署预测路径，不是训练步骤或对比损失耗时。Traffic 速度快于当前 iTransformer `.34059 ms`，Solar 与 iTransformer `.19146 ms` 的差距约 `1.6%`，在预先设定的 `<=10%` 容差内。

随后完成了同一结构的 `d_model=64,d_ff=128` 高维复核。Traffic/H96/seed2021 的 active 最佳 validation MSE 为 `.380135`，严格 null 为 `.380277`；Solar/H96/seed2021 分别为 `.141115` 与 `.141131`。相对同坐标 iTransformer（Traffic `.413808`、Solar `.163454`），两者均有明显验证集优势，但 active 相对 null 的差异仅为 Traffic `0.037%`、Solar `0.011%`，尚不能归因于 AutoCon。修复 relation-model 白名单后，日志中的 `lag_pairs` 已实际生效；Traffic active 的 ACF telemetry 约为 active anchors `91.7%`、contrastive anchors `32.7%`、sign accuracy `91.7%`，说明对比目标确实有有效样本而非静默退化为 shuffle。上述仍是单 seed、H96、8 epoch、validation-only 结果，未开放测试集。

随后对该 d64 配置做了完整三 seed paired screen。Traffic active/null 的最佳 validation MSE 分别为 seed2021 `.380135/.380277`、seed2022 `.385083/.385154`、seed2023 `.384888/.384834`，active 相对 null 的平均变化为改善 `0.0139%`、跨 seed 标准差 `0.0259%`，且 seed2023 退化。Solar active/null 分别为 seed2021 `.141115/.141131`、seed2022 `.139635/.139674`、seed2023 `.141515/.141492`，平均改善 `0.0074%`、标准差 `0.0225%`，同样有 seed2023 退化。因此 AutoCon 的预测 effect size 在两个高维数据集都接近零，不能写成稳定的精度贡献；只能写成共享 Mamba 表示上的有效训练目标和可审计消融。

固定 Traffic/H96/seed2021/d64 进行的权重校准也不支持简单提高 `AutoCon_lambda`：`lambda=.01` 的最佳 validation MSE 为 `.3843`，明显差于同配置 `lambda=.001` 的 `.380135` 和 relation-null 的 `.380277`。这避免了把 `.001` 的近零差异误判为“权重不足”。

低维反例筛查（ETTh1/H96/seed2021）没有显示该结构必然失效：active/null 最佳 validation MSE 为 `.677074/.678174`，略优于同快照 PatchTST `.685596`，但仍只有单 seed、validation-only 证据，且没有证明 AutoCon 的独立贡献。后续论文主张应限定为“在高维多变量坐标上重点验证的轻量变量轴 Mamba”，不能声称已在所有数据集上统一优于基线。

## 2026-09-25 Solar 单 seed测试级 replay：结构有效，AutoCon effect 为负

为验证测试集流程，使用新生成并绑定 `acf_prior.npz` sidecar 的 Solar/H96/seed2021 checkpoint 做独立 `is_training=0` replay。评估日志明确从训练 sidecar 加载完整 ACF 表，未重新拟合 ACF。结果为：NLinearVariableMambaAutoCon active `MSE=.202450`，同结构 relation-null `MSE=.202396`，同 seed iTransformer `MSE=.222394`。因此候选相对 iTransformer 的测试 MSE 改善约 `8.96%`，但 active 相对 null 退化约 `0.027%`；该结果支持“轻量变量轴 Mamba 主干在此 Solar 坐标有效”，不支持“AutoCon 提高测试精度”。目前仍只有一个测试 seed，不能作为完整论文主表。

本次 replay 暴露并修复了一个工程性但重要的可复现性缺口：NLinearVariableMamba 现在实现 `set_compact_acf_prior/export_acf_prior_state/save_acf_prior/load_acf_prior`，并在通用 loader 中支持 `enc_in/acf_prior_lags` 契约。后续所有测试释放必须带 sidecar、哈希和独立 replay 日志；旧的 validation-only 工件没有 sidecar，不能直接升级为测试结论。

同一 sidecar-bound 流程随后在 Traffic/H96/seed2021 完成：active `MSE=.528765`、relation-null `.528942`、同 seed iTransformer `.537446`。候选相对 iTransformer 改善约 `1.62%`，active 相对 null 改善约 `0.033%`。Traffic 与 Solar 的单 seed 测试方向一致，但两个 active/null effect size 都非常小，仍不足以支持 AutoCon 的独立预测贡献或统计显著性声明；至少还需要预注册的多 seed 测试释放和固定的比较集合。

在相同 RTX 5090 D、FP32、50 warm-up/200 timed、device-resident prediction-only 协议下，d64 版本参数为 `61,506`；Solar/C137/H96/B32 的中位前向延迟为 `.20931 ms`、p95 `.21823 ms`、峰值 allocated `28.92 MB`，Traffic/C862/H96/B8 为 `.20662 ms`、p95 `.21543 ms`、峰值 allocated `39.96 MB`。工件见 [`Solar d64 resource`](../../artifacts/exploratory/nlinear_variable_mamba_preflight_20260925/resource_solar_d64_active/model_resources.json) 与 [`Traffic d64 resource`](../../artifacts/exploratory/nlinear_variable_mamba_preflight_20260925/resource_traffic_d64_active/model_resources.json)。该基准只测预测前向，不包含反向、ACF 计算或对比损失耗时。

## 2026-09-25 Prediction-connected AutoCon fusion：结构值得保留，对比损失尚未通过 null 检验

`NLinearVariableMambaAutoCon` 现将其 AutoCon 表示接入预测而非只作为辅助头：归一化历史与变量轴 Mamba token 形成 `[B,T,D]` 表示，时间均值经门控 FiLM 调制同一 token，再由 `variable_head` 产生预测残差。融合标量采用零初始化 ReZero；初始化时预测与未融合模型严格相同，但 MSE 对 `repr_fusion_scale`、投影层和 Mamba 均有非零梯度。训练、计划校验器、sidecar replay 和资源基准都已登记这个模型；旧 MambaACF 专属的 ACF-memory、通道门控约束没有被放开。

融合结构的探索性 H96/seed-2021 validation 读数为 Solar active/null `.138978/.139380`，Traffic `.374573/.374639`。与融合前同配置的 active 读数 Solar `.141115`、Traffic `.380135` 相比，结构分别下降约 `1.52%/1.46%`，但这只是架构比较，不能归因于 contrastive loss。Solar 的当前强基线 iTransformer 为 `.163454`；Traffic 同坐标 iTransformer 为 `.413808`，所以该轻量变量轴结构在这两个高维 H96 探索坐标具有精度信号。

为检验 AutoCon relation loss，本轮固定融合、数据、Mamba 配置、训练轮数、seed 和 train-only ACF 合同，只改变 `AutoCon_lambda`。在 Solar H96 的三个严格配对 seed 中，`lambda=1e-4` active/null 最佳 validation MSE 分别为 seed-2021 `.138859/.138517`、seed-2022 `.137683/.137439`、seed-2023 `.138264/.138327`；active 平均 `.138269`，null `.138095`，平均相对退化 `0.126%`，仅 `1/3` 胜出。此前 `lambda=.001` 的三 seed screen 也为 `1/3` 胜出，平均更差。独立的 seed-2024/2025 冻结检查为 `.138026/.138043` 与 `.137262/.137054`，一胜一负，平均仍偏向 null；样本量过小，不能作统计结论。故当前不得声称 ACF contrastive relation loss 提升预测精度；它只能作为已实现、可审计、预测路径共享的训练目标和负消融结果保留。所有上述作业都明确 `skip_test=true`，不使用测试集选模型。

为排除“global non-overlap pair 太稀疏”的解释，另做唯一的局部关系对照：`scope=both,min_lag=2,lambda=1e-4` 让 `[B,96,D]` 时间表示获得窗口内短滞后 ACF targets，日志中 local target coverage 为 `1.61%`，并且训练时间由约 `3.8` 秒/epoch 增至约 `4.3` 秒/epoch。Solar H96 三个严格 paired seed 的 active/null 读数为 `.138406/.138445`、`.137485/.137342`、`.137611/.137440`，平均 active `.137834`、null `.137742`，相对退化 `0.067%`，仍仅 `1/3` 胜出。因此停止 global/local scope 与 relation-weight 的进一步搜索；当前实验证据不支持将这个 relation loss 作为论文的精度贡献。

在最新同设备 FP32、30 warm-up/100 timed、B32、device-resident prediction-only 资源测量中，融合模型在 Solar/C137/H96 为 `69,763` 参数、`.28685 ms` median、`.30268 ms` p95；同工具 DLinear 为 `.06789 ms`，iTransformer `.31768 ms`，PatchTST `1.70774 ms`。因此它快于 iTransformer 约 `9.7%`、快于 PatchTST，但明显慢于 DLinear；不能将“计算更快”泛化为对所有基线更快，也不能将这条单坐标资源读数当作端到端训练速度。工件见 [`fusion Solar resources`](../../artifacts/exploratory/nlinear_variable_mamba_repr_fusion_20260925/resource_benchmark_solar_h96/NLinearVariableMambaAutoCon/model_resources.json)。

相同协议在 Traffic/C862/H96/B8 下，融合模型为 `.27712 ms` median、`.28668 ms` p95；DLinear `.07869 ms`、iTransformer `.94571 ms`、PatchTST `2.60000 ms`。这说明高维 Traffic 上该模型相对当前 iTransformer/PatchTST 有显著 prediction-only 速度优势，但仍不能超过线性基线。工件见 [`fusion Traffic resources`](../../artifacts/exploratory/nlinear_variable_mamba_repr_fusion_20260925/resource_benchmark_traffic_h96/NLinearVariableMambaAutoCon/model_resources.json)。

Solar 长预测的单 seed exploratory screen 使用同一 TXT/10 分钟 loader、`seq_len=96`、seed-2021、8 epoch、`skip_test=true`。H336 中融合 active 候选为 `.153489`，DLinear `.229089`、iTransformer `.178777`、PatchTST `.193603`；H720 中候选 `.154733`，DLinear `.225950`、iTransformer `.180631`、PatchTST `.190809`。因此候选在 H336/H720 都通过了单 seed 的强基线筛查，且未出现 horizon 崩溃；但 H96 是在其上选择结构/权重的坐标，所以 H336/H720 只能作为探索性信号。任何论文主张仍需要冻结超参数后，在 seed-2022/2023 的相同基线集合上重复，并在验证选择完成后才允许独立 test replay。

仅有 H96、seed 2021、8 epoch 的 validation-only 初筛，不能作为测试或论文主结果。Solar active (`lambda=.001`) 的最佳 validation MSE 为 `.142908`，相同配置 null (`lambda=0`) 为 `.142903`；它们都优于同坐标 iTransformer 基线 `.163454`，但 active 相对 null 恶化约 `.0035%`。Traffic active 为 `.406417`，同种子、相同配置 null 为 `.406308`，active 恶化约 `.027%`；两者优于 iTransformer `.413808`，但未超过已有 MambaACF `.394335`。所有这些作业均使用 `--use_gpu true --gpu 0 --skip_test --no_save`，没有测试集指标。一次漏传 `enc_in/dec_in/c_out=862` 的 null 命令在形状错误处退出，保留在同一实验目录但不纳入读数；上面的 retry 才是有效配对。结论是：变量轴 Mamba 残差值得作为高维速度-精度结构继续复核，但目前没有任何证据证明 AutoCon 提高了该结构的预测精度，不能围绕此 `lambda` 声称贡献，也不能进入论文测试释放。

## 2026-09-25 LinearLatentMamba：线性基座恢复精度，但 AutoCon 增益尚未成立

新增 `LinearLatentMamba/LinearLatentMambaAutoCon`：NLinear 风格的全通道线性预测作为主基座，低秩变量压缩后的 24 个时间 token 只经过一次小型 Mamba-1 扫描；预测残差和 AutoCon 表示均来自同一 latent state。AutoCon 直接读取预测主干的 latent Mamba states，因而对比损失会更新共享编码器，不额外增加第二个预测 backbone。

资源预检（低秩 `rank=4,d_state=4`）在 RTX 5090 D、FP32、50/200 device-resident forwards 下为 Solar/C137/B32 `0.22099 ms`、Traffic/C862/B8 `0.21448 ms`，参数分别 `10,933/16,733`，峰值 allocated 分别约 `20.37/27.68 MB`。这使它在 Traffic 形状上快于当前 iTransformer/PatchTST 资源参考，但 Solar 仍慢于 iTransformer `.19146 ms`，不能宣称跨数据集统一更快。工件见 [`Solar`](../../artifacts/exploratory/linear_latent_mamba_preflight_20260925/resource_solar/model_resources.json) 与 [`Traffic`](../../artifacts/exploratory/linear_latent_mamba_preflight_20260925/resource_traffic/model_resources.json)。训练配置使用 `rank=16,d_state=8`；按训练配置复测后 Solar/C137/B32 为 `.21670 ms`、`16,369` 参数，Traffic/C862/B8 为 `.20859 ms`、`39,569` 参数，工件见 [`Solar rank16`](../../artifacts/exploratory/linear_latent_mamba_preflight_20260925/resource_solar_rank16/model_resources.json) 与 [`Traffic rank16`](../../artifacts/exploratory/linear_latent_mamba_preflight_20260925/resource_traffic_rank16/model_resources.json)。

在 Solar H96、seed 2021、8 epoch validation-only 探索中，`d_model=32,d_ff=64,latent rank=16,d_state=8,d_conv=2,residual_init=0.1` 的最终共享 latent 版本 AutoCon null (`lambda=0`) 最佳 validation MSE 为 `0.182381`，同源码 active `lambda=.001` 为 `0.182278`，改善仅约 `0.056%`。此前尝试的 direct representation-to-forecast route 在 `lambda=.03` 下恶化到 `0.502574`，已移除；bounded route 的 paired screen 也出现约 `2.88%` 退化。这说明当前证据只支持“AutoCon 对共享 Mamba 表示施加了极小方向性影响”，不支持稳定精度增益。所有训练均为 GPU validation-only，`skip_test=true`，没有测试指标或论文主结果释放。当前候选可继续做多 seed null/active 复核，但不能写成已达到论文级精度结论。

## 2026-09-25 InvertedRaMamba：变量 token 单扫描仍未通过 Solar 速度门槛

新增 `InvertedRaMamba/InvertedRaMambaAutoCon` 作为公开 RaMamba/DMamba 变量轴思路的效率约束本地候选：将每个变量的时间窗口投影成一个 token，执行一次前向 Mamba-1 扫描，并用同一 token 状态构造 AutoCon 的时间表示；趋势支路为线性预测。该实现明确省略公开结构中的双向扫描和较大 FFN，不声称与官方模型等价。

在 RTX 5090 D、Solar/C137/H96/B32、prediction-only、50 warm-up/200 timed 的统一资源协议中，`d_model=32,d_state=4,d_conv=2,expand=1` 时参数为 `19,456`，中位延迟为 `.23211 ms`，p95 为 `.24243 ms`。同坐标 iTransformer 为 `.19146 ms`，因此慢约 `21.2%`，超过预注册的 `<=10%` 容差；未启动训练、lambda 搜索或测试释放。原始资源工件见 [`InvertedRaMamba`](../../artifacts/exploratory/inverted_ramamba_solar_preflight_20260925/resource_inverted/model_resources.json)。CUDA shape/gradient smoke 与注册契约测试通过，但该候选目前只能作为可复现实验负结果保留。

## 2026-09-25 LatentTemporalMamba：低 token 单扫描仍慢于 Solar iTransformer，训练前停止

为避免 MambaTS 的 `C x patch_count` 长序列扫描，新增 `LatentTemporalMambaAutoCon`：每个时刻的变量向量先压缩到最多 16 维，四步平均为 24 个时间 token，再执行一次 Mamba-1 scan；DLinear-style seasonal/trend 是全变量主预测，来自同一 Mamba 状态的 ReZero decoded residual 直接修正预测。AutoCon 的 `[B,96,D]` 表示也由这一状态上采样而来，因此 contrastive loss 与 forecast residual 共享 `variable_compress` 与 Mamba 梯度，而非另建辅助-only encoder。C=7/137/862 的 CUDA forward/backward smoke 均通过。

第一版逐 horizon/variable decoder 为 `29,713` 参数、`.29656 ms` 中位延迟；随后将预测修正压缩为“共享 horizon 因子 x 共享变量因子”的 rank-one residual，缩至 `22,409` 参数仍为 `.30624 ms`。相同工具的 iTransformer 为 `14,944` 参数、`.19146 ms`，两种候选均慢至少 `54.9%`。独立 Mamba-1 原语探针（B32、宽度 8、长度 1/6/24）中位值仍为 `.09643--.10042 ms`，表明 Mamba kernel 固定成本与标准化/分解路径共同构成下限，而非仅仅是长 token 序列。尽管第一版比 Solar 上的 v7 `.62221 ms` 快约一半，二者均超过不慢于 iTransformer `10%` 的预注册筛选容忍范围。资源工件见 [`dense decoder`](../../artifacts/exploratory/latent_temporal_mamba_solar_preflight_20260925/resource_latent/model_resources.json)、[`rank-one decoder`](../../artifacts/exploratory/latent_temporal_mamba_solar_preflight_20260925/resource_latent_rank1/model_resources.json) 与 [`iTransformer`](../../artifacts/exploratory/mambatsv7_solar_preflight_20260925/resource_itransformer/model_resources.json)。故不启动训练、lambda 搜索或测试；这个结果仅说明当前 Mamba-1 kernel 与该任务形状下的最快注意力基线仍存在固定开销。

## 2026-09-25 MambaTSv7 Solar 资源预检：训练前停止

Solar/C137/H96/B32 的 v7 AutoCon 配置（train-only ACF memory/selector、MeanStd）在同一 50/200 device-resident 协议为 `92,740` 参数、`.62221 ms`，相对 iTransformer `.19146 ms` 慢约 `225%`。资源工件为 [`v7`](../../artifacts/exploratory/mambatsv7_solar_preflight_20260925/resource_mambatsv7/model_resources.json) 和 [`iTransformer`](../../artifacts/exploratory/mambatsv7_solar_preflight_20260925/resource_itransformer/model_resources.json)。不对一个已经明显不满足效率条件的候选启动 Solar 训练，也不把未发生的训练写作负结果。

## 2026-09-25 SolarEnergy 当前强基线扩展：iTransformer 成为新增高维坐标门槛

SolarEnergy AL 已作为新的 10 分钟、`52,560 x 137` 多变量坐标接入。原始无表头文本位于 `dataset/solar/solar_AL.txt`，SHA-256 为 `230327ef...1a0122ac2`；上游压缩文件 MD5 为 `41ef7fdc958c2ca3fac9cd06d6227073`。其 array loader 合同固定为 `data_format=txt`、`data_interval_minutes=10`、target index `136`，由计划指纹和加载器测试共同验证。

已完成 H96、seed `2021/2022/2023`、DLinear/NLinear/PatchTST/TimesNet/iTransformer 的 `15/15` validation-only 作业。严格读取器核验完整唯一坐标、训练完成工件、有限 validation loss、配置中的 TXT/10 分钟合同、零测试指标工件，以及一致的源码快照 `cd904e61...b00d3a74`。冻结计划为 [`solar_current_strong_baseline_extension_20260925.json`](solar_current_strong_baseline_extension_20260925.json)，机器读数为 [`Solar baseline readout`](../../artifacts/exploratory/solar_current_strong_baseline_extension_20260925/readout/readout.md)。平均 validation MSE 分别为 DLinear `.210919`、NLinear `.251149`、PatchTST `.176532`、TimesNet `.206788`、iTransformer `.163454`；iTransformer 获胜。对应平均作业墙钟为 `11.099/11.823/31.382/88.053/18.346` 秒。这是 validation-only 的新基线坐标，不是测试集结论；任何面向 Solar 的 Mamba+AutoCon 候选必须同时与 iTransformer 的精度和速度比较，不能选择性忽略该基线。

## 2026-09-25 ETTh2/ETTm2 当前强基线扩展：NLinear 均为最优

已完成 ETTh2、ETTm2 的 H96、seed `2021/2022/2023`、DLinear/NLinear/PatchTST/TimesNet/iTransformer 共 `30/30` 个 validation-only 作业。每项均为 `status=ok`、`returncode=0`、`skip_test=true`，严格读取器确认 30 个唯一坐标、有限验证损失、零测试指标工件，且作业前后共享源码快照 `cd904e61...b00d3a74`。冻结计划为 [`ett_variants_current_strong_baseline_extension_20260925.json`](ett_variants_current_strong_baseline_extension_20260925.json)，机器读数为 [`ETT variants baseline readout`](../../artifacts/exploratory/ett_variants_current_strong_baseline_extension_20260925/readout/readout.md)。实际训练命令明确使用 `--use_gpu true`。

ETTh2 的 mean validation MSE 为 DLinear `.220205`、NLinear `.207745`、PatchTST `.213271`、TimesNet `.244804`、iTransformer `.222693`；ETTm2 相应为 `.132768`、`.125717`、`.126063`、`.131103`、`.130235`。两个新数据集均由 NLinear 获胜。这扩大了当前基线覆盖，也提高了后续 Mamba 候选所需跨 ETT 非劣门槛：不能只与 PatchTST/iTransformer 比较，必须包含 NLinear。

## 2026-09-25 BiPatchMamba：双向 Bi-Mamba4TS 路线在训练前触发速度停止规则

已根据官方 Bi-Mamba4TS 的可分离结构新增本地 `BiPatchMamba/BiPatchMambaAutoCon`：patch token 在 `mixing` 模式按每个 patch 的变量轴扫描，在 `independent` 模式按每个变量的 patch 轴扫描；每个 block 采用原生 Mamba-1 的前向/反向 scan 和 token FFN。AutoCon 暴露的时间表示来自同一 channel-patch state，并经零初始化 forecast correction 直接进入预测，因而不是与预测脱钩的辅助头。原项目的 Mamba+ 是单独编译的自定义 CUDA 实现，当前主机无 `nvcc`，本地实现不声称与其数值或性能等价。

在 RTX 5090 D、Traffic/C862/H96/B8、device-resident、50 warm-up/200 timed forwards、关闭辅助表示返回的同一资源协议中，`mixing` 为 `94,145` 参数、`1.00419 ms` 中位延迟，`independent` 为 `3.85378 ms`；用当前相同工具重测的 PatchTST 为 `46,080` 参数、`.88744 ms`。因此最快 BiPatch 分支仍慢 `13.16%`，超过本项目的 `<=10%` 延迟容差，独立分支更不具备可行性。资源工件为 [`mixing`](../../artifacts/exploratory/bipatch_mamba_preflight_20260925/resource_traffic_h96/model_resources.json)、[`independent`](../../artifacts/exploratory/bipatch_mamba_preflight_20260925/resource_traffic_h96_independent/model_resources.json) 与 [`current PatchTST`](../../artifacts/exploratory/bipatch_mamba_preflight_20260925/resource_traffic_h96_patchtst_current/model_resources.json)。两种 token 轴的 CUDA forward/backward shape 与 AutoCon 直连梯度 smoke 已通过，但没有启动训练、没有验证/测试指标；按先验效率门槛停止此路线，不将它计作候选精度结果。

## 2026-09-25 NLinear 当前快照基线补充：未改变主结论

为消除当前 MambaACF H96 强基线表遗漏 NLinear 的问题，已完成 ETTh1、ETTm1、Electricity、Traffic 的 H96、seed `2021/2022/2023` 共 `12/12` 个 validation-only NLinear 作业。每项均为 `status=ok`、`returncode=0`、`skip_test=true`，无测试指标工件，并与 MambaACF retry 的 `84` 个作业共享源码快照 `04d2d14f...1242c1050`。严格读取器核验了完整坐标、配置、有限验证损失与源码前后哈希；冻结计划为 [`nlinear_current_h96_baseline_extension_20260925.json`](nlinear_current_h96_baseline_extension_20260925.json)，机器读数为 [`NLinear extension readout`](../../artifacts/exploratory/nlinear_current_h96_baseline_extension_20260925/readout/readout.md)。

NLinear 的 mean validation MSE 为 ETTh1 `.697481`、ETTm1 `.407191`、Electricity `.202028`、Traffic `.506844`，均未取代既有的 PatchTST/iTransformer/TimesNet 最优坐标。因此它补齐了对照，而未改变“当前 MambaACF 只在两个高维数据集有信号、不能支持统一主模型或 AutoCon 稳定增益”的停止结论。实际训练命令均使用 `--use_gpu true`，运行于 `cuda:0` RTX 5090 D；这只说明执行设备，不构成速度优越证据。

## 2026-09-25 MambaACF 当前快照强基线复验：高维有效、低维和统一速度失败，停止主候选

在修正一轮非 Mamba baseline 误传 `mamba_acf_relation_min_lag` 后，使用新的不可变 retry provenance 完成 `84/84` validation-only 作业：ETTh1、ETTm1、Electricity、Traffic 的 H96、seed `2021/2022/2023`，覆盖 DLinear、PatchTST、TimesNet、iTransformer、MambaACF forecast-memory、完全相同 AutoCon-capable MambaACF 的 `lambda=0` relation null、以及 `lambda=.001` active。所有 retry 作业为 `status=ok`，四个 phase 共享同一源码快照，严格读取器检查训练完成工件、config、train-only `acf_prior.npz`、有限 `val_loss`、AutoCon telemetry 和零测试指标工件。旧 provenance 中的 1 个参数校验失败不被重用或计入结果。冻结计划为 [`mamba_acf_current_strong_baseline_screen_20260925.json`](mamba_acf_current_strong_baseline_screen_20260925.json)，机器读数为 [`current MambaACF readout`](../../artifacts/exploratory/mamba_acf_current_strong_baseline_screen_20260925_retry/readout/readout.md)。

active MambaACF 的 H96 validation MSE 相对同快照最强基线在 Electricity/Traffic 为 `-8.945%/-4.706%`，但 ETTh1/ETTm1 为 `+6.435%/+8.090%`。这证明现有 ACF memory+AutoCon Mamba 路径在高维场景有真实信号，却不能支持跨数据集不劣。更关键的是，active 相对完全相同的 relation null 仅为 ETTh1 `-0.044%`、ETTm1 `-0.010%`、Electricity `-0.170%`、Traffic `+0.023%`，虽有三个数据集的方向性改善，但不足以把 AutoCon relation 写成稳定精度贡献。

统一 device-resident prediction-only 资源进一步否决“统一高效”说法：Traffic/C862/H96/B8 中 MambaACFAutoCon `.82957 ms`，略快于 PatchTST `.88792 ms`，但慢于 iTransformer `.34059 ms` 与 DLinear `.10198 ms`；ETTh1/C7/H96/B32 中 MambaACFAutoCon `.81554 ms`，慢于 PatchTST `.28352 ms`、iTransformer `.15562 ms`、DLinear `.09411 ms`，参数为 `40,405`。资源工件见 [`ETTh1 benchmark`](../../artifacts/exploratory/mamba_acf_current_strong_baseline_screen_20260925_retry/resource_benchmark_etth1_h96/model_resources.json) 与 [`Traffic preflight`](../../artifacts/exploratory/mamba_acf_multidataset_candidate_screen_20260910/resource_preflight_traffic_h96_current/model_resources.json)。因此停止该当前 MambaACF 主候选的长 horizon、lambda 搜索和测试释放；可作为高维专用消融保留，不能作为目标中的统一论文主模型。

## 2026-09-25 官方 Mamba-3 环境可行性审计：当前机器阻断

官方 Mamba README 已列出 Mamba-3，但已验证的训练环境为 Python `3.9`、`mamba-ssm=2.3.1`，不导出 `Mamba3`。独立的 Python `3.11`/PyTorch `2.10.0+cu128` 环境满足 Python 前提，却未安装 Mamba；对最新版官方 `mamba-ssm` 的无写入 dry-run 显示构建需要 `nvcc`，而主机没有 CUDA toolkit/compiler，且该环境的 `CUDA_HOME` 为无效的冒号拼接路径。故不修改当前稳定 Mamba-1/2 环境，也不声称已实现或测试 Mamba-3；只有在具备可用 CUDA toolkit 的隔离环境完成真实 GPU 前反向 smoke 后，才能把 Mamba-3 作为独立候选。

## 2026-09-25 RaMambaLite 单扫描变量 Mamba：AutoCon 无效，停止

完成新的 `84/84` validation-only H96 矩阵：ETTh1、ETTm1、Electricity、Traffic，seed `2021/2022/2023`，同一源码快照下的 DLinear、PatchTST、TimesNet、iTransformer、plain RaMambaLite、相同 AutoCon-capable 网络的 `lambda=0` null，以及预注册 `lambda=.001` active。全部作业为 `status=ok`、`returncode=0`，严格读取器确认坐标完整、源码快照一致、每项 `skip_test=true`，且没有测试指标工件。冻结计划为 [`ramamba_lite_autocon_h96_screen_20260925.json`](ramamba_lite_autocon_h96_screen_20260925.json)，严格读数为 [`RaMambaLite H96 readout`](../../artifacts/exploratory/ramamba_lite_autocon_h96_screen_20260925/readout/readout.md)。

plain RaMambaLite 与最强强基线的 validation MSE 差距为 ETTh1 `+0.281%`、ETTm1 `+0.015%`、Electricity `+11.592%`、Traffic `+0.265%`，没有任何单元获胜。active AutoCon 相对相同架构 null 分别为 `+0.039%/+0.039%/+0.318%/+0.138%`，仅 `1/3`、`1/3`、`0/3`、`2/3` seed 改善；因此没有“自监督对比学习提升 RaMambaLite 预测精度”的证据。停止该结构的 AutoCon、长 horizon 和测试释放，不再围绕其 lambda、表示头或训练预算调参。

ETTh1/C7/H96/B32、30 warm-up/100 timed 的 device-resident prediction-only 测量中，plain RaMambaLite 有 `25,632` 参数、`.27165 ms` 中位延迟；PatchTST 为 `.28349 ms`、TimesNet `2.82050 ms`、iTransformer `.15747 ms`、DLinear `.09451 ms`。它快于 PatchTST/TimesNet，却慢于 iTransformer/DLinear；结合无精度优势，也不保留为高效论文主候选。资源工件为 [`RaMambaLite resource benchmark`](../../artifacts/exploratory/ramamba_lite_autocon_h96_screen_20260925/resource_benchmark_etth1_h96/model_resources.json)。实际训练与资源测量均在 `cuda:0` RTX 5090 D、PyTorch `2.8.0+cu128`、`mamba-ssm=2.3.1` 完成。

## 2026-09-25 MambaTS v7 fresh-seed 长预测确认：触发停止规则

冻结 v7 架构后，已完成 H336/H720、ETTh1/ETTm1/Electricity/Traffic、fresh seed `2024/2025/2026` 的 `120/120` validation-only 矩阵，与 DLinear、PatchTST、TimesNet、iTransformer 使用单一源码快照和同一训练合同。所有任务为 `status=ok`、`returncode=0`，严格读出确认 120 个唯一坐标、零测试指标工件、全部 v7 train-only ACF sidecar，以及有限的 ACF telemetry。读出见 [`v7_fresh_long_readout`](../../artifacts/exploratory/mambatsv7_multihorizon_fresh_seed_confirmation_20260925/readout/readout.md)，冻结计划见 [`mambatsv7_multihorizon_fresh_seed_confirmation_20260925.json`](mambatsv7_multihorizon_fresh_seed_confirmation_20260925.json)。

v7 在 ETTm1 H336/H720 为 `-2.850%/-2.114%`（均 3/3 seed）、Electricity H336 `-1.697%`（3/3）、Traffic H336/H720 `-8.676%/-7.957%`（均 3/3）；但 ETTh1 H336/H720 为 `+6.395%/+2.667%`，Electricity H720 为 `+3.616%`，这三项均为 `0/3` seed 胜出。依据计划的“在八个 dataset-horizon 单元中落后最强 baseline 三项即停止”规则，停止 v7 的论文主候选路径：不再借助这些 fresh seeds 调 ACF lag、selector、AutoCon 权重或训练预算，也不进行 v7 测试释放。虽然 v7 在 Electricity 长预测的完整作业时间更短、在 Traffic/ETTm1 保持精度优势，但它没有通过跨域长预测非劣门槛，更不能支持“统一高水平论文主结果”。

## 2026-09-24 MambaTS v7 同快照强基线复验：精度三胜一负，速度目标未达成

修复 MambaTS 缺失 ACF runtime telemetry 后，完成同一源码快照的 `72/72` validation-only 矩阵：DLinear、PatchTST、TimesNet、iTransformer、v6、v7，覆盖 ETTh1、ETTm1、Electricity、Traffic 的 H96 和三个 seed。三个 phase 全部 `status=ok`、`returncode=0`、源码快照一致，严格读取器确认零测试指标工件；v7 的 train-only sidecar 存在，`acf_runtime_gate_mean`、`acf_effective_memory_rms`、`acf_effective_memory_to_forecast_rms` 均为有限值。完整表见 [`same_snapshot_v7_readout`](../../artifacts/exploratory/mambatsv7_same_snapshot_baseline_confirmation_20260924/readout/readout.md)，计划见 [`mambatsv7_same_snapshot_baseline_confirmation_20260924.json`](mambatsv7_same_snapshot_baseline_confirmation_20260924.json)。

相对该矩阵各数据集最强 baseline，v7 在 ETTm1 `-2.047%`、Electricity `-7.503%`、Traffic `-11.751%`，但 ETTh1 落后 PatchTST `+1.511%`。相对 v6，四个数据集均未退化（`-1.158%/-2.122%/-0.102%/-0.060%`），且 v7 ACF memory/forecast RMS 的跨 seed 均值为 `.0765/.0479/.0614/.1258`，说明 ACF 路径不是零贡献分支。仍不能把这写成普适非劣：H96 是唯一 horizon，ETTh1 失败，且尚无 fresh-seed 或测试集证据。

Traffic/C862/H96/B8、50 warm-up/200 timed、prediction-only 同快照资源显示 v7 中位延迟 `.82080 ms`、参数 `128,990`；PatchTST `.88974 ms`、`46,080` 参数，TimesNet `2.82805 ms`、`1,302,046` 参数，iTransformer `.34550 ms`、`14,944` 参数，DLinear `.10282 ms`、`18,624` 参数。v7 的确快于 PatchTST/TimesNet，但慢于 iTransformer/DLinear，故不满足“总体快于所有强基线”的目标。资源工件见 [`model_resources.json`](../../artifacts/exploratory/mambatsv7_same_snapshot_baseline_confirmation_20260924/resource_traffic_h96/model_resources.json)。下一阶段应固定 v7，不再调 H96，使用 fresh seeds 和 H336/H720 验证 ETTh1 失败是否持续；只有跨 horizon 达到预设精度门槛，才讨论进一步压缩 v7 的 ACF selector/表示路径。

## 2026-09-24 MambaTS v7 ACF/线性基座 screen：验证信号积极，但尚未达到论文结论

已完成注册的 `24/24` validation-only 配对 screen：v6 线性基座控制臂 vs v7「同一线性基座 + train-only raw ACF memory + AutoCon 表示选择修正」，覆盖 ETTh1、ETTm1、Electricity、Traffic 的 H96、三个 seed。所有任务 `status=ok`、`returncode=0`、坐标唯一、源码快照前后哈希一致；`skip_test=true` 且测试指标工件为零。严格读出见 [`v7_acf_linear_base_readout`](../../artifacts/exploratory/mambatsv7_acf_linear_base_screen_20260924/readout/readout.md)，计划与边界见 [`mambatsv7_acf_linear_base_screen_20260924.json`](mambatsv7_acf_linear_base_screen_20260924.json)。

相对 v6，v7 的 mean validation MSE 在 ETTh1 `0.704112 -> 0.695958`（`-1.151%`，3/3 seed）、ETTm1 `0.396339 -> 0.387929`（`-2.103%`，3/3）改善；Electricity `-0.104%`、Traffic `-0.052%` 基本持平，分别仅 `1/3` 与 `2/3` seed 改善。平均作业墙钟时间则为 ETTh1 `+10.646%`、ETTm1 `+11.326%`、Electricity `-12.987%`、Traffic `+5.297%`，不能宣称普遍更快。Traffic/C862/H96/B8、50 warm-up/200 timed 的 prediction-only GPU 资源显示 v6/v7 中位延迟 `0.71336/0.79062 ms`、参数 `121,887/128,990`，即 v7 约 `+10.9%` 延迟、`+5.8%` 参数；资源工件见 [`model_resources.json`](../../artifacts/exploratory/mambatsv7_acf_linear_base_screen_20260924/resource_traffic_h96/model_resources.json)。

本轮训练日志中的 `acf_runtime_gate_mean`、`acf_effective_memory_rms` 等遥测均为 `None`，所以目前只能确认配置和 ACF sidecar 存在，不能声称 ACF residual 在训练中产生了可量化的有效贡献。该候选暂定为“值得做同快照强基线复验”的方向，不进入测试集释放，也不能写成已达到外部 baseline 非劣或论文发表要求。

## 2026-09-24 MambaACF Lite AutoCon：效率达标但精度失败，停止

新的、授权且可追溯的 `12/12` validation-only screen 比较 full `MambaACFAutoCon` 与固定 Lite 别名，覆盖 ETTh1/Traffic、H96、三个 seed。两臂都使用相同的 train-only raw ACF sidecar、`lambda=.03` 的 active AutoCon lag-pair relation、MeanStd、优化器、数据划分和随机种子；Lite 仅固定为一条 Mamba scan 与 rank-8 预测头。全部记录 `status=ok`、`returncode=0`、源码快照一致，12 个 ACF sidecar 和 loss log 均存在，测试指标工件为零。relation telemetry 并非退化：Lite 的 global contrastive-active anchor 均值为 ETTh1 `87.27%`、Traffic `84.21%`，degenerate-positive 均值为 `5.08%/8.40%`。

Lite 在 ETTh1/H96 的 mean validation MSE 为 `.723684`，相对 full `.735669` 为 `-1.627%`（`2/3` seed 改善）；但 Traffic/H96 为 `.437579` 对 `.385874`，即 `+13.409%` 且 `0/3` seed 改善。其完整作业墙钟在 ETTh1/Traffic 分别为 `-0.822%/-11.474%`，Traffic/C862/H96/B8、50 warm-up/200 timed 的同快照 prediction-only GPU 资源也由 full 的 `1.27824 ms`、`159,704` 参数降至 Lite 的 `.73890 ms`、`132,139` 参数（约 `-42.2%` 前向延迟、`-17.3%` 参数）。速度收益真实但精度仅在低维数据成立，故按预先停止规则停止 Lite 分支，不扩大 horizon、不调 Lite rank、也不启动测试释放。严格读数见 [`lite_autocon_readout`](../../artifacts/exploratory/mamba_acf_lite_autocon_screen_20260924/readout/readout.md)，资源工件见 [`model_resources.json`](../../artifacts/exploratory/mamba_acf_lite_autocon_screen_20260924/resource_traffic_h96/model_resources.json)，计划与授权记录见 [`mamba_acf_lite_autocon_screen_20260924.json`](mamba_acf_lite_autocon_screen_20260924.json)。

## 2026-09-24 iTransformer PEMS 基线扩展：完成，仅作快速基线

轻量 iTransformer 已完成 PEMS03/04/07/08、H12/H96、五个 seed 的 `40/40` validation-only 作业。每个作业均为 `status=ok`、`returncode=0`，40 个 dataset-horizon-seed 坐标无重复；实际运行环境为 `cuda:0`（RTX 5090 D、PyTorch `2.8.0+cu128`），`skip_test=true`，专用检查确认零测试指标工件。其 PEMS 数据哈希与既有 Mamba 矩阵及 TimeMixer 扩展一致；Mamba 参考来自不同源码快照，故跨快照精度对比仅作描述性读数。

该配置在八个坐标全部落后于已有最佳参考：PEMS03/04/07/08 的 H12 分别为 `+27.246%/+33.950%/+19.857%/+11.044%`，H96 为 `+121.847%/+181.670%/+81.602%/+43.991%`（正数表示 iTransformer MSE 更高）。不过，在相同 PEMS07/C883/H96/B2、50 warm-up/200 timed、device-resident prediction-only 协议下，iTransformer 中位前向延迟为 `.21232 ms`、参数 `14,944`，快于既有 MambaTS-v3-null 的 `.32693 ms`。因此它是必要的快速/轻量基线，而不是当前 Mamba 主模型候选或测试释放对象。完整读数见 [`itransformer_baseline_readout`](../../artifacts/exploratory/itransformer_pems_baseline_extension_20260924/readout/readout.md)，资源工件见 [`model_resources.json`](../../artifacts/exploratory/itransformer_pems_baseline_extension_20260924/resource_pems07_h96/model_resources.json)，执行协议见 [`itransformer_pems_baseline_extension_20260924.md`](itransformer_pems_baseline_extension_20260924.md)。

## 2026-09-23 TimeMixer 基线扩展：完成，停止作为主候选

新增的 TimeMixer-style 多尺度分解/MLP mixing 基线已完成 PEMS03/04/07/08、H12/H96、五个 seed 的 `40/40` validation-only 作业。所有任务实际使用 `cuda:0`（RTX 5090 D、PyTorch `2.8.0+cu128`），`skip_test=true`，无测试指标工件。严格读取器同时核验了 40 个坐标、训练日志、有限验证损失以及与既有矩阵一致的数据集哈希；由于源码快照不同，结果只作描述性比较。

TimeMixer 的均值验证 MSE 相对每个坐标最佳参考均退化：PEMS03/04/07/08 的 H12 为 `+8.416%/+8.258%/+9.621%/+4.184%`，H96 为 `+95.349%/+64.464%/+55.349%/+32.009%`。它的平均训练作业时间通常低于 TimesNet/PatchTST/MambaTS，但这不足以抵消长预测精度损失；同形状 PEMS07/C883/H96/B2 的 device-resident 前向中位延迟为 `.51338 ms`、参数 `183,171`，也慢于既有 MambaTS-v3-null 的 `.32693 ms`。因此不进入 Mamba 主模型或测试释放。完整读数见 [`timemixer_baseline_readout`](../../artifacts/exploratory/timemixer_pems_baseline_extension_20260923/readout/readout.md)，资源工件见 [`model_resources.json`](../../artifacts/exploratory/timemixer_pems_baseline_extension_20260923/resource_pems07_h96/model_resources.json)，协议见 [`timemixer_pems_baseline_extension_20260923.md`](timemixer_pems_baseline_extension_20260923.md)。

## 2026-09-23 MambaTS v3 AutoCon 低秩通道适配器：确认失败，停止

冻结的 `240/240` validation-only 确认矩阵已经完成，覆盖 PEMS03/04/07/08、H12/H96、五个 seed，以及 DLinear、PatchTST、TimesNet、Mamba2、精确 MambaTSv3AutoCon null 和 rank-8 adapter。独立读取器验证了统一源码快照、完整的 240 条记录和零测试指标伪迹。适配器相对精确 null 在 6/8 单元改善：PEMS03 H12/H96 `-0.257%/-2.953%`，PEMS04 `-0.268%/-1.216%`，PEMS07 `-0.395%/-2.264%`；但 PEMS08 H12/H96 分别退化 `+1.244%/+2.160%`。同时，PEMS04/H96 比最佳 TimesNet 高 `9.820%`，不满足预注册的 2% 非劣界。因此不能借助测试集或再调 PEMS08 来挽救该候选，停止其架构扩展。

资源门槛本身通过：在 PEMS07/C883/H96、B2、50 warm-up/200 timed 的 prediction-only device-resident 协议下，null/adapter 中位延迟为 `.32693/.34517 ms`（`+5.58%`），参数 `104,291/105,411`，峰值 allocated increment 同为 `38.854 MB`。这只说明轻量读出没有破坏推理效率，不构成端到端训练速度或泛化精度主张。人工复核文件为 [`mamba_ts_v3_channel_adapter_confirmation_20260923.md`](mamba_ts_v3_channel_adapter_confirmation_20260923.md)，原始读取结果为 [`readout.md`](../../artifacts/exploratory/mamba_ts_v3_channel_adapter_confirmation_20260923/readout/readout.md)。

## 2026-09-23 六模型基线扩展：完成，plain Mamba 停止扩展

独立 provenance 根的 frozen exploratory matrix 已完整结束：ETTh1、ETTm1、Electricity、Traffic，M 特征，H96/H336/H720，seed `2021/2022/2023`，覆盖 DLinear、NLinear、PatchTST、TimesNet、Mamba-1、Mamba-2，共 `216/216` 个 validation-only 作业。phase validator 为 `valid=true`，36 个模型-数据集-horizon 坐标覆盖完整，所有作业 `skip_test=true`，专用分析器确认没有测试指标工件。机器读数见 [`readout.md`](../../artifacts/exploratory/mamba_multidataset_plain_baseline_expansion_20260923/readout/readout.md)。

普通 Mamba 没有赢得任何一个 12 个数据集-horizon 坐标：相对每个坐标最优模型，Mamba-1/Mamba-2 的平均 validation MSE 缺口为 `+8.18%/+7.53%`，最近的单元仍分别为 `+1.79%/+2.48%`。PatchTST 赢得 8 个坐标、TimesNet 赢得 4 个坐标。相同短训练合同下，Mamba-1 平均作业墙钟时间 `34.4 s`，低于 PatchTST 的 `41.6 s`，但高于 DLinear/NLinear 的 `31.2/30.4 s`；Mamba-2 为 `50.8 s`。这些是短预算端到端作业时间，不是同步吞吐或硬件无关速度主张。

因此不再通过更换 Mamba-1/2 版本或微调 plain head 来追逐主结果。后续候选必须保留 AutoCon 的真实表示目标，并对预测信息路径提出可证伪的结构变化，同时与 v3 的快速高维路径和 DLinear/PatchTST/TimesNet 同合同配对。Traffic/C862/H96 的 device-resident resource preflight 仍可作为结构成本参考：NLinear/DLinear/Mamba-1/Mamba-2/PatchTST/TimesNet 的中位预测延迟分别为 `.0336/.0785/.4265/.7117/.8818/2.7905 ms`（B8），不等价于端到端速度。计划见 [`mamba_multidataset_plain_baseline_expansion_plan_20260903.json`](mamba_multidataset_plain_baseline_expansion_plan_20260903.json)。

## 2026-09-23 低秩通道适配器：探索阶段记录（已由后续确认否决）

已实现并通过合同测试的 `MambaTSv3AutoCon` 可选分支，将现有共享 Mamba 扫描后的每通道 patch 状态做时间池化，再经过共享 rank-8 瓶颈生成预测残差。它不增加第二次 Mamba 扫描，且以零初始化 horizon gate 严格嵌套在 null；AutoCon 表示和 forecast MSE 共同作用于同一编码状态。32-job PEMS validation-only screen 已 `32/32` 成功：PEMS03/04/07 在 H96 分别改善 `2.308%/2.186%/1.658%`，并满足预设“至少两个数据集、两个 horizon 改善”规则；PEMS08 H12/H96 分别退化 `0.307%/0.382%`，必须保留为失败分析。固定 PEMS07 形状的合成前向为 `0.32765 -> 0.34851 ms`（`+6.37%`），训练墙钟增加 `0.83%--3.60%`；这些不是端到端速度或测试集结论。实现和完整读数见 [`mamba_ts_v3_channel_adapter_screen_20260923.md`](mamba_ts_v3_channel_adapter_screen_20260923.md)。

基于该探索结果，已冻结独立确认边界 [`mamba_ts_v3_channel_adapter_confirmation_20260923.md`](mamba_ts_v3_channel_adapter_confirmation_20260923.md) 和 240-job 计划 [`mamba_ts_v3_channel_adapter_confirmation_20260923.json`](mamba_ts_v3_channel_adapter_confirmation_20260923.json)。该矩阵现已完成，但未通过预设精度/null 回归门槛；结论与停止决定见本文顶部，不能设计测试释放。

下一阶段论文级实验边界已整理为 [`paper_grade_experiment_boundary_20260923.md`](paper_grade_experiment_boundary_20260923.md)：优先补齐同一 provenance 下的 NLinear 和六个 plain baseline，冻结验证选择规则后再开放测试；不再扩展已停止的 residual-init 或通道选择器方向。

## 2026-09-23 Exchange 通道级误差诊断：不支持通道选择器

对 v6 与 PatchTST 的 Exchange H96 validation checkpoint 做三 seed、逐通道 MSE 汇总。v6 平均 `0.270135`，PatchTST `0.259005`；v6 在 8 个通道均未取得三 seed 平均优势，误差主要集中在前 4 个通道。因此当前没有证据表明缺口来自少数异常变量，不增加通道选择器/门控搜索。详见 [`mambatsv6_residual_init_screen_20260923.md`](mambatsv6_residual_init_screen_20260923.md)。

## 2026-09-23 MambaTS v6 残差初始化：停止

在 v6 Exchange checkpoint 中已学习 residual scale 约为 `0.10--0.12` 且关闭残差会使 validation MSE 上升后，进行了唯一预记录的 warm 初始化 `0.1` 对 zero-init 的 Exchange/ILI H96 三 seed筛选。`36/36` 作业成功、测试关闭。warm 在 Exchange 仅 `-0.181% (2/3)`，仍落后 PatchTST `4.111%`；在 ILI 为 `+1.211% (0/3)`，并落后 NLinear `3.694%`。因此不扩大初始化网格，也不进入测试释放；详见 [`mambatsv6_residual_init_screen_20260923.md`](mambatsv6_residual_init_screen_20260923.md)。

## 2026-09-23 MambaTS v6 线性初始化残差：停止

MambaTS v6 把 DLinear 风格 trend/seasonal 分解设为主预测，并以零初始化 Mamba 残差修正，同时保留真实时间索引的 AutoCon 表示训练。四数据集 H96、七模型、三 seed、20 epoch 的冻结 validation-only 矩阵已 `84/84` 完成且无测试工件。它在 Weather 为七臂最佳（`0.443833`，相对 v3 `-0.395%`），ILI 也优于 DLinear（`-1.401%`），但仍落后 NLinear `2.450%`；Exchange 仍落后 PatchTST `4.299%`，ETTh1 落后 iTransformer `3.166%`。Weather/C21 同形状前向为 `0.372 ms`，慢于 v3 `0.278 ms`、PatchTST `0.281 ms` 和 DLinear `0.095 ms`。因此停止测试释放和长预测；详见 [`mambatsv6_h96_screen_20260923.md`](mambatsv6_h96_screen_20260923.md)。

## 2026-09-22 MambaSM 双向变量扫描：停止

本地 `MambaSM/MambaSMAutoCon` 完成 ETTh1、Weather、Exchange、ILI 的 H96、三 seed、20 epoch validation-only 筛选（`72/72` 严格验证成功，测试关闭）。该 S-Mamba/TimesMamba 启发候选使用逐变量 token 的前向/反向 Mamba-1 scan，并仍产生真实的逐时间 AutoCon 表示。它只在 ETTh1 稳定优于 MambaTS v3（`-2.718%`，`3/3`），但仍落后 iTransformer `1.169%`；Weather、Exchange、ILI 相对 MambaTS v3 分别为 `+1.846%`、`+1.535%`、`+6.149%`。虽然参数量较低（C7 `38,176` 对 `61,016`），同形状前向延迟不占优，C21 为 `0.399 ms` 对 `0.307 ms`。因此停止长预测和测试释放，只保留结构基线；详见 [`mambasm_h96_screen_20260922.md`](mambasm_h96_screen_20260922.md)。

## 2026-09-22 TimeMachine 四路径 Mamba 基线：停止

TimeMachine 风格的四路径 Mamba-1/Mamba-2 及 AutoCon 接口已完成首次统一强基线矩阵：ETTh1、Weather、Exchange、ILI，H96，九模型臂、三 seed，`108/108` 严格验证通过且测试关闭。它不是新的主干：MambaTM1/MambaTM2 在 ETTh1 分别为 `0.693979/0.710837`，仍高于 iTransformer `0.683150`；Weather 为 `0.480892/0.484294`，明显高于 MambaTS v3 `0.445621`；Exchange/ILI 均大幅落后线性/patch 基线。MambaTM2 的完整训练 job 时间在四数据集均远高于 MambaTS v3（例如 Weather `125.01s` 对 `34.33s`）。因此只保留为公开结构基线，不做长 horizon 或测试释放；下一候选必须保持单一高效 temporal scan 并改进受控跨变量路径。详见 [`mambatm_h96_baseline_extension_20260922.md`](mambatm_h96_baseline_extension_20260922.md)。

## 2026-09-22 表示门控线性专家：停止

针对 Exchange/ILI 低维失败坐标，已完成 AutoCon 表示门控共享线性专家的严格 validation-only 配对重试。原始 expert-on 任务曾因验证阶段未构建内部 representation 而全部失败；该错误已修正并由 `17` 个 MambaTS contract tests 覆盖。失败工件保留，修正后的 expert-on 在独立 snapshot/output root 下成功 `12/12`，原始 expert-off control 也通过 `12/12` 完整性校验；两臂均无 test 指标。结果不支持该混合：Exchange H96/H336 为 `+0.976%/+1.697%`（均 `0/3`），ILI H24 为 `-1.190%`（`2/3`），ILI H60 为 `+2.096%`（`0/3`）。因此不扩展、不进入测试释放，也不能解释为 AutoCon 的稳定增益；实现只保留为负消融。详见 [`mambatsv3_repr_linear_expert_retry_20260922.md`](mambatsv3_repr_linear_expert_retry_20260922.md)。

## 2026-09-22 低维长预算复验

Exchange/ILI 的 36-job、20 epoch validation-only 复验已完成（两个 phase 均 `18/18`）。MambaTS v3 AutoCon 相对 DLinear/PatchTST 的差距没有因预算增加而缩小：Exchange H96/H336 为 `+5.88%/+7.46%` 对 DLinear，ILI H24/H60 为 `+8.39%/+9.80%`；与 PatchTST 也分别为 `+6.81%/+8.88%` 和 `+10.90%/+9.91%`。这排除了“八 epoch 不足”解释。下一架构候选应是由 AutoCon 表示控制的轻量线性专家/Mamba 混合门控，必须和 no-gate/no-expert 控制配对验证。

## 2026-09-22 Transfer baseline extension

新增的 Weather/Exchange/ILI 144-job validation-only 矩阵已经完成并通过两个 `72/72` validator。MambaTS v3 AutoCon 在 Weather H96/H336 均为六臂最佳，但 Exchange H96/H336 分别落后 PatchTST `5.87%/8.17%`，ILI H24/H36/H48/H60 分别落后最佳线性/patch 基线 `7.06%/8.20%/10.28%/9.10%`，且均为 `0/3` seed 胜出。所有结果保持测试关闭。因此该结构不能写成跨域普适候选；下一步先验证这是否只是八 epoch 预算不足，再决定是否需要新的低维/线性结构分支。详见 [`mambatsv3_transfer_baseline_extension_20260922.md`](mambatsv3_transfer_baseline_extension_20260922.md)。

## 2026-09-22 Weather held-out test release

冻结的 75-job Weather 测试释放已完成：active MambaTS v3 AutoCon（`lambda=.001`）与 PatchTST、TimesNet、iTransformer、Mamba-2，覆盖 H96/H336/H720、五个 fresh seed、20 epoch 最大预算。两个 phase validator 均为 `valid=true`（`15/15` 与 `60/60`），每个 cell 的候选均为 `5/5` seed 较低 MSE，并通过预注册 `1.02` 单侧非劣门槛。候选的 prediction-only latency 在三长度均低于 PatchTST/TimesNet/Mamba-2，但高于 iTransformer；manifest end-to-end job wall time 同样低于四者。该结论仅限 Weather 和此训练协议；此前 `lambda=0` null 在 H336/H720 不稳定，故不能把本次 baseline 胜出归因为 AutoCon 的独立增益。详见 [`mambatsv3_weather_test_release_20260922.md`](mambatsv3_weather_test_release_20260922.md)。

## 2026-09-22 Weather 多预测长度确认 screen

Weather 的 63-job validation-only screen 已完成并通过四个 phase validator：MambaTS v3 plain、AutoCon `lambda=0` null、active `lambda=0.001` 以及 PatchTST/TimesNet/iTransformer/Mamba-2，覆盖 H96/H336/H720 和三 seed。active AutoCon 相对 null 在 H96 改善 `-0.358% (3/3)`，但 H336/H720 分别为 `+0.066% (1/3)`、`+0.289% (0/3)`；因此不能声称 AutoCon 对长预测有独立稳定增益。候选在该短预算 Weather validation screen 上三个 horizon 均优于四个外部基线，但这是描述性结果，尚无 test 指标。H720 prediction-only 资源为 `0.272 ms` median，略快于 PatchTST `0.281 ms`、明显快于 Mamba-2 `0.697 ms`，但慢于 iTransformer `0.151 ms`。完整表格见 [`mambatsv3_weather_multihorizon_confirmation_screen_20260922.md`](mambatsv3_weather_multihorizon_confirmation_screen_20260922.md)。

## 2026-09-22 MambaTS v3 主干与 AutoCon null screen

> 新的 24-job `MambaTSv3AutoCon` H96 screen 已完成 `24/24`，后续 H336/H720 三臂 screen 也已完成 `72/72`。同一架构的 `lambda=0` 与 `lambda=0.001` 在 H96 的 paired mean relative change 为 `+0.009%/+0.006%/+0.008%/+0.004%`；长预测八个数据集-horizon 单元仅 `-0.005%` 至 `+0.110%`，没有一致胜出。因此 AutoCon 表示目标确实执行，但当前不能宣称独立精度增益。更有价值的是 v3 forecast backbone 的 H96 描述性结果：相对同坐标最佳外部 baseline，ETTh1 `+3.54%`、ETTm1 `-0.89%`、Electricity `-6.12%`、Traffic `-11.53%`。同日 v4 train-only ACF memory screen 在 Traffic 退化约 `+0.64%`，所以 ACF memory 暂不作为快速主线默认组件。完整结果见 [`mamba_ts_v3_multihorizon_autocon_screen_20260922.md`](mamba_ts_v3_multihorizon_autocon_screen_20260922.md)。

> 预测连接的 v3 表示融合已完成新的 `24/24` H96 screen：`lambda=.003` 相对精确 null 在 Electricity `-0.666% (3/3)`、Traffic `-0.055% (3/3)`，但 ETTh1 `+0.229%`、ETTm1 `+0.081%`，故只构成高维候选信号。全局 fusion Traffic resource smoke 为 `0.720 ms/152.6 MB`；通道保留 fusion 的完整时间版训练超过 7 分钟被停止，patch 版仍为 `1.415 ms/343.7 MB`，不满足效率目标。详见 [`mamba_ts_v3_repr_forecast_fusion_screen_20260922.md`](mamba_ts_v3_repr_forecast_fusion_screen_20260922.md)。

> 该全局融合已完成独立 `60/60` λ/warm-up 校准：`.0003/.001/.003/.003-warm4` 没有一个在四数据集同时无退化，warm-up 仅在 ETT 改善而在 Traffic 退化。因此不再扩大当前损失权重搜索，也不开放测试集；完整审计见 [`mamba_ts_v3_global_repr_autocon_calibration_20260922.md`](mamba_ts_v3_global_repr_autocon_calibration_20260922.md)。

## 2026-09-10 MambaACF 多数据集候选 screen

> 新的独立 fresh provenance 候选矩阵 [`mamba_acf_multidataset_candidate_screen_20260910.md`](mamba_acf_multidataset_candidate_screen_20260910.md) 已完成 `48/48` 个 validation-only 作业：四个数据集、H96、三个 seed，覆盖 ACF-off、train-only ACF memory、AutoCon relation `lambda=0` null 和 `lambda=0.001`。候选 `MambaACFAutoCon` 在 Electricity/Traffic 的 H96 均值低于同坐标 plain screen 的最佳外部 baseline，但 ETTh1/ETTm1 仍分别比 PatchTST 高约 `6.43%/7.06%`。严格 null-to-relation 配对只有 `-0.04%/-0.01%/-0.15%/+0.02%`（ETTh1/ETTm1/Electricity/Traffic），所以当前不能宣称对比损失带来独立稳定增益。测试工件为 0，尚无 H336/H720 或速度结论；计划仍是 `authorized=false` 的 exploratory snapshot，下一步只能在人工复核后做冻结的长 horizon 复验。

## 2026-09-10 complete plain-baseline matrix

> 新的独立 fresh provenance 矩阵 [`mamba_multidataset_plain_baseline_full_screen_20260910.md`](mamba_multidataset_plain_baseline_full_screen_20260910.md) 已完成 `216/216` 个 validation-only 作业：ETTh1、ETTm1、Electricity、Traffic × H96/H336/H720 × seed 2021/2022/2023 × DLinear、NLinear、PatchTST、TimesNet、Mamba1、Mamba2。`phase_validation.json` 为 `valid=true`，`status_counts.ok=216`，所有记录的 source-tree hash 相同，测试指标为 0。结果不支持 plain Mamba 普遍优于基线：ETTh1 三个 horizon 都由 PatchTST 最好；Electricity 三个 horizon 都由 TimesNet 最好；Traffic 的 PatchTST/TimesNet 仍明显优于 Mamba；Mamba-2 主要在高维 Electricity/Traffic 上优于 DLinear/NLinear。该矩阵建立了后续 AutoCon/ACF 候选的同坐标 baseline contract，但本身是短预算 validation screen，不是测试集或发表结果。

## 2026-09-09 当前代码边界与回归状态

> 状态校正（2026-09-10 本轮）：在加入 MambaTS v2/v3/v4/v5、PEMS 强基线与 TimesNet fresh-seed screen、lambda 校准、TimesNet 交叉分析、ACF representation-fusion 与 launcher relocation contract 测试后，完整回归为 `633 passed, 43 warnings`。

> Lite AutoCon resource preflight (2026-09-09): 新增公开别名 `MambaACFLiteAutoCon`，固定为单尺度 Mamba-2、因果 embedding、rank-8 双时间投影，并保留训练时 AutoCon ACF lag-pair relation。ETTh1/Traffic 的同形状 prediction-only smoke 显示，相对 full `MambaACFAutoCon` median latency 降低约 `40.4%/42.6%`，可训练参数降低约 `55.8%/17.3%`；但 Lite 仍略慢于普通 Mamba-2，且 ETTh1 慢于 PatchTST。低秩 head 的历史 screen 曾退化 `+1.095%` MSE，因此本次只是待证伪结构候选。24-job H96 三 seed 计划 [`mamba_acf_lite_autocon_pilot_20260909.json`](mamba_acf_lite_autocon_pilot_20260909.json) 已 validate-only 通过，`execution.authorized=false`，尚未训练、尚未产生验证或测试精度；资源人工复核见 [`mamba_acf_lite_autocon_resource_preflight_20260909.md`](mamba_acf_lite_autocon_resource_preflight_20260909.md)。

> MambaTS baseline update (2026-09-10): 新增 `MambaTS`/`MambaTSAutoCon` 以及独立的 `MambaTSv2`/`MambaTSv3` 入口。实现保留官方 MambaTS 可辨识的 patch token、按变量组织的 Mamba-1 scan 和 per-variable head，但明确标记为 MambaTS-inspired，不复刻外部 VAST/VPT/ATSP。四个 PEMS 数据集的 hash、feature-0 选择和 loader 审计见 [`pems_dataset_audit_20260909.md`](pems_dataset_audit_20260909.md)。旧的 192-job 计划仍保持 `execution.authorized=false`；当前 v3 的独立 exploratory 结果见 [`mamba_pems_ts_v3_strong_baseline_screen_20260909.md`](mamba_pems_ts_v3_strong_baseline_screen_20260909.md) 与 [`mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md`](mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md)。

> MambaTS resource smoke (2026-09-09): 固定 synthetic GPU 前向中，ETTh1/C7 的 `MambaTS` median `0.2176 ms`、PEMS03/C358 batch=2 的 median `0.2250 ms`；`MambaTSAutoCon` 分别为 `0.2210/0.2226 ms`。同一 PEMS03 shape 下 MambaTS 比 Mamba-2 快约 `69.4%`、比 PatchTST 快约 `21.7%`，但 DLinear latency 低 `58.0%`；这些数字只覆盖 prediction-only device-resident forward，不能写成训练更快或模型精度更高；详细协议见 [`mamba_ts_resource_smoke_20260909.md`](mamba_ts_resource_smoke_20260909.md)。

> MambaTS v3 validation update (2026-09-09): `MambaTSv3` 保留 v1 的 shared channel-major scan，并增加 zero-initialized low-rank cross-variable context 与 per-channel persistence gate；v2 的 independent scan 因高维运行时间过高而不作为主线。PEMS03/04/07/08、H=12/96、四 epoch 的 plain screen 中，v3 相对 Mamba-2 为 `8/8` 个坐标改善（seed 2021），两个 fresh seeds 的独立配对为 `16/16` 改善，平均相对 MSE 约 `-8.22%`。四数据集强基线 screen（DLinear/PatchTST/TimesNet/Mamba2/v3，seed 2022）显示 v3 相对 DLinear/PatchTST 均为 `8/8` 改善，相对 TimesNet 为 `5/8`；TimesNet 在 PEMS03/04/08 的 H=96 更低，所以不能写成普适胜出。详细记录见 [`mamba_pems_ts_v3_fresh_seeds_20260909.md`](mamba_pems_ts_v3_fresh_seeds_20260909.md) 和 [`mamba_pems_ts_v3_strong_baseline_screen_20260909.md`](mamba_pems_ts_v3_strong_baseline_screen_20260909.md)。

> TimesNet fresh-seed update (2026-09-09): 对 `MambaTSv3`/TimesNet 在 PEMS 四数据集、H=12/96 的 seed 2023 复跑中，v3 为 `6/8` 配对改善（H=12 为 `4/4`，H=96 为 `2/4`）；PEMS03/04 的 H=96 仍由 TimesNet 胜出。与 seed 2022 screen 合并的描述性结果为 `11/16` 胜出，但 H=96 仅 `3/8`，因此 TimesNet 是长期主表的必要 comparator，v3 的可辩护定位是短期候选/长期竞争者。记录见 [`mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md`](mamba_pems_ts_v3_timesnet_fresh_seed_20260909.md)。

> MambaTS v3 + AutoCon update (2026-09-09): 同一 PEMS/seed 2022 配对中，v3 plain 相对 Mamba-2 为 `8/8` 改善（平均 `-9.12%`）；给 v3 加现有 AutoCon 目标后仅 `1/8` 坐标改善，平均变化 `+1.64%`，因此当前证据不支持 AutoCon 对 v3 有独立精度收益。`MambaTSv3AutoCon` 相对 `Mamba2AutoCon` 为 `8/8` 改善，但该差异混合了架构和目标两个因素；记录见 [`mamba_pems_ts_v3_autocon_screen_20260909.md`](mamba_pems_ts_v3_autocon_screen_20260909.md)。

> MambaTS v3 AutoCon lambda calibration (2026-09-10): 针对上述目标干扰，新建两个冻结 validation-only 矩阵，分别为 seed 2022 的 40 jobs（lambda `0/0.001/0.003/0.01/0.03`）和 seed 2023 的 24 jobs（lambda `0/0.001/0.003`），均为 `valid=true`、无测试指标。相对同结构 `lambda=0`，`lambda=0.001` 在合并 16 个精确坐标中 `11/16` 改善，平均变化 `-0.0095%`；H=12 近乎持平，H=96 为 `-0.0181%`。AutoCon loss 保持非零，说明表示目标确实执行，但当前证据只能支持“低权重近似保持预测性能”，不能支持“AutoCon 提升精度”。`lambda=0.03` 在首轮平均 `+0.3142%`，不再作为 v3 默认值。完整合同、工件和下一轮门槛见 [`mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md`](mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md)。

> Calibrated v3 AutoCon vs TimesNet (2026-09-10): 固定 `lambda=0.001` 的新 16-job screen 与 seed-2022/2023 TimesNet 工件按精确坐标配对，candidate 在 `10/16` 坐标改善；H=12 为 `8/8`、平均 `-19.50%`，H=96 为 `2/8`、平均 `+6.74%`。因此 AutoCon 保留了 v3 的短期优势，但没有解决长期预测短板；当前论文定位应明确为短期候选/长期受限，而不是全 horizon SOTA。完整读数见 [`mamba_pems_ts_v3_autocon_strong_baseline_20260910.md`](mamba_pems_ts_v3_autocon_strong_baseline_20260910.md)。

> MambaTS v4 AutoCon ACF screen (2026-09-10): 在 v3 上加入 train-only static ACF residual 和 zero-initialized ReZero gate 后，PEMS03/04/07/08、H=12/96、seed=2022/2023 的 16/16 作业成功，gate 在 16/16 checkpoint 中非零。相对 v3，H=12 为 `3/8`、平均 `+0.074%`，H=96 为 `5/8`、平均 `-0.057%`；相对 TimesNet，H=12 为 `8/8`、平均 `-19.449%`，H=96 为 `2/8`、平均 `+6.678%`。因此 ACF 分支确实参与优化，但没有证据证明它提升 v3，也没有解决 H=96 长期短板；当前只能作为 exploratory architecture candidate。读数与 replay 限制见 [`mamba_pems_ts_v4_autocon_screen_20260910.md`](mamba_pems_ts_v4_autocon_screen_20260910.md)。

> MambaTS v5 signed-anchor screen (2026-09-10): v5 保留 v4 的 patch/shared-scan/AutoCon/ACF-memory 路径，只增加 train-only causal strongest-lag signed anchor。PEMS03/04/07/08、H=12/96、seed=2022/2023 的 16/16 作业成功；相对 v4，H=12 为 `8/8` 胜但平均仅 `-0.0080%`，H=96 为 `4/8` 胜且平均 `+0.0017%`。anchor gate 在 16/16 行非零但幅度很小，平均 job wall-time 约增加 `0.61%`。该分支已作为中性消融停止扩展，不应作为主论文贡献。详见 [`mamba_pems_ts_v5_anchor_screen_20260910.md`](mamba_pems_ts_v5_anchor_screen_20260910.md)。

> Multidataset baseline execution smoke (2026-09-10): 独立 fresh output root 下对 DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2 的 24 个抽样任务全部成功，覆盖 ETTh1/ETTm1/Electricity/Traffic 与 H96/H336/H720；高维 H720 示例耗时约 `108--162s`。所有任务 `skip_test=true`，因此 smoke 只证明执行链路；详见 [`mamba_multidataset_plain_baseline_smoke_20260910.md`](mamba_multidataset_plain_baseline_smoke_20260910.md)。完整 216-job 结果另见上方的 full-screen readout，不能与 smoke 或历史矩阵混合。

> Long-context ACF screen (2026-09-10): 为使 PEMS 的 lag=288 日周期在 forecast lookup 中合法可用，固定 H=96、`seq_len=384`、lags `[24,168,288]`，对同窗口 v3/v4 AutoCon 做 8-job validation-only 配对。v4 仅 `1/4` 坐标胜出，平均相对变化 `+0.737%`；平均 wall-time `60.46s` 对 `58.56s`（约 `+3.2%`）。因此“扩大上下文后再加静态 ACF”没有通过当前精度/效率筛选，已停止扩展，不能与 `seq_len=96` 结果混排。详见 [`mamba_pems_ts_long_context_acf_screen_20260910.md`](mamba_pems_ts_long_context_acf_screen_20260910.md)。

> ACF representation-fusion screen (2026-09-10): 新增一个只影响 `MambaTSv4AutoCon` representation 的 raw ACF summary adapter，forecast backbone 保持不变。PEMS H=12/96、seed 2022/2023 共 16/16 作业成功，`ts_acf_repr_scale` 在 16/16 checkpoint 中非零（均值约 `0.2815/0.2801`）。相对原 v4，H=12 为 `3/8`、平均 `+0.075%`，H=96 为 `4/8`、平均 `-0.244%`；相对 TimesNet H=96 仍为 `2/8`、平均 `+6.379%`，候选平均 wall-time 增加约 `3.7%`。因此该路径证明 AutoCon 表示确实接收 ACF 信息，但没有达到精度晋级门槛；当前只保留为机制消融。详见 [`mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.md`](mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.md)。

> Direct-anchor update (2026-09-04): the current-tree standalone screen is summarized in [`mamba_acf_anchor_current_readout_20260904.md`](mamba_acf_anchor_current_readout_20260904.md). A separate frozen-source, fresh-seed held-out replication is summarized in [`mamba_acf_anchor_current_test_replication_readout_20260904.md`](mamba_acf_anchor_current_test_replication_readout_20260904.md): ETTh1 improved in `5/5` pairs (`-0.563%` scaled MSE) and Traffic in `4/5` (`-0.978%`), with `9/10` pooled wins (`-0.799%`). This supports promoting the scalar anchor to a controlled interaction study; it does not establish universal accuracy, SOTA superiority, or publication readiness. The replication is still exploratory because it covers only plain MambaACFVariable, H96, two datasets, and one frozen protocol.

> Interaction update (2026-09-07): the resulting 24-job anchor x soft-memory screen is recorded in [`mamba_acf_anchor_memory_interaction_screen_readout_20260904.md`](mamba_acf_anchor_memory_interaction_screen_readout_20260904.md). The combined arm is descriptively lower than plain (ETTh1 `-0.656%`, `3/3`; Traffic `-0.475%`, `2/3`), but the interaction is positive on ETTh1 (`+0.000486`, `0/3` super-additive) and negative on Traffic (`-0.001027`, `2/3`), with pooled CI crossing zero. There is no reliable synergy evidence and the combination must not be promoted or stacked. The plan was executed post-hoc while its immutable metadata remains draft; it is validation-only and uses `mamba_causal_embedding=false`, unlike the fresh direct-anchor test replication (`true`), so the two screens cannot be pooled. The plan-sign erratum is [`mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md`](mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md).

> Causality wiring correction (2026-09-07): the shared `_MambaForecastModel` now actually selects the existing `CausalDataEmbedding` when `mamba_causal_embedding=true`; this applies to ordinary Mamba-1/Mamba-2 and their AutoCon variants, not only MambaACF. The default remains the historical circular `DataEmbedding`, so old configurations retain their construction path. This is a code-contract correction, covered by regression tests and the full `613 passed, 43 warnings` suite; no new training run or accuracy claim has been made from it.

> Patch-token correction (2026-09-07): the first ETTh1/Traffic patch-token pilot used `hard_selected` with `mamba_acf_relation_lag_mask=false`, which the current v3 contract intentionally interprets as all-candidate supervision. A separate rerun with `lag_mask=true` and the same model/seed/short budget produced non-degenerate relation telemetry (`contrastive-active` about `84--87%`, `degenerate-positive` about `5--8%`) and patch-on validation-loss reductions of `2.79%` (ETTh1) and `3.24%` (Traffic). This remains one-seed, validation-only evidence; patch-on also added `6.47%/13.57%` short-run training wall time and `40.49%/65.88%` synthetic prediction latency. Full readout: [`mamba_patch_token_contrastive_pilot_readout_20260907.md`](mamba_patch_token_contrastive_pilot_readout_20260907.md). The earlier all-candidate pilot is retained separately and must not be mixed with this corrected relation contract.

> Patch-token resource preflight (2026-09-07): the four-dataset 24-job draft now has eight matched prediction-only resource records under [`mamba_patch_token_contrastive_screen_preflight_readout_20260907.md`](mamba_patch_token_contrastive_screen_preflight_readout_20260907.md). Patch-on increases median forward latency by `40.15--67.67%` and reduces synthetic sequences/s by `28.59--40.08%`; total parameters increase by `27.59--79.23%`. These are device-resident structural measurements, not train-step or end-to-end speed results. The plan remains `execution.authorized=false` and has no training or test metrics.

> Variable-token grouping pilot (2026-09-07): on Traffic/C862/H96 with the valid lag-masked relation contract, `group_size=16` raised best validation loss by `1.96%` and four-epoch wall time by `4.23%` relative to `group_size=1`. A six-point resource scan (`1/4/8/16/32/64`) found all grouped variants slower in median forward latency (`+5.71%--+8.19%`) while reducing peak increment by about `34%`. The implementation therefore does not meet the efficiency objective and is stopped as a mainline candidate; readout: [`mamba_variable_group_pilot_readout_20260907.md`](mamba_variable_group_pilot_readout_20260907.md).

> Variable-token representation-fusion pilot (2026-09-07): a paired ETTh1/Traffic H96, seed-2021, four-epoch run with the valid lag-masked relation contract found no gain from `mamba_variable_repr_fusion`: best validation loss changed by `+0.216%` (ETTh1) and `+0.096%` (Traffic). The adapter adds `2,146` parameters and stable post-warmup training time rose about `2.78%/2.01%`; prediction-only latency changed in opposite directions (`+4.29%` ETTh1, `-2.75%` Traffic). This is a stopped exploratory branch, not evidence for a speed or representation-learning claim; readout: [`mamba_variable_repr_fusion_pilot_readout_20260907.md`](mamba_variable_repr_fusion_pilot_readout_20260907.md).

> Traffic native baseline pilot (2026-09-07): DLinear/NLinear/PatchTST/TimesNet/iTransformer/Mamba-1/Mamba-2 all completed Traffic/C862/H96, seed-2021, four-epoch validation-only runs. Best validation MSE was `0.431119` (TimesNet), `0.438770` (PatchTST), `0.442530` (iTransformer), `0.494128` (Mamba-2), and `0.497026` (Mamba-1); Mamba-2 was `12.6%` above PatchTST and `14.6%` above TimesNet. Native baseline normalization differs from AutoCon/MeanStd, so this is coverage/resource evidence only; full readout: [`baseline_traffic_pilot_readout_20260907.md`](baseline_traffic_pilot_readout_20260907.md).

> Native baseline expansion (2026-09-07): the same seven native baseline entries now complete ETTh1, ETTh2, ETTm1, ETTm2 and Traffic at H96, seed-2021, four epochs, with independent prediction-only resource records. The lowest short-budget validation MSE was DLinear on ETTh1 (`0.653467`), NLinear on ETTh2 (`0.208738`) and ETTm2 (`0.126379`), iTransformer on ETTm1 (`0.395855`), and TimesNet on Traffic (`0.431119`). These are dataset-local, native-normalization pilot values and cannot be pooled with AutoCon/MeanStd candidate results; cross-dataset summary: [`baseline_native_pilot_summary_20260907.md`](baseline_native_pilot_summary_20260907.md).

> MeanStd parity pilot (2026-09-07): ETTh1/Traffic、H96、seed 2021/2022、4 epoch 下，PatchTST、Mamba-1、Mamba-2、plain MambaACF 和 MambaACFAutoCon 共 20 个 validation-only 作业全部完成。Traffic 的 plain MambaACF/AutoCon mean validation MSE 为 `0.405427/0.405349`，比 PatchTST `0.443120` 低约 `8.51%/8.52%`；ETTh1 则分别比 PatchTST 高约 `7.22%/5.95%`。AutoCon relation 相对 plain MambaACF 的均值变化仅 ETTh1 `-1.18%`、Traffic `-0.019%`，所以关系目标虽然 telemetry 非退化，但尚无独立平均收益证据。Traffic 资源 median 约比 PatchTST 低 `17.7%`，ETTh1 却高约 `159%`；速度与精度都具有数据集依赖性。完整读数见 [`mamba_normalization_parity_pilot_readout_20260907.md`](mamba_normalization_parity_pilot_readout_20260907.md)。

> Variable-token parity pilot (2026-09-07): 在同一 MeanStd/AutoCon 合同下增加 `mamba_variable_mixer=true`，Traffic 两个 seed 相对 shared AutoCon 分别改善 `6.52%/5.19%`，ETTh1 为 `5.54%` 改善和 `0.51%` 退化；均值变化为 ETTh1 `-2.57%`、Traffic `-5.86%`。但 prediction-only median latency 相对 shared AutoCon 增加 ETTh1 `72.19%`、Traffic `71.27%`，trainable 参数增加 `54.21%/14.63%`。因此它是待多 seed 复验的精度候选，不是当前效率方案；协议见 [`mamba_variable_parity_pilot_20260907.md`](mamba_variable_parity_pilot_20260907.md)。

> Variable-token parity third-seed replication (2026-09-07): 补充 seed 2023 后，Traffic 三个 seed 全部改善，shared/variable mean validation MSE 为 `0.406053/0.383383`，平均 delta `-5.583%`；ETTh1 为 `0.726339/0.709336`，平均 delta `-2.299%`，但仅 `2/3` seed 改善。三 seed 平均训练 wall 增加 ETTh1 `8.84%`、Traffic `13.97%`；prediction-only median latency 增加 `69.59%/68.59%`，吞吐下降约 `41%`。因此 variable-token 是 Traffic 的精度候选，但不是效率方案，也仍不是确认性论文结果；完整读数见 [`mamba_variable_parity_replication_readout_20260907.md`](mamba_variable_parity_replication_readout_20260907.md)。

> MeanStd core baseline pilot (2026-09-07): 为修复 native baseline 的归一化不一致，DLinear/NLinear 新增默认关闭的 opt-in `MeanStd` 路径，并在 ETTh1、ETTm1、Electricity、Traffic 上与 PatchTST、TimesNet、iTransformer、Mamba-1、Mamba-2 以同一 H96/三 seed/四 epoch 合同完成 `84/84` validation-only 作业。最佳 mean validation MSE 分别为 ETTh1 `PatchTST=0.690529`、ETTm1 `iTransformer=0.395980`、Electricity `TimesNet=0.175228`、Traffic `TimesNet=0.438170`；Mamba-1/2 相对这些数据集最强非 Mamba 参考分别为 `+4.73%/+6.41%`、`+5.98%/+4.00%`、`+3.88%/+2.25%`、`+12.62%/+12.27%`。Mamba-1/2 只在 Electricity 上逐 seed 稳定优于 PatchTST，Traffic 上三 seed 均落后 PatchTST/TimesNet/iTransformer。28 个 prediction-only 资源记录显示 Mamba-1/2 比 TimesNet 快，但比 iTransformer 和线性基线慢；这批结果不支持“普遍不掉点且更快”。完整读数见 [`mamba_meanstd_baseline_pilot_readout_20260907.md`](../../artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/summary/meanstd_baseline_pilot_readout.md)，计划与重建工具见 [`mamba_meanstd_baseline_pilot_20260907.json`](mamba_meanstd_baseline_pilot_20260907.json) 和 [`summarize_meanstd_baseline_pilot.py`](../../scripts/summarize_meanstd_baseline_pilot.py)。

> Classical baseline extension pilot (2026-09-07): 为覆盖额外数据域并修复仓库内经典模型入口，Weather/Exchange-Rate 上新增 Autoformer、FEDformer、ETSformer、Transformer、Crossformer，并与 DLinear/NLinear/PatchTST/TimesNet/iTransformer/Mamba-1/Mamba-2 采用同一 MeanStd/H96/三 seed/四 epoch 合同完成 `72/72` validation-only 作业。Weather 的短预算最优均值为 Crossformer `0.420389`、TimesNet `0.459811`、Mamba-2 `0.476681`、Mamba-1 `0.487984`；Exchange-Rate 最优为 PatchTST `0.260995`、NLinear `0.261362`、ETSformer `0.271541`，Mamba-1/2 为 `0.310850/0.340135`。因此 Mamba 在 Weather 优于多数小模型但仍落后 Crossformer/TimesNet，在 Exchange-Rate 落后线性/patch/ETSformer；没有跨数据集的普适精度结论。24 条 prediction-only 资源记录显示 Mamba-1 约 `0.43--0.45 ms`、Mamba-2 约 `0.71--0.72 ms`，仍慢于线性和 iTransformer；完整读数见 [`classical_extension_pilot_readout.md`](../../artifacts/exploratory/mamba_classical_baseline_extension_pilot_20260907/summary/classical_extension_pilot_readout.md)，计划与汇总器见 [`mamba_classical_baseline_extension_pilot_20260907.json`](mamba_classical_baseline_extension_pilot_20260907.json) 和 [`summarize_classical_extension_pilot.py`](../../scripts/summarize_classical_extension_pilot.py)。
> Classical core extension pilot (2026-09-07): 在 ETTh1、ETTm1、Electricity、Traffic 上补跑 Autoformer/FEDformer/ETSformer/Transformer/Crossformer，共 `60/60` 个 H96、MeanStd、三 seed、四 epoch validation-only 作业和 20 条 prediction-only 资源记录。短预算均值中 Crossformer 在 ETTh1 (`0.697765`) 与 Electricity (`0.203918`) 最低，但在 Traffic (`1.106847`) 明显落后 Autoformer/FEDformer；ETTm1 的 ETSformer (`0.412577`) 最低。该阶段只补齐经典基线覆盖，未包含 Mamba 臂，不能与已有 84-job MeanStd pilot 直接合并排名；完整读数见 [`classical_extension_pilot_readout.md`](../../artifacts/exploratory/mamba_classical_baseline_core_extension_pilot_20260907/summary/classical_extension_pilot_readout.md)。

> 计数说明：上面的旧段落保留了 2026-09-07 早先审计快照；当前工作树完整回归已更新为顶部所示的 `633 passed, 43 warnings`。

当前工作树完整回归为 `633 passed, 43 warnings`；本轮新增了 CMamba-lite/GDD 工程 pilot，但没有启动任何确认性或测试集选择实验。Mamba-1/Mamba-2 CUDA smoke 以及 MambaACF/MambaACF1 forecast-view 的真实前向/反向 smoke 均通过（当前环境 `mamba_ssm=2.3.1`），这些只证明形状、有限值和梯度路径，不构成精度证据；可复跑入口为 [`smoke_mamba_acf_forecast_relation.py`](../../scripts/smoke_mamba_acf_forecast_relation.py)。GDD 两轮 ETTh1/Traffic 单 seed validation pilot 的完整复核见 [`mamba_cmamba_lite_autocon_pilot_readout_20260904.md`](mamba_cmamba_lite_autocon_pilot_readout_20260904.md)：`.01` 初始化后仿射分支确实获得非零参数，但 GDD 相对 control 仍为 ETTh1 `+1.304%`、Traffic `+1.672%`；把 GDD 放到 variable LayerNorm 后仍为 `+1.218%`、`+1.956%`。Traffic 参数增量约 `17.1%`，因此按停止规则停止该分支，不作 CMamba 复现、精度或速度主张。当前树上新的 memory/temperature 配对验证结果见 [`mamba_acf_memory_temperature_current_readout_20260904.md`](mamba_acf_memory_temperature_current_readout_20260904.md)：memory on 相对 off 的 pooled best-val 下降约 `0.342%`（5/6），temperature 1 相对 4 仅下降 `0.053%`（4/6），两者均为 skip-test 的方向性证据；temperature 调参已停止。随后完成的逐通道 top-2 legal lag-bank screen 见 [`mamba_acf_channel_lag_bank_current_readout_20260904.md`](mamba_acf_channel_lag_bank_current_readout_20260904.md)：pooled best-val 仅下降 `0.052%`（4/6），ETTh1 `-0.060%`、Traffic `-0.037%`，因此也停止作为独立精度主线；它只保留为低成本诊断/部署选项。上述三批都是当前树、H96、skip-test 的 exploratory 证据，不能与旧 LastVal/test screen 合并。逐通道稀疏 lag-bank 候选的最新人工复核稿见 [`mamba_acf_channel_lag_bank_contract_20260903.md`](mamba_acf_channel_lag_bank_contract_20260903.md)，其当前 v4 资源工件只证明结构可实例化，不包含预测读数；strict draft 被全窗口归一化阻断。保留 strict 草案的同时，已登记一个明确限定为 forecast-memory indexing 因果性的修订计划 [`mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json`](mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json)，12 个 job 已 validate-only 通过但仍未授权、未训练；该 validation-only 计划关闭 `stream_metrics`，主指标读取 checkpoint `loss_logs.json:val_loss`。对应只读分析器 [`analyze_mamba_acf_channel_lag_bank_screen.py`](../../scripts/analyze_mamba_acf_channel_lag_bank_screen.py) 在工件不完整时返回 `NOT_RUN`，不会生成伪读数。新增逐步 reliability screen 已登记 12 个未授权 job，计划和人工复核稿见 [`mamba_acf_step_reliability_screen_plan_20260903.json`](mamba_acf_step_reliability_screen_plan_20260903.json) 与 [`mamba_acf_step_reliability_screen_readout_20260903.md`](mamba_acf_step_reliability_screen_readout_20260903.md)，当前没有该候选的精度结果。

新增 `mamba_acf_relation_representation=forecast` 已完成输出等价、形状、预测头梯度、单通道拒绝和入口一致性合同；它仅在训练时把最终归一化预测送入 global relation loss。该 loss 对预测步做均值池化，所以候选是窗口级跨通道关系的直接路由消融，不是逐步 forecast dynamics 目标。计划 [`mamba_acf_forecast_relation_view_screen_plan_20260903.json`](mamba_acf_forecast_relation_view_screen_plan_20260903.json) 已登记 `encoder/forecast × lambda=0/.03` 的 24 个 validation-only job，`execution.authorized=false`；没有 checkpoint、精度、训练时间或速度结果。零权重臂仍计算 relation path，只是其梯度系数为零，不能当作关闭辅助路径的吞吐对照。

本轮新增 `mamba_acf_channel_order=similarity`，但它不是主线精度结论。该模式在 train-only ACF sidecar 的完整 selected-lag signed profile 上做确定性 cosine-seriation，使用 `O(C^2 K)` 点积和 `O(CK)` 临时内存，运行时只保留 `order/inverse` 非持久 buffer；sidecar 不直接保存排列，回放由 sidecar、配置和同一 helper 确定性重建。前向不构造通道相似度矩阵。`input` 仍是默认，`strength` 与 `similarity` 都只作为独立 variable-token ablation；它们不能被解释为跨通道因果图或 cross-correlation。回放复用了 [`utils/acf_channel_order.py`](../../utils/acf_channel_order.py) 的同一实现，并已加入模型/CLI/计划校验/资源签名/回归测试。当前没有该模式的训练精度结果，也没有把它加入确认性主矩阵。

本轮新增了显式 `MeanStd` 窗口归一化模式。它使用每个输入窗口的 population variance，并计算 `sqrt(var + 1e-5)`，与 `PatchTST.forecast()` 的实际数值合同逐元素一致；`ReVIN` 保持为 sample-standard-deviation 合同，`Mean` 和 `LastVal` 也作为独立的窗口合同保留。该模式已接入 `MambaACF`、Mamba-1/2 AutoCon 共享 head、历史 `AutoConNet`、运行参数帮助和严格表示因果性 validator。它不增加参数或 Mamba 扫描，只用于消除 Traffic 高维比较中的归一化混杂。对应的 36-job H96 配对计划见 [`mamba_normalization_parity_screen_plan_20260903.json`](mamba_normalization_parity_screen_plan_20260903.json)；计划已 `validate_only` 通过，但 `execution.authorized=false`，没有新的训练指标。`MeanStd` 仍读取完整窗口均值/方差，所以不会解除严格逐位置 representation-causality 的限制。

新增的 `mamba_variable_repr_fusion` 是一个独立的 AutoCon 候选：它复用已有 variable-token Mamba 的 `[B,C,D]` 状态，做 softmax channel-weighted summary，经小型 `D -> D -> D` adapter 后注入 AutoCon temporal representation；forward 与 representation 请求共用同一次 token realization，且该可选模块的构造在局部 RNG scope 中进行，避免 off/on 消融改变共享骨干的初始化。该开关默认关闭，只允许 MambaACF AutoCon + variable mixer，且窗口级摘要明确标记为非严格因果。原 12-job ETTh1/Traffic H96 off/on 计划见 [`mamba_variable_repr_fusion_screen_plan_20260903.json`](mamba_variable_repr_fusion_screen_plan_20260903.json)，已 `validate_only` 通过但仍未授权、未启动；本轮只做了独立的单 seed 四 epoch pilot。pilot 的两个数据集均无 validation 改善，prediction-only 资源也没有一致方向的变化，因此该候选已停止扩展；完整读数见 [`mamba_variable_repr_fusion_pilot_readout_20260907.md`](mamba_variable_repr_fusion_pilot_readout_20260907.md)。

逐步 reliability 的实现/诊断契约见 [`mamba_acf_stepwise_reliability_contract_20260903.md`](mamba_acf_stepwise_reliability_contract_20260903.md)。当前新增的 `acf_effective_memory_to_forecast_rms` 只记录 ACF residual 相对加回前预测的强度，不改变预测数值；lag-bank 与 direct-anchor 的支持域也在 profile 中明确分开。

Traffic `C=862` 的当前源码 lag-bank 资源复核已保存到 [`mamba_acf_channel_lag_bank_resource_current_20260903`](../../artifacts/exploratory/mamba_acf_channel_lag_bank_resource_current_20260903/)。在 `B=8,T=96,H=96`、单尺度、50 warm-up/200 timed prediction-only forwards 下，off/on median 为 `0.780848/0.780432 ms`，peak allocated 为 `42.444824/42.645020 MB`，trainable parameter 数完全相同。这个单次 device-resident smoke 的中位数差约 `-0.05%`，应解读为没有证明速度变化；它不包含训练、反向或精度。

本轮还补齐了 Mamba checkpoint 的恢复身份合同。新运行会持久化版本化的 runtime contract；旧 config 会按模型的**有效**分支默认值归一化（例如关闭的 gate/rank、`local_width=null` 与 `d_model` 的等价关系，以及 plain 模型中 dormant 的 representation-fusion 请求），避免把不影响推理的历史字段误判为架构差异。评估在模型构造前拒绝错误的 Mamba/非 Mamba 模型身份、缺失/损坏的 `config.json`，以及启用静态或 train-only ACF 分支时缺失、无效或哈希不匹配的 `acf_prior.npz`；因此不会先重新拟合 train prior 再加载 checkpoint。该改动提升的是可复核性和失败闭合，不提供任何精度增益证据。

本轮又修复了一个仅影响回放 provenance 的边界：`is_training=0` 且所有 ACF forecast 分支关闭时，plain `MambaACF` 不再无谓读取 train split；即使 Namespace 残留或显式提供 sidecar 路径，也不会加载或重新生成 prior。对于 AutoCon 实验已在构造阶段挂载的 sidecar，外层回放现在只做 hash/state metadata 绑定，不重复调用 loader，并保留 `acf_values/lags/shape`。已有 checkpoint 的 sidecar/resolution 元数据会保留在新的 `config.json` 中，且多次 `--itr` 不会把上一副本的派生 lag 字段带入下一副本。该修复由回归测试覆盖，不改变预测函数或历史指标。

旧的 standalone replay JSON 中，部分 ACF-off 记录仍会出现 `sidecar_load_message=true`；那是旧 loader 行为的历史事实，不是当前 ACF-off 合同。新的回放合同要求 live ACF 分支加载 sidecar，ACF-off 分支既不加载也不拟合 train-only prior，并继续核对复制工件和规范化配置身份的 hash/字段。

本轮用当前 `run.py` 做了三个单 job identity smoke：[`ACF-off`](../../artifacts/exploratory/mamba_acf_replay_identity_smoke_off_current_20260903/replay_report.md)、[`plain required`](../../artifacts/exploratory/mamba_acf_replay_identity_smoke_on_current_20260903/replay_report.md) 与 [`AutoCon required`](../../artifacts/exploratory/mamba_acf_autocon_replay_identity_smoke_current_20260903/replay_report.md) 均为 `contract=pass`、`metric=match`；前者明确无 sidecar load/fit，后两者明确加载 sidecar 且无重新拟合。它们只验证评估合同，不增加训练样本或精度证据。旧目录仍保留为上一版源码哈希的历史 smoke。

这里的 runtime contract 是**保守的配置/构造合同**，不是试图找出所有情况下的最小 forecast identity。因 raw ACF lag 列表会影响模型初始化时的分支宽度/校准适配，sidecar replay 仍要求沿用源实验的 raw lag 列表与 `lag_mode`；sidecar 的 effective lags 另行校验并在加载后覆盖。手工回放应从持久化 `config.json` 复制这些参数，不应只凭 sidecar 的 effective lags 拼接一条新命令。

本轮对高维确认阶段做了只读 selection 回放。使用计划注册的 selection 文件和 sidecar 时，严格 validator 拒绝了绑定：`selected_hyperparameters.provenance.json` 记录的 `attester_sha256=7421ce64…` 与当前 attester 不同（当前为 `e24bf7e9…`），且它与 selection 自身的 source snapshot（`57743f10…`）也不一致；该哈希实际来自另一阶段的 source snapshot。因而高维 `_v3` 表格仍只能标记为后验、非 publication-ready 诊断，不能把这次回放写成“确认性验证通过”。没有改写历史 sidecar，也没有为绕过 gate 而重新生成确认结果。

本轮同时把布尔值、分数、非有限值和越界值的拒绝规则同步到 ACF utility、CLI、训练计划和 phase validator；Mamba1/Mamba2 的 `S/M/MS` 输出边界也有回归测试。相关检查均为结构/契约验证，不产生新的预测精度结论。

随后发现并修复了一个恢复路径漏洞：评估时若 CLI 的 channel-lag-bank 开关/top-k 与历史 config 不同，原流程可能在同一 checkpoint 上静默改变 forecast-memory 拓扑。现在该组合在模型构造前严格拒绝；`set_compact_acf_prior` 也会拒绝 compact/full ACF 表的 lag/value 不一致。针对性回归覆盖了开关双向不一致、稳定 tie-break 和不足合法 lag 的失败路径。

Mamba1/Mamba2（以及对应 AutoCon 版本）共享同一个预测 head：`enc_in=c_out` 时输出合同正常。现有正式计划只使用 `features=S` 或 `features=M`；`features=MS` 的旧训练器合同是模型先返回全部输入通道，再由 trainer 取最后一个通道。除此之外的 `enc_in != c_out` 组合目前没有通用输出投影，可能产生静默广播或在 head 相加时失败，因此不能把这类配置当作已支持的 Mamba1/2 结果。若未来需要一般 `c_out`，应单独设计 restore 后的 target-channel projection，并以 shape contract tests 和独立配对实验验证。

## 2026-09-03 严格表示因果性修正

当前 `MambaACF` 的四种已支持归一化都读取完整输入窗口：`LastVal` 取窗口最后一点，`Mean` 取全窗口均值，`ReVIN`/`MeanStd` 取全窗口均值和标准差。因而即使未来只改变最后一个观测，前缀的 normalized state 也会改变；这是实际数据依赖，不只是命名问题。`resource_profile()` 现在把它列为 `window_full_normalization`，统一的运行时、训练计划、资源和结果 validator 在 `mamba_require_strict_representation_causality=true` 时返回 `window_full_normalization_forbidden`。

这不改变现有预测的数值路径，也不否认 forecast-time 输入窗口本身是可用信息；它只禁止把当前 AutoCon representation 写成“逐位置严格因果”。因此现有逐通道 lag-bank strict draft 被标为 blocked，不能再宣称 `validate_only` 通过。要恢复该更强声明，必须先实现 prefix-causal normalization，并以新增的前缀不变性、训练和配对重训复核其影响。

## 2026-09-03 新候选：窗口条件低秩通道状态混合器

当前树新增 `LowRankGlobalChannelStateMixer`，接在 `MambaACF`/`MambaACF1` 的共享状态路径。它复用 MambaACF 已计算的 `[B,D]` 窗口摘要，经过 `D -> r -> 2D` 低秩投影生成共享 hidden-space scale/shift，并以 ReZero 残差调制 `[B,T,D]` 状态，因此 forecast head 与 AutoCon 表示分支消费同一调制状态。该路径新增矩阵计算为 `O(Dr+rD)`，不引入 dense `C x C` 参数；跨变量信息通过现有共享 embedding 与 scale-gate 摘要间接进入适配器。这里的实现是受 CMamba 思路启发的高效全局适配器，不是外部 CMamba 的复刻。

该开关默认关闭，`rank` 必须为正；实际秩由 `min(C-1, summary_width-1)` 封顶（单通道时保留 1 维瓶颈），并与 `mamba_channel_independent=true` 不兼容。ReZero 标量初始为零，保证启用候选在初始化时与参考模型严格嵌套，但也意味着适配器权重在第一个更新前没有梯度，需要在训练日志中检查残差标量是否打开。适配器使用由完整观测窗口产生的共享摘要并把它广播到所有时间位置，所以即使 `mamba_causal_embedding=true`，也不能声明严格的逐位置 representation causality；资源 profile 已把这一限制写入 `representation_grid_causality_limits`。

`resource_profile()` 现在额外记录该分支的 summary-to-rank、rank-to-hidden-affine、逐时间状态调制 MAC 估计和激活元素数；summary 直接复用现有 scale-gate 的 `[B,D]` 窗口摘要，不重复扫描原始窗口。这些是排除 bias、归约、非线性、kernel launch 与数据搬运的结构代理，不是 GPU 实测 FLOP、显存或端到端速度。候选的唯一已登记精度合同见 [`mamba_data_dependent_channel_mixer_screen_readout_20260903.md`](mamba_data_dependent_channel_mixer_screen_readout_20260903.md) 与 [`mamba_data_dependent_channel_mixer_screen_plan_20260903.json`](mamba_data_dependent_channel_mixer_screen_plan_20260903.json)：60 个作业、ETTh1/Traffic、H96、3 seeds，当前 `execution.authorized=false`、没有 checkpoint 或指标。该候选尚未证明提高精度、降低速度成本或具备论文证据。

在同一 RTX 5090 D 上完成了当前源码的四配置 B=8 isolation resource spot check：四条记录均为 plain `MambaACF`、`AutoCon=false`、ACF memory off；ETTh1/Traffic 均使用 `B=8`。ETTh1 的 on/off median 比为 `1.0671x`、p95 比为 `1.0724x`；Traffic 分别为 `1.0811x` 和 `1.0545x`，rank-4 均增加 453 个参数。尽管 batch 已统一为 8，这些显式 CLI defaults 仍不包含预注册候选的 ACF memory/AutoCon 分支及其它架构开关；它们只说明固定 synthetic shape 的隔离结构成本，不能外推到候选或端到端速度。当前记录见 [`mamba_data_dependent_channel_mixer_resource_current_b8_20260903`](../../artifacts/exploratory/mamba_data_dependent_channel_mixer_resource_current_b8_20260903/README.md)。旧的 spot check 与四次重复 smoke 绑定不同源码/协议，仍保留为历史工件，不与当前记录混合；所有这些记录都不得解读为速度或精度收益。

## 2026-09-03 新候选：逐预测步 reliability

审计发现，静态 ACF memory 的默认 reliability 是跨 horizon 聚合后再广播到所有预测步；当长 lag 只在前段 horizon 合法时，强峰可能被不可用槽位稀释。新计划 [`mamba_acf_step_reliability_screen_plan_20260903.json`](mamba_acf_step_reliability_screen_plan_20260903.json) 只切换 `mamba_acf_step_reliability`，保留同一 `MambaACFAutoCon`、raw train-only prior、global signed relation、数据划分、seed 和预算，共 12 个 H96 job，已 validate-only 通过但未授权、未训练。人工复核稿见 [`mamba_acf_step_reliability_screen_readout_20260903.md`](mamba_acf_step_reliability_screen_readout_20260903.md)。

该候选的判定不能只看 MSE：每个 epoch 还必须记录 `acf_runtime_gate_mean`、`acf_seasonal_rms`、`acf_effective_memory_rms` 和 `acf_effective_memory_to_forecast_rms`，以确认 ACF residual 在预测尺度上没有被初始化 gate 或主干完全压没。若没有 paired retraining 和这些 telemetry，不能把已有同 checkpoint 反事实敏感性写成 treatment effect。

## 2026-09-03 已训练 checkpoint 的 ACF 反事实依赖（只读）

对同一 `MambaACFAutoCon` checkpoint 仅切换 `use_acf_memory`、保持参数、测试窗口、sidecar 和评估状态不变的 source-snapshot 重放中，6 个 seed 的 `memory_off` MSE 均高于 `memory_on`：ETTh1 均值 `0.393167 -> 0.457892`（相对上升 `16.463%`），Traffic 均值 `0.564118 -> 0.624881`（相对上升 `10.771%`）。gate 与 effective-memory RMS 也均非零。这支持一个窄结论：这些已训练函数的预测输出实际依赖 ACF residual 分支；它不估计 memory-on/off 独立重训的平均处理效应，也不能替代配对重训或确认性统计。完整逐 seed 表、公式、哈希和限制见 [`mamba_acf_counterfactual_audit_20260903.md`](mamba_acf_counterfactual_audit_20260903.md)。

## 2026-09-03 Traffic 高维逐通道稳健性（后验描述性）

在高维 `_v3` 工件的 862 个 Traffic 通道上，Mamba2 相对 PatchTST 的 scaled-MSE 均值差为 H96 `+29.205%`（8/862 通道更好）、H336 `+20.611%`（8/862）、H720 `+19.833%`（7/862）。5% 对称 trimmed channel macro 与逐通道 leave-one-out 仍保持 Mamba2 劣势；Mamba2AutoCon 也没有改变方向。这说明差距是广泛的 channel-wise 现象，而非少数异常通道，优先暴露共享 `d_model` 压缩/跨变量表达瓶颈。通道和窗口不是独立推断单位，且该 `_v3` selection 标记为 `posthoc_verified`、`publication_ready=false`，因此这里只作结构诊断，不作显著性或泛化主张。后续 ACF 候选应优先评估逐通道/分组 lag bank 与 channel-preserving 结构，并保持同预算对照。

## 2026-09-02 新候选：可靠性插值

当前树新增默认关闭的 `mamba_acf_reliability_interpolation`。它从静态、train-only、raw ACF prior 已编译的两个统计量派生：逐预测步/通道的 legal-lag strength 与跨 horizon 的 aggregate strength。候选计算
`exp(tanh(alpha) * clip(log((step+eps)/(aggregate+eps)), -log 2, log 2))`，其中 `alpha` 是唯一新增的零初始化标量；它只调整 seasonal memory 的绝对幅度，不改变 signed lag、relation loss、hidden gate 或 prior schema。`alpha=0` 时与关闭路径逐项嵌套，且不读取当前窗口或目标。

该开关被限制为 `mamba_use_acf_memory=true`、`mamba_dynamic_acf=false`、`AutoCon_acf_source=raw`、`mamba_acf_step_reliability=false`，并已接入模型 profile、CLI、训练计划校验、阶段结果校验和资源 manifest signature。资源 signature schema 因新增字段发生变化，旧 signature 不应与新资源记录直接合并。

资源方面，最小结构 smoke 显示该分支只增加 1 个标量参数；canonical ETTh1 device-resident GPU 前向（B=32、50/200 次）的一次 on/off 配对为 `1.399312 -> 1.506592 ms`（median，约 `+7.67%`），峰值分配记录基本不变。该读数不包含数据加载、反向传播或训练，且尚无该候选的精度结果。

配对计划见 [`mamba_acf_reliability_interpolation_screen_plan_20260902.json`](mamba_acf_reliability_interpolation_screen_plan_20260902.json)：ETTh1/traffic、H96、每臂 3 seeds、off/on 共 12 jobs。计划已 `validate_only` 通过，但 `execution.authorized=false`、尚未训练，因此目前没有该候选的精度结论或论文证据。

## 2026-09-02 核心审计更新

当前 AutoCon relation 训练目标仍为 `MSE + effective_lambda * relation_loss`。`AutoCon_lambda=0.03` 只是历史候选值，不是可迁移默认：共享 trunk synthetic probe 的加权 auxiliary/MSE 梯度比约 `0.071--0.124`，两者 cosine 约 `-0.020--0.059`；因此不引入盲目的 GradNorm 或 uncertainty weighting，先登记 validation-only 配对 lambda 校准。双重 `F.normalize` 在当前探针中 loss 差不超过 `3.6e-7`、梯度相对差不超过 `2.4e-7`，暂不改变语义。

归因边界同样需要保留：默认 `mamba_acf_relation_representation=encoder` 且 `mamba_acf_repr_fusion=false` 时，relation loss 只更新表示分支，不能写成直接训练 forecast ACF memory/gate；`forecast` 是独立、受限的训练期消融，它直接作用于最终归一化 forecast 的窗口级跨通道摘要，必须与 encoder 模式在相同 lambda/null、数据、seed 和预算下配对比较。当前 signed hinge 约占 auxiliary 数值 `0.04%`，不能把 `91%` 左右的 sign accuracy 解释为强符号监督。普通 token embedding 是 circular convolution；严格全模型表示因果性除 `mamba_causal_embedding=true` 外，还要求单位 token stride、单尺度、非双向扫描以及关闭窗口级 gate/非 raw 居中变换。`wo_freq` 原地修改时间 marks 的问题已修复，回归覆盖输入不变和 stride-view 复用。

`resource_profile()` 现把 ACF descriptor 的因果 lag lookup 与完整表示网格的严格因果性分开报告。即使 raw ACF context 本身只查过去，`mamba_token_stride>1`、多尺度池化后插值、双向扫描、普通 circular embedding、窗口级 channel/scale gate、当前全窗口 `LastVal/Mean/ReVIN/MeanStd` 归一化，或 residual/trend 的居中变换都会阻止逐位置表示因果性主张；确认性计划固定 raw、causal embedding、单位 stride 和单尺度时仍须另外使用 prefix-causal normalization，当前实现尚未满足。

详细审计见 [`mamba_acf_loss_scale_audit_20260902.md`](mamba_acf_loss_scale_audit_20260902.md)，对应的未授权校准计划见 [`mamba_acf_lambda_calibration_screen_plan_20260902.json`](mamba_acf_lambda_calibration_screen_plan_20260902.json)。该段记录的 `427 passed, 36 warnings` 是 2026-09-02 审计快照；当前工作树回归数见本页顶部。没有因本轮审计启动新的精度训练。relation loss 的 global/local 幅值 CE 与已加权 signed-hinge contribution 已按 epoch 写入 `loss_logs.json`，旧日志缺失时不回填。

> **此前更新（2026-09-02）：** 本页是当前状态入口；旧版 [`mamba_acf_core_status_20260901.md`](mamba_acf_core_status_20260901.md) 仅作历史快照。新增 current-tree 变量-token公平 screen：30/30 作业、五个 phase validator 全部通过、30/30 sidecar 独立回放通过；同一 `MambaACFVariableAutoCon` 的 `lambda=.03` 相对 `lambda=0` 的 pooled MSE 为 `-0.979%`（5/6），但仍只有三个 seed、H96，属于探索性证据。与独立 PatchTST 均值作描述性比较时，variable calibrated 在 ETTh1 高 `1.188%`、Traffic 低 `0.829%`；不能声称普遍不劣或更快。新增 memory-gate mode 只有资源测量，direct-anchor 只有结构/梯度合同，二者均未做精度 screen；NLinear 缺口已登记但未训练。新增单尺度/多尺度严格配对计划已登记并通过 validate-only，但 `12` 个 job 仍未授权、未训练。direct-anchor 的独立资源 smoke 见 [`mamba_acf_anchor_copy_resource_smoke_20260902.md`](mamba_acf_anchor_copy_resource_smoke_20260902.md)，当前源码指纹下的 50/500 测量为 ETTh1 约 `+4.04%`、Traffic 高维约 `+4.35%`，不是端到端速度证据。

确认性 accuracy claim 也已显式收窄：当前 `statistical_gate` 只支持 combined MambaACF 对 PatchTST/Mamba2 的 primary non-inferiority；DLinear、NLinear、TimesNet、iTransformer、Mamba1 是必须报告但仅描述性的 secondary baseline。不能把未来完整 baseline phase 的结果自动解释成“对全部基线不劣”。

## 先给结论

Mamba 作为时间编码器做预测是有条件可行的，不是普适方案。当前冻结表中的 `Mamba2` 是无 relation loss 的 Mamba-2 编码器加 AutoConNet-compatible forecasting head，并非裸 Mamba forecaster；它在 Electricity 上接近或略优于 PatchTST，但在 Traffic 的三个预测长度都明显落后。Traffic 的失败主要指向跨变量关系和局部片段建模不足，而不是单纯把 Mamba 层数加深就能解决。

当前仓库最准确的定位是：**以 Mamba 为时间编码核心、用 signed-ACF 提供训练集统计先验、用 AutoCon 做独立表示学习的多变量直接预测器**。它不是“只有 Mamba 算子”的纯模型，因为还包含输入嵌入、线性残差头和可选跨变量分支。

需要先区分代码版本：当前 plain Mamba 的预测路径已通过回归；历史 `corrected` 表中的 AutoCon 行仍来自旧的辅助损失/表示契约。它们可以作为可复核的历史诊断，但不能与当前代码重新训练出的 AutoCon 结果混合，也不能据此宣称新的 AutoCon 增益。

当前共享结构还有一个必须正面处理的容量瓶颈：Traffic 的 `C=862` 个变量在每个时间点先压到 `d_model=32`（或 64），Mamba 扫描的是 `[B,T,d]` 的共享状态，最后再用共享时间头恢复全部通道。它参数高效，但可能把异质通道的独立动态压进低维状态；PatchTST 保留每通道的 patch 状态。因此 Traffic 的差距不能简单归因于“Mamba 不适合”，需要 channel-independent、variable-token 和相同参数预算的线性残差对照。

当前单尺度候选的资源优势已经被严格匹配的 synthetic forward 测量确认，但精度尚未确认：`mamba_scales=[1]` 相对 `[1,2]` 的 ETTh1/Traffic median latency 分别下降 `38.69%/39.19%`。对应的 12-job 配对训练计划见 [`mamba_acf_single_vs_multiscale_screen_plan_20260902.json`](mamba_acf_single_vs_multiscale_screen_plan_20260902.json)，状态为 `execution.authorized=false`。另外，runner 现在拒绝 `mamba_token_stride>1` 与 MambaACF AutoCon 的 `local/both` relation scope 组合，因为插值表示的时间坐标与 ACF lag 坐标不一致；正式 relation 实验固定 `stride=1`，或只允许 global scope。

## AutoCon 代码到底做了什么

历史 `Mamba1AutoCon` / `Mamba2AutoCon` 只把 AutoConNet 的特征提取器换成 Mamba。预测端仍保留 AutoConNet 的多尺度趋势分解、长度映射和线性 seasonal head；历史 `Mamba1/Mamba2` 则在同一模块中关闭 relation loss，仍使用该预测头。因此历史结果回答的是“换编码器后的兼容预测头是否变化”，不能当作纯 Mamba forecasting 结果。

历史 AutoCon 目标按 ACF 为每个时间位置找 top-1 正样本；旧实现会丢弃 ACF 符号，并将超出表范围的距离 clamp 到最后一个值。它适合复现旧实验，但不适合直接解释为严格的周期关系。

新 `MambaACF` 将预测状态和对比表示分开：Mamba 状态走 direct multi-horizon head，表示经过独立 projection head 再进入 AutoCon/ACFRelationLoss。ACFRelationLoss 保留符号、对无效 lag 做 mask，`ACFLagPairBatchSampler` 让 batch 实际包含目标 lag 的窗口对。

当前关系损失为 `ACFRelationLoss.objective_version=3`。它把候选集定义为“在 ACF 表范围内、非 identity、且距离不少于 `mamba_acf_relation_min_lag` 的所有 pair”，而不是只保留 selected lags。`mamba_acf_relation_target_mode` 有三种取值：`hard_selected`（传入 selected lags 时为 v2 兼容的 selected-positive；未传入时保留 legacy all-candidate fallback）、`soft_multi_peak`（语义/机制控制：所有 `abs(ACF)>soft_min_abs` 的候选为正目标，并对 selected lags 乘 `1+selected_lag_boost`）和 `soft_full`（所有非零 ACF 候选）。`relation_scope=global` 只比较窗口级表示；`local`/`both` 还计算窗口内部位置，并按 active positions 归一化。v3 screen 尚未证明 soft 目标带来稳定精度收益，因此不能把它当作默认精度方案。新日志需包含 objective version、target mode、global candidate/positive/contrastive-active-anchor/degenerate-positive-anchor/high-ACF-excluded 及 local active-position 诊断；旧 v2 日志不能与 v3 合并。

这里有四个不能省略的实现边界。第一，默认 `DataEmbedding` 使用带
`padding_mode='circular'` 的 `TokenEmbedding`，所以只有 ACF memory 的索引是严格因果的，
整个预测网络不能宣称严格左因果。第二，ACF prior 在完整 train 序列上做 moving-average
分解，而推理时在每个窗口独立分解并复制端点；窗口边界的 residual 与 prior 拟合分布可能不一致，
长 lag/短窗口必须做敏感性分析。第三，当前 head 和 memory signal 依赖固定 `seq_len`，变长输入
不是已承诺的模型合同。第四，历史 pilot 存在 plain `weight_decay=0`、AutoCon `1e-5` 以及
`[1]`/`[1,2]` scales 的差异；这些结果不能用于干净归因 ACF，正式比较必须显式固定 optimizer、
scales、宽度和训练预算。

## ACF 的高效用法

ACF 应被当作“结构先验和可靠性权重”，而不是未来值生成器：

1. 只在 train split 上用 FFT、通道分块估计 signed ACF，保存 compact peaks。
2. 将 ACF 关系用于表示学习：强度由绝对 cosine 拟合，符号由独立 hinge 项约束。
3. 对 causal memory 只读取历史中确实存在的 `L - 1 + h - lag`，返回 validity mask；不可用的长 lag 不循环复制。
4. plain、AutoCon 和 memory 必须使用同一 `AutoCon_acf_source`。探索性运行默认使用 `residual`；严格确认方案固定使用 `raw`，因为当前整段 train prior 与逐窗口 residual/trend memory 在窗口边界并不数值一致。memory 采用与 prior 相同的信号族，但这不等于 residual/trend 已满足严格数值等价。
5. 对 forecast memory 使用 `train_peaks_available`：只从 train-only 峰值候选中选择，并按合法 forecast-step 比例加权；它是 horizon-specific，解析出的 lag 必须写入元数据。

高 ACF 不等于高预测收益。Traffic 的固定周期相关很强，但简单 seasonal copy 仍可能失败，说明未来状态、跨变量耦合和相位变化比“相关系数大”更关键。

## 已有数字和证据等级

### 冻结高维后验复核结果（5 seeds；`_v3` 工件已完成）

| 数据集 / H | PatchTST | Mamba2 | Mamba2AutoCon |
| --- | ---: | ---: | ---: |
| Electricity / 96 | 0.184448 | 0.183130 | 0.182321 |
| Electricity / 336 | 0.207953 | 0.211477 | 0.210460 |
| Electricity / 720 | 0.252428 | 0.241182 | 0.241146 |
| Traffic / 96 | 0.501767 | 0.648306 | 0.649773 |
| Traffic / 336 | 0.516942 | 0.623489 | 0.623169 |
| Traffic / 720 | 0.550696 | 0.659914 | 0.658317 |

这些结果支持“带兼容预测头的 Mamba 编码器具有数据集依赖性”，不支持“纯 Mamba 普遍优于强基线”。高维 `_v3` 工件已完整覆盖 `270/270` 个 confirmatory 坐标（Electricity/Traffic × 3 horizons × 9 arms × 5 seeds），其 [`phase_validation.json`](../../artifacts/paper_final/corrected/highdim_confirmatory_M_v3/phase_validation.json) 记录 `valid=true`、`manifest_unique_jobs=270`、`status_counts.ok=270`。原始 manifest 有 280 条尝试记录，其中 10 条早期记录因文件名过长失败，随后同一 job ID 均有成功记录；这里的 270/270 指最终去重后的有效作业，而不是 280 次都成功。

这批结果的 selection 绑定模式是 `prelaunch_replay_of_posthoc_attestation`，而 selection sidecar 明确为 `attestation_mode=posthoc_verified`、`publication_ready=false`（见 [`provenance.json`](../../artifacts/paper_final/corrected/highdim_confirmatory_M_v3/provenance.json) 与 [`selected_hyperparameters.provenance.json`](../../artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.provenance.json)）。因此它是完整、可复核的后验 replay/冻结读数，不能称为原始 launch-time 预注册的正式确认性证据。

### 新 MambaACF Traffic pilot（单 seed、3 epochs）

| 结构 | MSE | 参数量 | 证据用途 |
| --- | ---: | ---: | --- |
| plain shared temporal | 0.600947 | 150,700 | 实现 pilot |
| MambaACF1 plain (Mamba-1) | 0.608056 | 152,768 | 同预算 Mamba-1 对照 |
| gate + causal embedding | 0.602418 | 未单独登记 | 混合消融，不作因果结论 |
| variable-token branch | 0.587315 | 169,242 | 值得进入正式消融 |
| patch-token branch（旧 all-candidate pilot） | 0.611913 | 197,402 | 历史配置；不能解释为有效 selected-lag contrastive learning |
| causal ACF memory | 0.596415 | 150,700 | 相对 plain 约 -0.75%，仍需多 seed |
| causal embedding + memory | 0.615312 | 150,700 | 严格左因果 pilot 反而变差，需单独公平消融 |
| ACF relation loss | 0.606501 | 150,700 | 当前 pilot 未显示收益 |
| variable-token + relation | 0.590902 | 169,242 | 略差于 variable-only，需调权重/覆盖率 |

此表的参数列来自当时的 pilot 快照；其中 plain/shared 配置包含后来识别出的
未使用参数。它们只用于记录当时的运行合同，不用于当前容量公平性比较。

当前应优先阅读 2026-09-07 的修正 patch-token pilot：在
`hard_selected + mamba_acf_relation_lag_mask=true` 下，ETTh1/Traffic 的单 seed
best validation loss 相对 off 分别下降 `2.79%/3.24%`，但 synthetic median
forward latency 增加 `40.42%--67.67%`（四数据集资源预检）。因此它是待复验的精度候选，
不是速度候选；完整读数见 [`mamba_patch_token_contrastive_pilot_readout_20260907.md`](mamba_patch_token_contrastive_pilot_readout_20260907.md)
与 [`mamba_patch_token_contrastive_screen_preflight_readout_20260907.md`](mamba_patch_token_contrastive_screen_preflight_readout_20260907.md)。

### 可用性加权 memory 的 3-seed 复核（仍属 pilot）

Traffic/H96、`d_model=32`、3 epochs 的 selector 自动从训练 split 得到 `[96,120,72]`。plain 对照使用完全相同的 seed 和训练合同：

| seed | selector | fixed memory | plain | selector vs fixed | fixed vs plain |
|---:|---:|---:|---:|---:|---:|
| 2021 | 0.586256 | 0.596419 | 0.600948 | 1.704% | 0.754% |
| 2022 | 0.590390 | 0.607324 | 0.611094 | 2.788% | 0.617% |
| 2023 | 0.587594 | 0.599909 | 0.605031 | 2.053% | 0.847% |
| mean | 0.588080 | 0.601217 | 0.605691 | 2.182% | 0.739% |

固定 memory 相对 plain 平均改善 `0.739%`；在同一 memory 框架内，selector 额外改善 `2.182%`。三个 seed 都同方向，但 `n=3`，按正式协议的双侧 exact Wilcoxon 为 `p=0.25`，而且只有一个数据集、一个 horizon、短训练预算；这只能说明“训练峰值与合法可用性联合选择”值得进入正式实验，不能称为稳定的普适收益。原始工件在 [`artifacts/mamba_acf_selector_replication_20260831/`](../../artifacts/mamba_acf_selector_replication_20260831/)。

逐步 reliability gate 的独立重跑（同一 Traffic/H96、`d_model=32`、3 epochs、seed 2021）为 MSE `0.585388`、MAE `0.396185`，训练和测试工件在 [`artifacts/fair_pilot_20260831/rerun_memory_stepwise/`](../../artifacts/fair_pilot_20260831/rerun_memory_stepwise/)。该结果与 selector pilot 同方向，但不是独立多 seed 证据，不能把它与上表的三 seed 均值当作同一实验组。

这些数字不能与冻结基线直接作优劣结论：训练 epoch、宽度和选择协议不同。它们只说明 variable-token
和可用性加权 causal memory 是当前值得做正式验证的候选；v3 soft relation 目前也只有机制 pilot，
尚不能进入主张。

### 低秩预测头筛选（H96，探索性）

[`mamba_acf_low_rank_head_screen_readout_20260901.md`](./mamba_acf_low_rank_head_screen_readout_20260901.md) 的两臂均使用同一 variable-token、raw train-only sidecar、memory、global hard relation 和 3-seed/dataset 矩阵，`12/12` 作业及 validator 通过。rank-16 只因子化 temporal head 与 linear residual，不改变 AutoCon 表示分支：MSE 相对 full head 为 `+1.095%`（`0/6` 更低），MAE 为 `+0.199%`；head 参数从 `18,624` 降至 `6,336`，但没有精度保持证据。full/rank16 实际总训练轮数为 `41/39`，elapsed 的 `-0.977%` 差异受 early stopping 影响，不能解释为纯速度收益。该结果来自 immutable provenance snapshot、没有 standalone replay，rank-16 暂不保留。

### 关系覆盖与 active-anchor 筛选（H96，探索性）

[`mamba_acf_relation_coverage_screen_readout_20260901.md`](./mamba_acf_relation_coverage_screen_readout_20260901.md) 完成 `18/18` 作业和三份 validator。balanced least-exposed-lag sampler 将 effective lags `[96,120]` 的 exposure span 从 random 的 `0.124--0.951%` 压到 `0--0.135%`，但 balanced 相对 random 的 MSE/MAE 为 `+0.239%`/`+0.249%`；active-anchor normalization 相对 random 的 MSE/MAE 为 `+0.013%`/`+0.018%`。因此当前两 lag 的采样频率与全 batch 分母都不是主要精度瓶颈。reference/active_norm 的 ETTh1 轮数为 `7/5/5`，balanced 为 `5/6/6`，Traffic 各为 8；elapsed 仍只作短预算描述。下一步应优先处理 high-ACF false negatives、跨变量 lag coherence 和 horizon-specific memory 校准。

关系损失的 coverage 不能只写成一个数字。旧 v2 审计中，未使用 selected-lag mask 时约 `15.9%`
是 dense global 候选覆盖、active-anchor 约 `99.9%`；使用 selected-lag mask 后，候选覆盖约
`3.1%`、active-anchor 约 `92.6%`。这些数字不等于 v3 soft target coverage，因为 v3 的候选分母仍
包含未选 lag。local scope 只在窗口内部位置 `0..T-1` 查距离；当 `T=96`、selected lags 为
`[96,120,72]` 时，lag 96/120 对 local loss 没有有效 pair，只有 lag 72 能贡献部分 pair。
正式日志应分别报告 local/global active pair、每个 lag 的计数和覆盖率；本轮 coverage screen 已将这些 telemetry 写入并由 analyzer 交叉核对 effective/selected lags。`ACFLagPairBatchSampler` 的 balanced 模式仍是可行不相交配对约束下的贪心策略，不是严格 stratified sampler。随后 current-tree raw target screen 已直接测试扩大 strong-ACF support，结果为 Traffic 0/3 seed 改善，因此该方向按注册规则停止，不再继续做阈值网格。

### 当前代码的 raw relation-target screen（H96，探索性）

[`mamba_acf_relation_target_screen_readout_20260901.md`](./mamba_acf_relation_target_screen_readout_20260901.md) 完成 `12/12` 个 current-tree 作业，两个 phase validator 均通过；同一批 checkpoint/sidecar 的 `is_training=0` replay 也为 `12/12`。两臂均使用 raw train-only sidecar、variable-token MambaACF AutoCon、global relation、关闭 forecast memory、lambda `0.03`、full-rank head 和 8 epoch；唯一注册差异是 hard selected-peak target 与 `abs(ACF)>0.5` 的 soft05 target。

| 对比（soft05 - hard） | ETTh1 | Traffic | pooled |
|---|---:|---:|---:|
| MSE 相对变化 | `-0.045%`（2/3） | `+0.279%`（0/3） | `+0.146%`（2/6） |
| MAE 相对变化 | `-0.033%` | `+0.380%` | `+0.160%` |

soft05 将 positive/anchor 从 `0.923` 提高到 `1.470`，并把独立 `abs(ACF)>0.1` high-ACF 条件排除从 `78.47%` 降到 `66.62%`，但 Traffic 三个 seed 全部退化。该 screen 触发预注册停止规则，且未满足后续扩展门槛；它只说明“扩大 strong-ACF target support”在当前 raw/global/短窗口协议下没有预测收益，不能外推为所有 soft target、memory 或 ACF 方法的否定。

### 早期 v3 soft relation 机制 pilot（Traffic/H96，单 seed、3 epochs）

在同一 `MambaACFAutoCon` 骨干、seed `2021`、global scope、未启用 forecast memory 的配对 pilot
中，hard 行使用 `hard_selected`（v2 兼容，`lag_mask=true`），soft 行使用
`soft_multi_peak`、`soft_min_abs=0.1`、`selected_lag_boost=1.0`（`lag_mask=false`）。测试流指标和
最后一个 epoch 的 global 诊断为：

| 目标 | scaled MSE | scaled MAE | positive/anchor | candidate fraction | high-ACF excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard_selected (v3 compatibility) | 0.617183 | 0.400212 | 0.944 | 0.157 | 0.681 |
| soft_multi_peak (v3 semantic control) | **0.609619** | **0.397436** | 4.287 | 0.157 | 0.000 |

soft 相对 hard 的 scaled MSE 在这次早期 pilot 中下降约 `1.23%`，但后续 30-run screen 的三 seed
结果没有复现稳定收益；这组早期数字仅说明目标语义可运行。两行都没有启用 ACF memory，不能与
memory-only 或历史 AutoCon 结果比较，也不能据此宣称 soft 目标提高精度。完整工件见
[`mamba_acf_soft_relation_pilot_20260831`](../../artifacts/mamba_acf_soft_relation_pilot_20260831/)。

### v3 30-run screen（五阶段均值，仍属探索性）

v3 screen 共 `30` 个成功运行（五阶段 × ETTh1/Traffic × 三个 seed）。每个数据集-阶段单元
只有 `n=3`，没有多重比较校正；`ALL` 合并行不是六次独立重复。均值 ± seed-level SD：

| 阶段 | ETTh1 MSE | Traffic MSE | ETTh1 MAE | Traffic MAE |
| --- | ---: | ---: | ---: | ---: |
| plain_screen | 0.389664 ± 0.002807 | 0.570445 ± 0.003733 | 0.411648 ± 0.002041 | 0.389375 ± 0.002401 |
| available_memory_screen | 0.389005 ± 0.001678 | 0.555458 ± 0.005670 | 0.408272 ± 0.002361 | 0.384809 ± 0.003573 |
| hard_relation_nonoverlap_screen | 0.391957 ± 0.002591 | 0.565820 ± 0.008408 | 0.413486 ± 0.003919 | 0.384731 ± 0.004063 |
| soft_relation_nonoverlap_screen | 0.392103 ± 0.001567 | 0.568679 ± 0.004372 | 0.412781 ± 0.002968 | 0.388632 ± 0.008201 |
| available_memory_soft_relation_nonoverlap_screen | 0.395972 ± 0.005431 | 0.551952 ± 0.004095 | 0.414361 ± 0.006776 | 0.381977 ± 0.007107 |

soft 相对 hard 的 MSE 均值变化为 ETTh1 `+0.000146`（约 `+0.04%`）、Traffic `+0.002859`
（约 `+0.51%`），方向不一致；因此三 seed screen 不支持 soft 的稳定精度增益。memory+soft
相对 available memory 在 ETTh1 为 `+1.79%`，Traffic 为 `-0.003506`、约 `-0.63%`；Traffic
方向有利但仍不足以形成确认性结论。详细 seed-level 数值和检验见
[`mamba_acf_v3_screen_readout.md`](./mamba_acf_v3_screen_readout.md)。
这里的“可复核”是指 immutable snapshot 下的指标与 manifest 校验，不是当前 runner 的
sidecar-backed standalone replay：该旧 screen 早于 `acf_prior.npz` 契约，memory/relation 臂没有
持久化 prior，必须在原始 train split 和匹配 snapshot 下 historical re-run；plain 臂不依赖该 sidecar。

### Provenance snapshot 的 sidecar 持久化合同复核（仍属探索性）

旧 screen 之后，按当时的 provenance snapshot 和 `raw` ACF 运行了同一 18-job 坐标。三个 phase 均为
`valid=true`，18/18 作业均保存完整 `acf_prior.npz` 和 SHA 绑定；详细 readout 见
[`mamba_acf_sidecar_replay_readout.md`](./mamba_acf_sidecar_replay_readout.md)。新 paired MSE
读数为：memory 相对 variable plain 在 ETTh1 `-0.567%`、Traffic `-1.618%`；relation 在
memory 之上分别再降 `-0.529%`、`-0.550%`。每个单元只有 3 个 seed、H96，relation global
coverage 约 3%，这些数字只证明该 provenance snapshot 下的方向性连续性以及 sidecar 持久化/合同完整性，不能作为确认性或
跨 screen 的 pooled 结论。需要特别注明：这些指标是在训练进程中拟合 train-only prior 后直接产生的，
不是后来独立加载 sidecar 的评估指标；当前工作树还修正了 lag canonicalization、评估元数据写回和 validator 完整性，
因此不能把这批数字称为 current-tree 重跑。随后已在隔离输出根对这 18 个历史 checkpoint+sidecar 完成
`is_training=0` standalone replay，`18/18` 成功且 MSE/MAE 与原记录在数值容差内一致；这只验证当前评估加载路径，
不等于 current-tree 重训或确认性结果，详见 [`mamba_acf_sidecar_replay_audit.md`](./mamba_acf_sidecar_replay_audit.md)。

### Variable-token + memory + relation screen（仍属探索性）

独立的三阶段 screen 已完成 `18/18` 个作业，三个 phase 的 manifest、provenance、数据哈希和
`phase_validation.json` 均通过；详细逐 seed 表在
[`mamba_acf_variable_memory_relation_screen_readout.md`](./mamba_acf_variable_memory_relation_screen_readout.md)。
ETTh1/Traffic 各只有三个 seed、H96、8 epoch，因此不能替代确认性实验：

需要额外保留一个复现限制：这批 screen 早于当前 ACF sidecar 持久化契约，18 个 job 目录均没有
`acf_prior.npz`，配置也没有 artifact/hash 绑定。6 个 variable_plain 作业关闭 memory/relation，
不依赖 sidecar；6 个 variable_memory 与 6 个 variable_memory_relation 作业在旧代码中从 train split
即时拟合 prior，不能仅凭 checkpoint+config 由当前 runner 独立 reload/replay。论文主表前必须用当前
代码重新生成 sidecar 和 resolved manifest。

| 对比 | ETTh1 MSE 变化 | Traffic MSE 变化 | 方向一致性 |
| --- | ---: | ---: | ---: |
| variable + available memory - variable plain | `-0.045%` | `-2.282%` | 1/3、3/3 |
| variable + memory + hard relation - variable + memory | `-0.802%` | `-0.655%` | 3/3、3/3 |

关系阶段的 effective lags 为 `[96, 120]`，使用 objective v3，global candidate、active-anchor 和
positive coverage 均为非零。第二个对比只给出“在 variable+memory 之上加入 relation”的增量，不能
把它解释成 variable、memory 和 relation 各自的独立因果贡献；当前也没有据此启动 884-job 确认矩阵。
另外，这批变量 token 作业固定了 `mamba_causal_embedding=false` 和 `mamba_acf_channel_order=input`，
没有随机置换对照；因此尚不能声称严格因果嵌入或通道顺序不变性。后续应至少加入固定
strength-order 与可重复的随机 permutation 鲁棒性检查。

### 当前代码的变量-token fair screen（raw sidecar，严格 null 对照）

为修正旧 screen 中“memory 分支到 relation 分支同时改变 executable model 与 loss”的混淆，重新注册了
[`mamba_acf_variable_token_fair_screen_plan.json`](./mamba_acf_variable_token_fair_screen_plan.json)。五个 phase 共 30 个作业：variable memory off/on、同一
`MambaACFVariableAutoCon` 的 `lambda=0` null、`lambda=.03` relation，以及 rank-2 horizon calibration。
每个 phase 都是 `6/6`、`valid=true`；严格 readout 见
[`mamba_acf_variable_token_fair_screen_readout_20260901.md`](./mamba_acf_variable_token_fair_screen_readout_20260901.md)，30 个 checkpoint/sidecar 的隔离 replay 为
`30/30` contract pass、`30/30` metric match、drift `0`。

| 配对对比 | n | MSE 相对变化 | 改善点数 | 读法 |
|---|---:|---:|---:|---|
| memory on - off | 6 | `-1.168%` | 6/6 | train-only raw ACF memory 的方向性证据 |
| relation `lambda=.03` - 同架构 null `lambda=0` | 6 | `-0.979%` | 5/6 | 当前唯一可解释的 relation-loss 对照；描述性 `p=0.0848` |
| rank-2 calibration - relation | 6 | `-0.025%` | 3/6 | 基本中性，暂不增加复杂度 |

relation 按数据集为 ETTh1 `2/3`、Traffic `3/3` 个 seed 改善；该 screen 的停止规则只允许进入单独审阅的冻结复现，不能直接进入论文主表。

与独立 42-job H96 fair-baseline screen 的均值（不同 provenance，只作描述性对照）如下：

该 fair screen 的 `relation` 臂使用 `MambaACFAutoCon`，而 `memory` 臂使用 `MambaACF`；因此 relation-memory 行同时改变可执行分支和目标函数，只能作为混杂的 staged comparison。纯 relation 效应应使用下文同一 `MambaACFVariableAutoCon` 架构的 `lambda=.03` 对 `lambda=0` null 对照。

| 数据集/H96 | PatchTST（独立 fair screen） | variable relation | variable calibrated | calibrated - PatchTST |
|---|---:|---:|---:|---:|
| ETTh1 | 0.380848 | 0.384496 | 0.385372 | `+1.188%` |
| Traffic | 0.555538 | 0.552047 | 0.550932 | `-0.829%` |

因此当前 variable candidate 尚没有达到“所有数据集精度不低于强基线”的要求；Traffic/H96 的描述性优势也不能外推到其它 horizon。资源测量也不是统一优势：calibrated candidate 的 synthetic median latency 为 ETTh1 `1.557 ms`、Traffic `1.669 ms`，参数量为 `58,953`/`169,248`；独立 fair-baseline 资源工件中同一 10/30 协议的 PatchTST 为 ETTh1 `0.283 ms`、Traffic `4.057 ms`，所以 candidate 只在 Traffic 该输入形状更快，不能外推为普遍速度优势。完整资源工件见 [`mamba_acf_variable_token_resources_20260901`](../../artifacts/exploratory/mamba_acf_variable_token_resources_20260901/) 和 [`mamba_acf_fair_baseline_resources_20260901`](../../artifacts/exploratory/mamba_acf_fair_baseline_resources_20260901/)。

当前源码下另有一个值得优先验证的速度候选：固定 MambaACF AutoCon、`causal_embedding=true`、普通 channel gate、raw 静态 ACF memory 和 prediction-only 调用，将时间尺度从 `[1,2]` 减为 `[1]` 后，严格匹配配置的 `100/300` synthetic 协议中位延迟由 ETTh1 `1.40067` 降至 `0.85872 ms`（`-38.69%`），Traffic `1.41931` 降至 `0.86306 ms`（`-39.19%`）。这只证明去掉第二条 Mamba 扫描能降低 forward 负载；单尺度的预测精度、AutoCon relation 的收益和端到端训练时间尚未测量，不能修改默认架构或写成论文速度结论。原始记录见 [`mamba_acf_resource_baseline_current_20260902.md`](mamba_acf_resource_baseline_current_20260902.md)。

新 `mamba_acf_memory_gate_mode` 提供 `hidden` 与 `variable_token` 两种门控来源。两者参数量相同，但 token 模式的输出从 `[B,H,1]` 变为 `[B,H,C]`；在不读数据的 deterministic synthetic prior forward 中，ETTh1 median 增加约 `0.16%`，Traffic 增加约 `4.68%`，Traffic gate MAC 为 `2,648,064` 对 hidden 的 `3,072`。本次 allocator 峰值恰好相同，不代表结构激活内存相同；后者按通道数放大。门控同时改变 variable-token 分支的梯度路径，尚无准确率结果，不能作为当前主模型或速度优化结论。详见 [`mamba_acf_memory_gate_mode_resources_20260902.md`](./mamba_acf_memory_gate_mode_resources_20260902.md)。

Direct-anchor 分支是另一个默认关闭的结构实验：它只增加一个 ReZero 标量，并使用严格因果的 indexed gather 读取 signed ACF lag。当前没有训练精度读数；当前源码指纹下的 50/500 独立资源 smoke 在 ETTh1（C=7、B=2）观察到 `1.2759 -> 1.3274 ms`（约 `+4.04%`），Traffic（C=862、B=1）观察到 `1.3049 -> 1.3616 ms`（约 `+4.35%`）。static buffer 分别增加 `0.0160 MB` 和 `1.9730 MB`，所以相同的相对 peak increment 不能被误读为零内存成本。它仍不是速度优势证据，不能据此宣称普遍更快。完整协议、源码 SHA-256 和原始 JSON/CSV 见 [`mamba_acf_anchor_copy_resource_smoke_20260902.md`](mamba_acf_anchor_copy_resource_smoke_20260902.md)。

注意：shared fair screen 的 calibrated 数字仍为 ETTh1 `0.393062`、Traffic `0.562918`（相对 PatchTST `+3.207%/+1.329%`），不要与本节 variable-token 数字混用；两项 screen 的 provenance 和可执行分支不同。

### 当前代码 v3 的公平结构复跑（仍属探索性）

为检查最近的静态 memory cache 和 ACF channel gate，重新固定 Traffic/H96、`seq_len=96`、`d_model=32`、Mamba-2（`d_state=16`、`d_conv=4`、`expand=2`）、3 epochs、seed 2021，并使用同一批次的 PatchTST 对照。结果仍只有单 seed，不能替代正式多 seed 复核：

| 结构 | MSE | MAE | 读法 |
| --- | ---: | ---: | --- |
| PatchTST 对照 | 0.565537 | 0.358836 | 同批 exploratory baseline |
| MambaACF，gate-only | 0.566947 | 0.384248 | 比 PatchTST 高 0.25%，gate 单独没有收益 |
| MambaACF，memory-only | **0.549255** | 0.378875 | 本批 MSE 最低；只是一 seed/3 epochs |
| MambaACF，gate + memory | 0.552341 | 0.377871 | 比 memory-only 高约 0.56%，gate 未带来额外 MSE 收益 |

原始日志分别位于 [`mambaacf_acf_gate_v3`](../../artifacts/extension_pilot_20260831/traffic_h96/mambaacf_acf_gate_v3/)、[`mambaacf_memory_only_v3`](../../artifacts/extension_pilot_20260831/traffic_h96/mambaacf_memory_only_v3/) 和 [`mambaacf_acf_gate_memory_v3`](../../artifacts/extension_pilot_20260831/traffic_h96/mambaacf_acf_gate_memory_v3/)。这组结果与前面的三 seed selector 表属于不同代码/协议批次，不能拼成一个统计结论。

### 历史资源测量（合成、device-resident；旧计时协议）

Traffic、`C=862`、H96、batch 8、RTX 5090 D 的前向资源记录如下。Mamba-2/PatchTST 是较早的 5 warmup + 20 timed forwards，MambaACF 是 20 + 100；因此这里只作工程量级读数，正式论文应统一计时协议并补充端到端训练时间。

| 结构 | 中位延迟 (ms) | 序列/s | 峰值增量 (MB) | 可训练参数 |
| --- | ---: | ---: | ---: | ---: |
| Mamba-2 plain | 0.7208 | 11010.8 | 23.10 | 167,562 |
| PatchTST | 1.0537 | 7574.1 | 133.90 | 46,080 |
| MambaACF memory-only | 1.3871 | 5765.8 | 25.86 | 160,793 |
| MambaACF gate + memory | 1.6655 | 4796.9 | 25.86 | 183,327 |

静态 cache 的当前明确收益是把 memory 分支的临时高维张量压低到约 25.9 MB 峰值增量；它没有让 MambaACF 比 plain Mamba-2 更快。ACF gate 只增加 113 个 ACF adapter 参数，但增加了可见延迟，当前不应作为主方案宣传。完整 JSON 工件在 [`resource_extension`](../../artifacts/resource_extension/)。

优化器现在只接收 `requires_grad=True` 的 active 参数；冻结模块仍保留 state-dict key，
但不会进入 Adam 参数组。上表的参数列是代码修正前生成的历史快照。当前代码会冻结未进入 forward 的
`channel_pool`，并在关闭 ACF memory 时冻结 `acf_gate`；因此同一配置的 active
trainable count 分别约为 plain/no-memory `145,387`、memory-only `160,760`、
gate+memory `183,294`（总参数量不变）。旧 JSON 不应被静默改写；正式论文前应
用当前代码和统一脚本重跑资源记录，并同时报告 total 与 active trainable。

还需要重跑优化器公平性：历史扩展配置有 plain 路径省略 `weight_decay`（实际为 `0`）而
AutoCon 路径默认 `1e-5` 的情况，且部分任务使用不同 scales。当前 optimizer 已过滤冻结参数，
但这些旧 pilot 仍不能作为严格的正则化、容量或 ACF 归因对照。

#### 当前源码统一协议复测（Traffic/C862/H96/B8）

为消除旧快照计时轮数不一致的问题，使用当前源码、固定 synthetic device-resident 输入、同一
`50` 次 warm-up 和 `200` 次计时重新测量。结果见
[`resource-v3-current-traffic-report.md`](./resource-v3-current-traffic-report.md) 和
[`model_resources.json`](../../artifacts/resource_v3_current/traffic_c862_h96_b8_w50_t200/model_resources.json)：

| 结构 | 中位延迟 (ms) | 合成序列/s | 峰值增量 (MB) |
| --- | ---: | ---: | ---: |
| PatchTST | 0.885824 | 9025.3 | 95.188 |
| Mamba-2 | 0.702608 | 11369.7 | 23.614 |
| MambaACF plain | 0.684640 | 11674.2 | 15.760 |
| MambaACF memory | 0.808272 | 9888.7 | 25.861 |

在这个单一硬件/形状下，plain MambaACF 的平均延迟比 PatchTST 低约 `22.69%`，memory
分支比 plain 高约 `18.06%`。这仍然是一次资源负载测量：不包含数据读取、传输、反向传播、优化器、
验证或 checkpoint，因此不能写成端到端训练更快，也不构成预测精度证据。

### ACF 效用审计（描述性）

在 train-only signed ACF 审计中，Electricity 的 lag 96 中位 ACF 为 `0.8939`，Traffic 的 lag 168 中位 ACF 为 `0.8164`；但 `seq_len=96` 时，lag 168 在 horizon 96 只有 24/96 步可直接读取，lag 336 在 horizon 96 完全不可用。Traffic test 的 seasonal-last-fallback scaled MSE 仍为约 `1.50`（lag 168、H96），而 last baseline 为约 `1.80`，说明 ACF 可以提供弱结构信号，却不能替代状态/跨变量建模。完整计算定义和 SHA256 记录见 [`acf-forecast-utility.md`](./acf-forecast-utility.md) 及 `artifacts/exploratory/acf_forecast_utility/`。

## 论文边界

### MambaTS v3 variable permutation（2026-09-22）

新增的 train-only variable permutation 已完成 36/36 个 validation-only 作业
（ETTh1、ETTm1、Electricity、Traffic，H96，3 seeds）。与同结构关闭
AutoCon 的 permutation null 相比，验证 MSE 变化分别为 ETTh1 `+0.220%`、
ETTm1 `-0.182%`、Electricity `-1.111%`、Traffic `-0.037%`，没有跨数据集
一致收益；因此不进入当前主线。完整记录见
[`mamba_ts_v3_variable_permutation_screen_20260922.md`](./mamba_ts_v3_variable_permutation_screen_20260922.md)。

资源 smoke（Traffic/C862/H96/B8，prediction+representation，20/80，RTX
5090 D）为中位 `0.6933 ms`、峰值增量 `153.5 MB`、参数 `136,700`。排列只在
训练阶段启用，推理阶段不会形成独立的速度优势，原始记录见
[`model_resources.json`](../../artifacts/exploratory/mamba_ts_v3_variable_permutation_resource_20260922/model_resources.json)。

### MambaTS v3 direct forecast ACF consistency（2026-09-22）

新实现的 parameter-free 训练期 loss 以 train-only signed ACF 选择可见 lag，
再直接约束预测序列的时间相关性；推理没有新增参数或路径。干净的 24-job
validation-only 重跑（四数据集、H96、三 seeds）结果为 ETTh1 `-0.031%`、
ETTm1 `-0.082%`、Electricity `+0.051%`、Traffic `+0.029%`（负数表示 MSE
改善）。高维数据没有一致收益，故停止该全局正则方向，不读取测试集、不做
lambda 调参，也不进入主表。完整 provenance、失败并发批次边界和逐 seed
结果见 [`mamba_ts_v3_forecast_acf_consistency_screen_20260922.md`](./mamba_ts_v3_forecast_acf_consistency_screen_20260922.md)。

### MambaTS v4 representation-selected ACF memory（2026-09-22）

当前最值得继续验证的候选是 v4 static train-only ACF memory 加 AutoCon
representation selector。selector 只生成共享的 horizon-wise correction，
不引入 channel-dependent 参数；四数据集、H96、三 seed 的 paired validation
变化为 ETTh1 `-0.475%`、ETTm1 `-0.943%`、Electricity `-1.376%`、Traffic
`-0.572%`，均为 selector 相对 static-memory control。Traffic/C862/H96/B8
prediction-only 资源中，selector `0.7822 ms`，PatchTST `0.8873 ms`，
TimesNet `2.8072 ms`；但 selector 比 static v4 `0.7168 ms` 慢约 `9.1%`，
且 ETTh1 仍落后 PatchTST。因此它只能进入冻结的多 horizon、同源码强基线
确认阶段，尚不能支持论文主张。完整记录见
[`mamba_ts_v4_acf_memory_selector_screen_20260922.md`](./mamba_ts_v4_acf_memory_selector_screen_20260922.md)。

多 horizon 复验没有通过稳定性门槛：H336/H720 的 24 个 paired validation
坐标中，ETTm1/Electricity 多数改善，但 ETTh1/H720 为 `+0.367%`、Traffic/H336
为 `+0.039%`。因此没有启动该 selector 的同快照强基线或测试评估。随后针对
失败坐标的逐变量置信度 pilot 在 Traffic/H336、Traffic/H720、ETTh1/H720
分别为 `-0.309%`、`-0.258%`、`-0.271%`，但 ETTh1/H336 为 `+0.659%`；这只
支持“全局门控可能忽略变量异质性”的机制假设，不支持稳定精度提升。详见
[`mamba_ts_v4_acf_memory_selector_multihorizon_20260922.md`](./mamba_ts_v4_acf_memory_selector_multihorizon_20260922.md)
和 [`mamba_ts_v4_acf_memory_channel_selector_pilot_20260922.md`](./mamba_ts_v4_acf_memory_channel_selector_pilot_20260922.md)。

### MambaTS v3 transfer baseline pilot（2026-09-22）

新增 Weather、Exchange-Rate、ILI 的 54-job、三 seed、MeanStd、validation-only
统一基线 pilot，覆盖 DLinear、PatchTST、TimesNet、iTransformer、Mamba-2 和
MambaTSv3AutoCon。候选在 Weather/H96 的均值为 `0.445116`，相对 PatchTST、
TimesNet、Mamba-2 分别为 `-10.010%`、`-3.174%`、`-6.615%`，均为 `3/3` seed
改善；但 Exchange/H96 相对 PatchTST 为 `+5.873%`（`0/3`），ILI/H24 相对
PatchTST 为 `+6.789%`（`0/3`）。Weather/C21/H96/B32 的 prediction-only
中位延迟为 `0.267 ms`，PatchTST 为 `0.283 ms`，但候选参数量和显存略高。
因此它只支持 Weather 分支的长 horizon/null-control 复验，不支持通用基线不劣或
“AutoCon 稳定增益”的论文主张。详见
[`mambatsv3_transfer_baseline_pilot_20260922.md`](./mambatsv3_transfer_baseline_pilot_20260922.md)。

现在可以写一份诚实的架构审计/实验设计报告，但还不能据此写成“ACF 稳定提高精度”或“Mamba 更快”的论文。当前筛选已经排除 balanced sampler、active-anchor normalization 和 rank-16 head 作为主要改进方向；进入主表前仍必须完成多数据集、多 seed、固定 validation gate 的 plain/variable/channel-independent/relation/memory 消融，并报告 paired delta、校正后的显著性、端到端训练时间、峰值显存和通道级分布。当前统一资源复测显示 plain MambaACF `0.684640 ms`、MambaACF memory `0.808272 ms`、PatchTST `0.885824 ms`（B=8、Traffic 862 通道、H=96、50/200 次、合成 device-resident forward）；这些数字不能等同于端到端训练速度，也不能替代容量公平性分析。

下一轮正式计划草案在 [`mamba_acf_extension_plan.json`](./mamba_acf_extension_plan.json)，外部结构取证在 [`external-mamba-ts-architecture-review.md`](./external-mamba-ts-architecture-review.md)，代码/历史结果审计在 [`autocon_mamba_evidence_memo.md`](./autocon_mamba_evidence_memo.md)。

### MambaTS v3 full-cover d32 H96 four-PEMS confirmation（2026-09-26）

在 d24 H96、六 token d16 H96 先后未通过后，固定了 full-cover
`patch_len=24/stride=24, d_model=32` 候选，并完成 PEMS03/04/07/08、H96、三
seed 的 active/null 共 24 个 validation-only 作业。两阶段均通过严格的 12/12
manifest 与 provenance 校验。active 相对 lambda-zero null 的均值变化为：
PEMS03 `+0.004%`、PEMS04 `+0.038%`、PEMS07 `+0.012%`、PEMS08 `-0.937%`；
因此 AutoCon 的方向并不一致，不能声称带来稳定精度收益。相对历史 H96 强基线，
PEMS03/07/08 分别好 `2.140%/0.992%/2.663%`，PEMS04 差 `2.213%`；停止规则
未触发，但这只是继续做同源基线和资源核验的许可，不是论文主结果。完整记录见
[`pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_readout_20260926.md`](./pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_readout_20260926.md)。

同形状 PEMS07/H96 的 forward-only 资源核验也已完成：d32 候选中位延迟
`0.625584 ms`、总参数 `69,707`；TimeMixer（同为 `d_model=32,d_ff=64`）为
`0.718736 ms`、`183,171` 参数。候选约快 `12.96%`、参数少 `61.96%`，但峰值增量显存
约多 `38.3 MB`。这支持“低延迟/低参数”的工程方向，不等同于端到端训练速度，也不能
覆盖 PEMS04 的精度缺口。资源记录见上述 d32 readout 中的两个 JSON 工件。

### 当前强基线矩阵（ETT/Electricity/Traffic，H96，2026-09-26）

已补齐 84 个 validation-only 坐标：DLinear、PatchTST、TimesNet、iTransformer、
MambaACF memory、AutoCon relation-null 和 active，覆盖 ETTh1、ETTm1、Electricity、
Traffic，三 seed。active 相对最强普通基线在 ETTh1/ETTm1 分别差 `6.435%/8.090%`，在
Electricity/Traffic 分别好 `8.945%/4.706%`；相对 null 的变化为
`-0.044%/-0.009%/-0.170%/+0.023%`，仍不是稳定的 AutoCon 因果增益。统一 batch=32
资源测量显示 d32 候选只在 Mamba/TimeMixer/TimesNet/PatchTST 比较中更快，仍慢于
DLinear 和 iTransformer，且峰值显存最高。完整表格见
[`mamba_acf_current_strong_baseline_screen_20260925_retry_readout_20260926.md`](./mamba_acf_current_strong_baseline_screen_20260925_retry_readout_20260926.md)。

### PEMS d32 同快照强基线矩阵：触发停止规则（2026-09-26）

为消除历史结果不可比的问题，重新完成 PEMS03/04/07/08、H96、三个 fresh seed 的
84 作业单快照矩阵：DLinear、PatchTST、TimesNet、iTransformer、TimeMixer、d32
MambaTS v3 的 lambda-zero null 与 active AutoCon。严格校验全部通过：基线
`60/60`、null `12/12`、active `12/12`，没有 test 指标。active 相对同快照最强的
TimesNet 在 PEMS03/04/07 差 `21.704%/33.732%/10.883%`，只在 PEMS08 好 `1.145%`，
满足“至少两个数据集超过 2% 即停止”的预注册停止条件。AutoCon 相对 exact null 的
变化仅为 `-0.010%/-0.160%/+0.012%/-0.159%`，也没有稳定因果增益。因此停止该
d32 分支的 PEMS 测试释放和后续调参；完整读数在
[`pems_d32_same_snapshot_baseline_screen_20260926_readout.md`](./pems_d32_same_snapshot_baseline_screen_20260926_readout.md)。

### MambaTS v8 causal spectral residual：PEMS03 预检停止（2026-09-26）

为针对 TimesNet 在 PEMS03/04/07 的周期性优势，新建 v8：保留 v3 shared
Mamba scan，并从每个观察窗口的 rFFT 提取 top-4 非 DC bin，在未来步做因果
Fourier 重建；频谱残差由零初始化 ReZero 和同一 AutoCon representation 的
horizon/channel gate 控制。PEMS03/H96、seed 2181、3 epoch 的 matched
validation-only 预检中，v8 的最优纯预测验证 MSE 为 `.423632`，同配置 v3 为
`.241888`，恶化 `75.136%`。因此不扩展任何多数据集、多 seed、资源或测试评估，
也不对 top-k/lambda 做事后调参。完整协议、证据边界和 artifact 链接见
[`mambatsv8_spectral_pems03_preflight_20260926.md`](./mambatsv8_spectral_pems03_preflight_20260926.md)。

### MambaACF low-rank channel gate：同构 null 预检（2026-09-26）

针对 v8 失败后保留的低秩、窗口条件化跨变量 gate，完成了 PEMS03/H96、seed
2181、三 epoch 的 paired validation-only 预检。为避免把 AutoCon 差异错误归因于
gate，补跑了同配置的 AutoCon relation-null：null 最优验证 MSE `0.218195`，rank-4
gate 为 `0.215591`，描述性变化 `-1.194%`；gate 参数量增加 `6.39%`。这不是稳定收益
证据，且两者的 AutoCon 诊断均显示正样本退化 `100%`、contrastive anchors `0%`、符号
准确率约 `48.2--48.9%`，说明当前关系目标没有形成有效对比学习信号。plain
MambaACF 的 `0.139925` 只能作为上下文，不能用于归因，因为它没有 AutoCon。该 gate
方向停止多数据集扩展、调参和测试集释放；完整记录见
[`mambaacf_channel_gate_pems03_preflight_20260926.md`](./mambaacf_channel_gate_pems03_preflight_20260926.md)。

### AutoCon relation-mask 修复与 channel gate 三 seed 复验（2026-09-26）

追溯发现，前一轮 gate 预检没有开启 `--mamba_acf_relation_lag_mask`；在当前
`hard_selected` 合同下，这会把所有 in-range pair 当作 positive，因而出现
`100%` degenerate positives。按显式 selected-lag mask 和 lag-pair sampler 修正后，
PEMS03/H96 三 seed 的 gate 相对同构 null 变化为 `-1.640%/-1.051%/+0.559%`，均值
`-0.711%`，2/3 seed 改善。global contrastive anchors 恢复到约 `73--74%`，degenerate
positives 降至约 `19%`，说明对比目标确实被激活；但均值收益不足以支持稳定增益，且
sign accuracy 仅约 `32--34%`。因此将 mask 设为后续关系学习实验的必需合同，channel
gate 仍只保留为 exploratory 候选，不开放测试集或进行事后调参。完整记录见
[`mambaacf_relation_mask_gate_pems03_multiseed_20260926.md`](./mambaacf_relation_mask_gate_pems03_multiseed_20260926.md)。

同一修复合同下，`AutoCon_lambda=.001` 相对严格 lambda-zero null 的三 seed 变化为
`-0.387%/-0.092%/-0.157%`，方向均改善但均值仅 `-0.212%`。故可说该对比目标不再
机械退化，不能说它已经带来有实际意义的预测精度贡献。

补齐同归一化 plain 对照后，plain MambaACF (`MeanStd`) 三 seed 均值为 `0.217667`，
修复后 active AutoCon 为 `0.218292`（`+0.287%`）；且单 seed 的 `.01/.03` 权重均差于
`.001`。因此停止该 PEMS03/H96 坐标上的任何事后 lambda 调参，并将下一次候选筛选移至
新的、预注册的多数据集坐标。

### MambaACF ACF-memory：Weather H96 独立扩展停止（2026-09-26）

将 Electricity/Traffic 中有竞争力的同一 MambaACF memory + relation 合同扩展至独立
Weather H96，三 seed、8 epoch、validation-only。active 相对 exact lambda-zero null
为 `+0.013%/-0.055%/-0.015%`，均值仅 `-0.019%`；虽然约 `45%` global anchors 是可用
contrastive anchors，但没有实际预测增益。该 active 均值 `0.465414` 也比既有 Weather
MambaTS v3 参考 `0.445116` 高 `4.560%`（跨快照比较仅用于保守筛选）。因此停止该
MambaACF 分支在 Weather 的测试集释放、长 horizon 扩展和事后调参。完整读数见
[`mambaacf_weather_h96_extension_readout_20260926.md`](./mambaacf_weather_h96_extension_readout_20260926.md)。

### MambaTS v3 Weather AutoCon 严格复现：通过基线扩展门槛（2026-09-26）

在当前源码快照下完成 Weather/H96、`lambda=0` 与 `lambda=.001` 的严格配对
复现，seed `2021/2022/2023`，共 `6/6` 个 validation-only 作业完成，测试集全部
跳过。null 的均值验证 MSE 为 `0.446700`，active 为 `0.445100`，配对均值改善
`0.3582%`，且 `3/3` seed 同向改善。该结果只说明 MambaTS v3 的 AutoCon 目标在
这一坐标存在小而可重复的方向性信号，尚不能声称跨数据集增益。按预设规则，进入
同一源码快照的 Weather 强基线矩阵；矩阵必须保留 lambda-zero null、报告种子级
波动，并继续禁用测试集。完整读出见
[`MambaTS v3 Weather replication readout`](mambatsv3_weather_autocon_replication_readout_20260926.md)。

### MambaTS v3 Weather H96：同归一化强基线矩阵通过（2026-09-26）

为避免普通基线隐式使用 `LastVal` 而候选使用 `MeanStd` 的不公平比较，重新完成
DLinear、NLinear、PatchTST、TimesNet、iTransformer 与 Mamba2 的 `18/18` 个
MeanStd、三 seed、validation-only 作业；每个配置都显式写入 `AutoCon_wnorm=MeanStd`
和 `skip_test=true`。MambaTS v3 active 的均值验证 MSE 为 `.445100`，精确 null 为
`.446700`；最强外部基线 TimesNet 为 `.459957`，候选低 `3.2301%`，且相对六类
基线均 `3/3` seed 胜出。这个 H96 结果足以进入 H336 的 active/null 确认，不能单独
构成跨 horizon、跨数据集或测试集论文结论。此前未显式 MeanStd 的基线批次仅保留作
命令烟雾证据，不纳入比较表。完整表和 provenance 见
[`Weather H96 MeanStd baseline readout`](mambatsv3_weather_h96_meanstd_baseline_readout_20260926.md)。

### MambaTS v3 Weather H336：AutoCon 严格确认失败，停止（2026-09-26）

在 H96 通过后，执行预注册的 Weather/H336 exact lambda-zero 对照（3 seed、6 个
validation-only 作业、全部 `skip_test=true`）。null 的均值 MSE 为 `.613700`，active
为 `.614133`，active 相对 null 退化 `.0706%`，只有 `1/3` seed 改善。因此 H96 的
小幅 AutoCon 信号没有迁移到更长预测期；按协议停止该 MambaTS v3 Weather 分支的 H336
强基线、H720、测试释放及事后 lambda 调参。H96 只能作为局部短期 validation 信号，
不能作为论文级长预测或 AutoCon 一般收益结论。完整读出见
[`Weather H336 AutoCon confirmation`](mambatsv3_weather_h336_autocon_confirmation_readout_20260926.md)。

### 轻量变量轴 Mamba Solar H336：有效 relation 对比仍未产生预测增益，停止（2026-09-26）

针对 Solar 长预测上已有精度/速度信号的 `NLinearVariableMambaAutoCon`，补齐当前源码
H336 的精确 lambda-zero paired replication。`2022/2023/2024` 三 seed 中 null/active
均值为 `.154233/.154333`，active 退化 `.0648%`，只 `1/3` 改善。relation telemetry
不是退化状态：完成作业约有 `50.36%` contrastive anchors、`39.32%` degenerate positives，
所以不能将负结果归咎于“全正样本”实现错误。按预设规则停止该 Solar 分支的事后调参、
新基线和测试释放；现有结构优势不得归因于 AutoCon。完整读出见
[`Solar H336 AutoCon replication`](nlinear_variable_mamba_solar_h336_autocon_replication_readout_20260926.md)。

### 预测级 ACF 滞后配对：Solar H336 初筛通过，待独立确认（2026-09-26）

为避免只在隐藏表征上施加 relation loss，新加入训练期的预测级 ACF 滞后配对损失：对由
train-only ACF 选出的两个窗口起点，直接匹配两次预测轨迹的逐变量 signed cosine 与训练
ACF。它不增加模型参数，也不进入推理路径；既有 prediction-connected AutoCon 表示和
relation loss 保持启用。Solar H336、fresh seed `2031/2032/2033`、固定权重 `.01` 的
off/on 复验为 `.153843/.153613`，平均改善 `.1495%`，三个 seed 全部改善；每个 on 作业
均有非零 pair loss，约 `89.7%` active anchors 和每 batch `14.39` 个无序有效 pair。

该结果仅通过单一 validation-only 初筛，绝不能解释为论文级精度、跨数据集泛化或速度收益。
由于训练中的 batch-pair 计算，短协议训练墙钟约增加 `17.4%`；prediction-only 推理没有
这条路径，仍需独立资源复测。下一步固定权重、不做 Solar 事后搜索，在独立数据集/预测长度
确认；测试集继续关闭。详见
[`Solar H336 forecast-pair readout`](nlinear_variable_mamba_solar_h336_forecast_pair_readout_20260926.md)。

### 预测级 ACF 滞后配对：Solar H720 独立确认失败，停止（2026-09-26）

固定 H336 初筛通过的 pair weight `.01`，在 Solar H720、fresh seed
`2041/2042/2043` 上执行严格 off/on 确认。pair-off/pair-on 均值为
`.153902/.154059`，pair-on 退化 `.1022%`，只有 `1/3` seed 改善；所有 on
作业均有非零 pair telemetry（约 `89.6%` active anchors、每 batch `14.38`
个有效无序 pair），因此失败不能归因于采样失效。按预注册停止规则，不再跑 ETTh1
确认、事后调权重、基线扩展或测试释放。H336 的 `.1495%` 改善只能保留为局部探索性
结果，不能写成可迁移的 AutoCon 机制贡献。短训练协议中 on 的墙钟约增加 `12.3%`，也不
支持速度优势。详见
[`forecast-pair independent confirmation`](nlinear_variable_mamba_forecast_pair_confirmation_readout_20260926.md)。

### 变量轴 Mamba 的 AutoCon-ACF memory selector：Traffic H96 初筛通过（2026-09-26）

在 `NLinearVariableMambaAutoCon` 的单次变量轴 Mamba 与 NLinear 基座之上，新增一个仅在训练期
ACF memory 已启用时工作的轻量 selector：同一 prediction-connected AutoCon 表示产生 rank-2
horizon 与 rank-1 channel gate，用零初始化 ReZero 调整 train-only ACF memory，不增加第二条
Mamba scan。Traffic/H96、three paired seeds、validation-only 的 selector off/on 均值为
`.396070/.391470`，改善 `1.1615%`，且 `3/3` seed 同向；可用 train-only lags 为
`[72,96,120]`，三组最终 selector scale 为约 `-.77` 至 `-.90`，不是零分支。

两个 phase 均通过严格 validator，测试集保持关闭。该 selector 的 Traffic prediction-only
resource preview 是 `.36568 ms`，对照 memory control 为 `.24714 ms`，故当前结果只能说明一个
有待复验的精度机制，不能宣称速度优势或论文结论。早期 fixed/train-peak 错配与全不可访问 lag
运行已明确隔离且不计入数值。下一步只能在冻结架构、训练合同和新 seed 下做独立数据集或 horizon
确认，不能事后搜索 rank/初始化。详见
[`Traffic H96 selector readout`](nlinear_variable_mamba_acf_memory_selector_traffic_h96_readout_20260926.md)。

Traffic H336 的冻结确认随后被机制审计拒绝：该 horizon 的 train-only resolved lags 为
`[144,168,336]`，均超过 `seq_len=96`，所以 ACF memory 为零、selector scale 在三组
treatment checkpoint 中均为 `0.0`。on/off 的浮点差异不是机制收益，不能计入 `3/3` 胜出或
确认性证据。现已在训练运行时和 phase validator 中加入“selector 必须存在至少一个
`lag <= seq_len`”的硬约束；完整审计见
[`Traffic H336 invalid confirmation`](nlinear_variable_mamba_acf_memory_selector_traffic_h336_confirmation_readout_20260926.md)。

### 变量轴 Mamba 的 AutoCon-ACF memory selector：ETTh1 H96 独立确认通过（2026-09-26）

冻结 Traffic H96 的 selector 架构、lag 策略、训练预算与 relation 配置后，在独立
ETTh1/H96、fresh seed `2064/2065/2066` 上完成 `6/6` validation-only 作业。selector
off/on 均值验证损失为 `.655488/.651336`，改善 `.6335%`，三个 seed 全部改善；每组
treatment checkpoint 的 scale 为非零 `0.4141/-0.4224/0.3974`，train-only 可访问 lag
稳定为 `[72,96,120]`。两 phase 均通过严格 validator，测试集仍关闭。

这使 selector 获得两个数据集上的短期 validation 信号，但不能外推到长 horizon、测试集、
外部基线或速度优势。Traffic 资源预检仍显示 selector 比 memory control 更慢；因此当前
定位是“有待测试集与长 horizon 复核的精度机制”，还不是论文结论。完整读出见
[`ETTh1 H96 selector confirmation`](nlinear_variable_mamba_acf_memory_selector_etth1_h96_confirmation_readout_20260926.md)。
同源外部确认随后完成：DLinear、NLinear、PatchTST、TimeMixer、iTransformer、plain MambaTSv3 共 36 个 GPU validation-only 作业全部通过。fusion candidate 在 METR-LA `0.275437`，相对最强 plain MambaTSv3 `0.285294` 改善 `3.455%`；PEMS-BAY `0.299235`，相对 `0.303959` 改善 `1.554%`；两个数据集均通过预注册 2% external gate。完整读数见 [`same-source baseline confirmation`](../../artifacts/exploratory/metro_pemsbay_mambatsv3_repr_fusion_h12_baseline_confirmation_20260927/readout/readout.md)。

统一 prediction-plus-representation 资源复测后，fusion-off/on 的 median latency 为 METR-LA `0.310/0.323 ms`、PEMS-BAY `0.331/0.349 ms`，额外开销约 `4.4%/5.2%`，低于 10% 预算；此前 32% 左右的差异来自 off 端关闭了表示路径，不能作为公平对照。候选仍只支持“在该 RTX 5090 D、输入 shape 和 forward 协议下，精度通过且表示融合开销受控”的局部结论，不是端到端训练速度或普适速度主张。当前最强候选可以进入冻结 held-out confirmation 设计，但不能直接声称高水平论文结果。
