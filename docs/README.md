# 文档入口

## 当前核心证据线

当前应优先人工复核的主线是 Mamba-1 单扫描 local-cluster ACF-InfoNCE：

- [主干选择与停止分支](research/final_backbone_selection_20261009.md)
- [证据绑定论文骨架](research/local_cluster_manuscript_outline_20261009.md)
- [中文论文初稿](research/local_cluster_acf_infonce_manuscript_draft_20261012.md)
- [提交人工交接](research/mambatsv3_acf_infonce_submission_handoff_20261008.md)
- [独立机器官方 PatchTST 审计](../artifacts/reproducibility/mambatsv3_local_cluster_independent_official_patchtst_20261008/independent_patchtst_audit.json)

只读重验命令：

```bash
python scripts/verify_final_local_cluster_evidence.py
```

该命令只重建派生审计记录，不启动训练、不读取新的测试数据，也不会改变 checkpoint。

其余条目保留为历史实验、基线覆盖与负结果审计，不能覆盖上述主线的范围边界。

人工复核先读 [`core-review.md`](core-review.md)（一页核心结论、代码数据流、当前数字和下一轮门槛），再读 [`research/mamba_paper_execution_protocol_20260903.md`](research/mamba_paper_execution_protocol_20260903.md) 和 [`research/mamba_current_status.md`](research/mamba_current_status.md)。后两页分别给出阶段协议与完整历史审计。

人工复核首先读 [`manual-review.md`](manual-review.md)。该文档给出当前结论、核心复验计划、主要限制和最小复核命令。

10 月 3 日的 ACF-InfoNCE core bundle 仍保留为已审计的历史 release 资料：
[`core evidence`](research/mambatsv3_acf_infonce_core_evidence_20261003.md)、
[`submission audit`](research/mambatsv3_acf_infonce_submission_audit_20261003.md) 和
[`reproducibility guide`](research/mambatsv3_acf_infonce_reproducibility_20261003.md)。
它不能覆盖本页顶部的现行 local-cluster 主线；三份历史 held-out release 的统一索引在
[`core bundle`](../artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003/readout.md)。

新的证据绑定稿件 outline 是 [`ACF-InfoNCE manuscript outline`](research/mambatsv3_acf_infonce_manuscript_outline_20261003.md)。它连同 source manifest 和 claim ledger 都处于 `unverified` 人工签认状态，明确不可提交；官方模型 adapter 与 source-selection 负消融的边界见同目录的 external-baseline protocol 和 source-selection ablation。

最新 ACF 机制读数先读 [`research/mamba_acf_anchor_current_test_replication_readout_20260904.md`](research/mamba_acf_anchor_current_test_replication_readout_20260904.md) 和 [`research/mamba_acf_anchor_memory_interaction_screen_readout_20260904.md`](research/mamba_acf_anchor_memory_interaction_screen_readout_20260904.md)。后者显示没有可靠的 anchor x memory 协同证据；其不可变计划的符号/执行状态更正见 [`research/mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md`](research/mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md)。

## 按需查阅

- `research/autocon_mamba_acf_core_review.md`：AutoCon 原始路径、纯 Mamba 适用边界、ACF memory/关系学习及当前 pilot 的核心复核。
- `research/mamba_acf_scale_conditioning_review_20260902.md`：当前 window-dependent scale conditioning 与 CLAM 的实现边界、最新 synthetic 资源读数和未授权状态。
- `research/mamba_acf_clam_review_20260902.md`：CLAM 候选的短版人工复核稿与最小下一步 screen 建议。
- `research/mamba_acf_resource_baseline_current_20260902.md`：当前源码下 DLinear、PatchTST、Mamba-1、Mamba-2 与 MambaACF 的预测专用资源基线；不含精度结论。
- `research/mamba_current_status.md`：当前 Mamba/AutoCon/ACF 数字、证据等级和下一轮实验优先级。
- `research/publication_readiness_audit_20260930.md`：已冻结 held-out、validation-only 结果、图基线和投稿门槛的逐项准备度审计。
- `research/baseline_coverage_dashboard_20261001.md`：按数据集、模型族和证据等级汇总当前基线覆盖，避免把 validation-only 或失败坐标误写成 held-out 主结果。
- `research/claim_evidence_registry_20261001.md`：把论文主张绑定到具体 readout、审计和证据范围，供人工逐条复核。
- `research/manuscript_claim_binding_audit_20261001.md`：数字结果段落的保守 claim-binding 审计；报告未绑定项，不自动认证稿件。
- `../artifacts/exploratory/claim_evidence_integrity_audit_20261001/audit.json`：稿件 C/E 标记、证据 registry 与本地文件可访问性的端到端审计。
- `../artifacts/exploratory/submission_readiness_audit_20261001/audit.json`：冻结 release、数值、C/E、引用和回归测试的汇总本地审计；仍需作者人工批准。
- `research/references/mamba_autocon_references.bib`：AutoCon 与外部 Mamba 结构参考的最小可核验 bibliography；逐条来源和待作者确认项见 `research/references/citation_verification_20260930.md`。
- `research/nlinear_variable_mamba_acf_selector_traffic_h96_strong_baseline_retry_20260927.md`：当前轻量 variable-Mamba ACF selector 的 Traffic/H96 强基线筛选；通过 validation gate，但测试仍冻结。
- `research/nlinear_variable_mamba_acf_selector_etth1_h96_strong_baseline_confirmation_20260927.md`：同一冻结 selector 的第二数据集 ETTh1/H96 强基线确认与前向资源边界。
- `research/nlinear_variable_mamba_acf_selector_electricity_h96_strong_baseline_confirmation_20260927.md`：第三数据集 Electricity/H96 的负确认；它关闭了当前 selector 的测试与继续调参。
- `research/mamba_acf_channel_mixer_electricity_h96_20260927.md`：新的 MambaACF rank-4 窗口条件跨变量状态混合器 27-job factorial screen；通过混合器与外部基线门槛，但 AutoCon relation 仅通过非劣门槛。
- `research/mamba_acf_channel_mixer_multidataset_h96_20260927.md`：同一 mixer 的 Electricity/Traffic/ETTh1 独立确认总览；高维局部通过、ETTh1 外部基线失败，因此已停止其通用主模型与测试释放。
- `research/pems_dataset_audit_20260909.md`：PEMS03/04/07/08 的文件 hash、数组 shape、loader 窗口合同和无图结构限制。
- `research/mamba_ts_baseline_design_20260909.md`：官方 MambaTS 结构核验、本地 inspired baseline 的实现边界和 PEMS 对照合同。
- `research/mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md`：MambaTS v3 的 AutoCon 权重校准、两 seed validation-only 结果和人工复核边界；低权重近似保持预测，但未证明 AutoCon 增益。
- `research/mamba_pems_ts_v3_autocon_strong_baseline_20260910.md`：固定 `lambda=0.001` 的 MambaTS v3 AutoCon 与 TimesNet 精确坐标复核；H=12 为 8/8 胜出，H=96 为 2/8，不能宣称全跨度优于 TimesNet。
- `research/mamba_pems_ts_v4_autocon_screen_20260910.md`：MambaTS v4 的 train-only ACF residual screen；ACF gate 在 16/16 坐标激活，但相对 v3 基本持平，H=96 仍落后 TimesNet，不能作为确认性主结果。
- `research/mamba_pems_ts_v5_anchor_screen_20260910.md`：MambaTS v5 的 train-only causal signed-ACF anchor screen；H=12 仅微小变化、H=96 基本无效，已归档为中性消融，不作为主论文模型。
- `research/mamba_multidataset_plain_baseline_smoke_20260910.md`：六个 plain baseline 和四个数据集的 24-job 执行 smoke；全部成功但 test withheld，不能当作精度排名。
- `research/mamba_multidataset_plain_baseline_full_screen_20260910.md`：同一来源下四个数据集、六个 plain baseline、三 horizon、三 seed 的完整 216-job validation-only 结果；包含逐组 MSE、成对 seed delta、训练墙时和严格限制。结果显示 plain Mamba 的精度明显依赖数据集，不能宣称普遍优于 PatchTST/TimesNet。
- `research/mamba_acf_multidataset_candidate_screen_20260910.md`：在同一四数据集合同下完成的 48-job MambaACF/ACF memory/AutoCon relation H96 配对 screen；候选在 Electricity/Traffic 低于外部 baseline，但 ETTh1/ETTm1 仍落后 PatchTST，且 `lambda=0.001` 相对同架构 `lambda=0` 的增量接近零，不能写成普遍或对比学习带来的稳定增益。
- `research/mamba_ts_v3_multidataset_autocon_screen_20260922.md`：当前快速 MambaTS v3 主干的 24-job AutoCon null screen；`lambda=0.001` 在四数据集均未优于 `lambda=0`，但 v3 在 Electricity/Traffic 的 H96 validation 描述性结果明显优于原始 Mamba-1/Mamba-2，值得做长 horizon 复验。
- `research/mamba_ts_v3_multihorizon_autocon_screen_20260922.md`：72-job H336/H720 三臂复验；`lambda=0.001` 相对同构 null 的变化仅 `-0.005%` 至 `+0.110%`，没有稳定增益，测试继续隔离。
- `research/mamba_ts_v3_repr_forecast_fusion_screen_20260922.md`：将 AutoCon 表示直接接入预测的 24-job H96 screen；Electricity `-0.666% (3/3)`，但两个 ETT 数据集退化，仍不是论文候选。
- `research/mamba_ts_v3_channelwise_repr_fusion_stop_20260922.md`：通道保留表示融合因 Traffic 训练与资源开销不满足效率门槛而停止，没有可用精度结果。
- `research/mamba_ts_v3_global_repr_autocon_calibration_20260922.md`：60-job λ/warm-up 校准；没有候选同时避免 ETT/Traffic 退化，因此停止当前全局融合的超参数调参。
- `research/mamba_ts_v3_variable_permutation_screen_20260922.md`：MambaTS v3 train-only variable permutation 的 36-job 多数据集 screen；无跨数据集收益，保留为负消融。
- `research/mamba_ts_v3_forecast_acf_consistency_screen_20260922.md`：直接将 train-only ACF 约束施加到预测时间序列的 24-job clean rerun；ETT 微小改善但 Electricity/Traffic 不一致，归档为负消融。
- `research/mamba_ts_v4_acf_memory_selector_screen_20260922.md`：v4 static ACF memory + AutoCon 表示选择器的 24-job 三 seed screen；四数据集均改善，但 ETTh1 仍落后 PatchTST，已冻结为下一轮多 horizon/强基线确认候选。
- `research/mamba_ts_v4_acf_memory_selector_multihorizon_20260922.md`：同一 selector 的 H336/H720 48-job 三 seed screen；ETTm1/Electricity 保持小幅收益，但 ETTh1/H720 与 Traffic/H336 退化，未通过跨数据集/跨 horizon gate。
- `research/mamba_ts_v4_acf_memory_channel_selector_pilot_20260922.md`：针对上述失败坐标的 8-job 单 seed 逐变量置信度 pilot；Traffic 两个 horizon 和 ETTh1/H720 改善，但 ETTh1/H336 退化，暂不晋级主模型。
- `research/mambatsv3_transfer_baseline_pilot_20260922.md`：新增 Weather、Exchange-Rate、ILI 的 54-job 统一基线 pilot；MambaTS v3 AutoCon 在 Weather/H96 为六臂最佳且略快于 PatchTST，但在 Exchange/ILI 不具普适竞争力。
- `research/mambatsv3_weather_multihorizon_confirmation_screen_20260922.md`：Weather H96/H336/H720 的 63-job 三 seed validation-only screen，含 MambaTS v3 plain、AutoCon null/active 与四个强基线；候选在三个 horizon 都优于外部基线，但 AutoCon 相对 null 只在 H96 改善，不能宣称长预测独立增益。
- `research/mambatsv3_weather_test_release_20260922.md`：Weather 五 fresh seed、三预测长度、75-job held-out test release；候选在 12 个已注册基线单元均通过 `1.02` 非劣门槛且 5/5 低 MSE，但结论只适用于该数据集，AutoCon 独立增益仍未确认。
- `research/mambatsv3_transfer_baseline_extension_20260922.md`：Weather/Exchange/ILI 的 144-job validation-only 扩展矩阵；Weather 最佳，但 Exchange 和全部 ILI 坐标落后 DLinear/PatchTST，不能选择性写成普适结果。
- `research/mambatsv3_d24_highdim_h12_transfer_20260927.md`：PEMS H12 d24 MambaTSv3 配置迁移到 Electricity/Traffic 的 54-job validation-only 复核；两个数据集均未通过 external gate，停止高维扩展与测试释放。
- `research/bipatchmamba_autocon_transfer_h12_screen_20260927.md`：Bi-Mamba+ 启发的双向 patch Mamba 四数据集 72-job validation-only 筛选；AutoCon 相对 null 多数改善，但 active 始终劣于 plain，已停止当前加性表示修正并转向有界门控。
- `research/bipatchmamba_repr_gate_pilot_h12_20260927.md`：有界、零初始化 representation gate 的 24-job validation-only 复核；消除了加性修正的大幅退化，但未稳定超过 plain BiPatch，已冻结该分支且测试继续隔离。
- `research/metro_pemsbay_mambatsv3_autocon_h12_screen_20260927.md`：新增 METR-LA/PEMS-BAY 后的 32-job MambaTSv3 AutoCon H12 validation-only 迁移；通过外部与 exact-null 非劣门槛，已注册冻结测试确认，AutoCon 独立精度贡献仍未成立。
- `research/metro_pemsbay_mambatsv3_d24_test_release_20260927.md`：METR-LA/PEMS-BAY 的 76-job held-out release；PEMS-BAY 通过全部主要比较器，METR-LA 对 PatchTST/TimeMixer 失败，结论限定为混合的短期非图交通证据。
- `research/mambatsv3_lowdim_budget_resolution_20260922.md`：Exchange/ILI 的 36-job、20 epoch 长预算复验；差距仍存在，说明不是训练轮数不足，下一步转向 AutoCon 表示门控的线性专家混合。
- `research/mambatsv3_repr_linear_expert_retry_20260922.md`：AutoCon 表示门控共享线性专家的 12-job 修正配对重试；仅 ILI H24 小幅改善，其余三坐标退化，作为负消融停止，不进入测试释放。
- `research/mambatm_h96_baseline_extension_20260922.md`：公开 TimeMachine 风格四路径 Mamba-1/Mamba-2 的 108-job 统一 AutoCon/强基线矩阵；精度和端到端时间均不满足主模型要求，保留为结构基线。
- `research/mambasm_h96_screen_20260922.md`：S-Mamba/TimesMamba 启发的双向变量扫描 MambaSM 之 72-job H96 validation-only 筛选；参数更少但没有速度优势，且除 ETTh1 对 MambaTS v3 的改善外未通过强基线门槛，停止扩展。
- `research/mambatsv6_h96_screen_20260923.md`：DLinear 风格分解主路径加零初始化 Mamba 残差的 84-job H96 validation-only 筛选；缩小了低维缺口且 Weather 最佳，但仍非跨数据集非劣、且同形状前向更慢，停止扩展。
- `research/mambatsv6_residual_init_screen_20260923.md`：唯一预记录的 v6 Mamba residual `0.1` warm 初始化之 36-job Exchange/ILI 筛选；Exchange 仅微小改善、ILI 稳定退化，停止该超参数方向。
- `research/mamba_ts_v4_multidataset_acf_screen_20260922.md`：MambaTS v4 train-only ACF memory/representation fusion 的 36-job H96 screen；Traffic 退化约 `0.64%`，relation null-to-active 近零，且旧工件不具备新 sidecar replay 资格。
- `research/mamba_pems_ts_long_context_acf_screen_20260910.md`：`seq_len=384`、lag=288 的长上下文 ACF screen；v4 相对同窗口 v3 仅 1/4 胜出且平均 `+0.737%`，wall-time 约增加 3.2%，该方向已停止扩展。
- `research/mamba_pems_ts_v4_autocon_repr_fusion_screen_20260910.md`：把 train-only raw ACF summary 注入 v4 AutoCon representation 的 16-job screen；gate 16/16 激活，但 H=12 基本持平、H=96 仅轻微改善且仍落后 TimesNet。
- `research/mamba_pems_ts_v3_autocon_lambda_calibration_20260910.json`、`research/mamba_pems_ts_v3_autocon_lambda_fresh_seed_20260910.json`：两个冻结 exploratory lambda 矩阵的计划文件；测试集保持关闭。
- `research/mamba_pems_baseline_autocon_pilot_20260909.json`：4 个 PEMS 数据集、plain/AutoCon 基线和 H12/H96 validation-only 矩阵；未授权启动。
- `research/mamba_ts_resource_smoke_20260909.md`：MambaTS plain/AutoCon 的固定 synthetic GPU 结构读数；不含训练或精度结论。
- `research/mamba_acf_lite_autocon_pilot_20260909.json`、`research/mamba_acf_lite_autocon_resource_preflight_20260909.md`：单尺度/因果/低秩 Lite AutoCon 候选的 24-job 未授权计划与结构资源预检；不含训练精度。
- `research/mamba_paper_execution_protocol_20260903.md`：短版人工复核协议；汇总 MeanStd 公平性、variable-token representation fusion、基线缺口、确认性矩阵和论文最低门槛。
- `research/mamba_variable_repr_fusion_candidate_20260903.md`、`research/mamba_variable_repr_fusion_screen_plan_20260903.json`：variable-token 表示融合候选及其 12-job 未授权 off/on 计划。
- `research/mamba_acf_v3_screen_readout.md`：当前代码下 ACF memory、hard/soft relation 的 30-run 配对 screen；明确 `n=3` 的探索性边界。
- `research/mamba_acf_v3_relation_screen_plan.json`：上述五阶段 screen 的不可变训练合同、数据哈希和对比定义。
- `research/mamba_acf_variable_mixer_screen_readout.md`：变量 token 分支的独立 12-run 配对读数；每数据集 n=3，仅作 exploratory evidence。
- `research/mamba_acf_variable_memory_relation_screen_readout.md`：变量 token + train-only ACF memory + signed relation 的已完成 18-run 配对 screen；每数据集仅 3 个 seed，仍仅作 exploratory evidence。
- `research/mamba_acf_relation_null_control_readout_20260901.md`：同一 AutoCon/同一 lag-pair sampler 下 `lambda=0` 与 `0.03` 的 12-run 配对 null control；包含 active-anchor、high-ACF exclusion 和 sign diagnostics。
- `research/mamba_acf_counterfactual_audit_20260903.md`：同一 checkpoint 的 ACF memory on/off 只读反事实、signed ACF 残差校正公式、sidecar/hash 索引和统计边界；不含重训结果。
- `../artifacts/exploratory/mamba_acf_relation_null_control_replay_20260901/replay_report.md`：上述 12 个 sidecar/checkpoint 的隔离 `is_training=0` 回放；只验证加载兼容性，不提升训练证据等级。
- `research/mamba_acf_channel_gate_screen_plan.json`、`research/mamba_acf_channel_gate_screen_readout_20260901.md`：raw-source 三臂低秩 channel gate screen（no gate、ordinary gate、ACF-conditioned gate），18 个作业和门控激活/资源读数；仅作探索性证据。
- `research/mamba_acf_low_rank_head_screen_plan.json`、`research/mamba_acf_low_rank_head_screen_readout_20260901.md`：full-rank 与 rank-16 预测头的 12-run 配对 screen；参数减少但 MSE 退化，作为压缩负结果归档。
- `research/mamba_acf_relation_coverage_screen_plan.json`、`research/mamba_acf_relation_coverage_screen_readout_20260901.md`：random/balanced lag-pair 与 active-anchor normalization 的 18-run coverage screen；包含每 lag exposure、实际训练轮数和 paired readout，仅作机制诊断。
- `research/mamba_acf_relation_target_screen_plan.json`、`research/mamba_acf_relation_target_screen_readout_20260901.md`：当前代码 raw train-only sidecar 下 hard/soft05 target-support 的 12-run 配对 screen；soft05 在 Traffic 0/3 seed 改善，按注册规则停止扩展。
- `research/mamba_acf_channel_order_conditioning_screen_readout_20260902.md`：通道 strength 顺序与 ACF token conditioning 的 18-run 探索性 screen；immutable snapshot 校验通过，但当前源码存在四个 post-launch drift 文件，不能称为 current-tree 重跑。
- `research/mamba_acf_memory_gate_mode_resources_20260902.md`：`hidden`/`variable_token` memory-gate 的结构与 synthetic GPU 成本；尚未训练，不能当作精度或端到端速度结果。
- `research/mamba_acf_memory_gate_mode_screen_plan_20260902.json`、`research/mamba_acf_memory_gate_mode_screen_review_20260902.md`（若已人工批准后才可执行）：门控模式精度 screen 的未授权草案；默认不启动训练。
- `scripts/analyze_mamba_acf_memory_gate_mode_screen.py`：上述门控 screen 的严格只读分析器；当前训练工件不存在时返回 `NOT_RUN`，不会生成伪读数。
- `research/mamba_acf_anchor_copy_screen_plan_20260902.json`：独立 raw static direct-anchor residual 的 12-job 未授权单因素 screen；两臂只改变 `mamba_acf_anchor_copy`，默认不启动训练。
- `research/mamba_acf_channel_lag_bank_screen_plan_20260903.json`、`research/mamba_acf_channel_lag_bank_contract_20260903.md`：共享 train-only ACF lag bank 上的逐通道 top-k support mask strict 历史草案；因当前全窗口归一化而阻断，未授权、未训练。
- `research/mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json`：同一候选的 forecast-level causal 修订草案；明确不声称 representation prefix-causality，12 个 job 已 validate-only 通过但仍未授权、未训练。
- `scripts/analyze_mamba_acf_channel_lag_bank_screen.py`：该修订计划的严格只读 paired readout；训练工件不完整时返回 `NOT_RUN`（退出码 `3`），不写伪结果。
- `../artifacts/exploratory/mamba_acf_channel_lag_bank_resource_current_20260903/`：Traffic `C=862`、50/200 device-resident forward 的 lag-bank off/on 资源记录；仅结构开销，不含精度或端到端速度证据。
- `research/mamba_data_dependent_channel_mixer_screen_plan_20260903.json`、`research/mamba_data_dependent_channel_mixer_screen_readout_20260903.md`：CMamba-inspired 窗口条件低秩通道状态混合器的 60-job H96 配对 screen；默认关闭、未授权、未训练，文档明确共享 AutoCon 状态与非严格 representation-causality 边界。
- `../artifacts/exploratory/mamba_data_dependent_channel_mixer_resource_smoke_20260903.json`：该候选在 RTX 5090 D 上 4 次重复的 prediction-only device-resident 资源 smoke；延迟/峰值/参数仅作结构预筛查，不含训练精度或端到端速度。
- `research/mamba_acf_step_reliability_screen_plan_20260903.json`、`research/mamba_acf_step_reliability_screen_readout_20260903.md`：静态 raw ACF memory 的 horizon-step reliability 单因素 12-job screen；保留同一 `MambaACFAutoCon` 和 signed relation，当前仅 validate-only、未授权、无精度结果。
- `research/mamba_acf_forecast_relation_view_screen_plan_20260903.json`、`research/mamba_acf_forecast_relation_view_screen_readout_20260903.md`：encoder/forecast relation view × lambda 0/.03 的 24-job 配对 screen；当前仅 validate-only、未授权，readout 缺少完整工件时返回 `NOT_RUN`，不读取 test 指标。
- `../scripts/smoke_mamba_acf_forecast_relation.py`：当前环境下 Mamba-1/Mamba-2 forecast relation view 的 CUDA shape/finite/backward smoke；只验证实现，不产生训练结果。
- `research/experiment_runtime_estimate_20260903.md`：基于成功 manifest 的只读运行成本估计；明确 180-job 基线扩展约 13.9 小时串行粗估，以及不能由该估计推出速度结论的边界。
- `research/mamba_acf_anchor_copy_screen_review_20260902.md`：该 screen 的 estimand、因果索引、容量边界和人工启动检查项。
- `research/mamba_acf_anchor_copy_resource_smoke_20260902.md`：anchor on/off 的独立合成 GPU 资源 smoke；当前源码指纹下的 50/500 测量约为 ETTh1 median `+4.04%`、Traffic median `+4.35%`，记录含源码指纹，不等同精度或端到端速度。
- `scripts/analyze_mamba_acf_anchor_copy_screen.py`：anchor-copy screen 的严格只读分析器；缺少完整工件时返回 `NOT_RUN`，不会生成伪读数。
- `research/mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json`、`research/mamba_acf_nlinear_baseline_gap_screen_review_20260902.md`：ETTh1/Traffic H96 NLinear 基线缺口的 6-job 未授权登记；结果不会追溯性并入旧矩阵。
- `research/mamba_acf_fair_baseline_screen_plan.json`、`research/mamba_acf_fair_baseline_screen_readout_20260901.md`：ETTh1/Traffic、H96、DLinear/PatchTST/Mamba-1/Mamba-2 与 MambaACF 组件的 42-job 公平探索性基线。
- `research/mamba_acf_variable_token_fair_screen_plan.json`、`research/mamba_acf_variable_token_fair_screen_readout_20260901.md`：当前代码 raw sidecar 下 variable-token memory、同架构 `lambda=0` null、`lambda=.03` relation 和 horizon calibration 的 30-job 严格 screen；唯一 relation 对照为同一 AutoCon 架构的 null-to-relation。
- `../artifacts/exploratory/mamba_acf_variable_token_resources_20260901/`：variable-token calibrated candidate 的 synthetic resource 记录；不等同端到端速度。
- `../artifacts/exploratory/mamba_acf_relation_target_screen_replay_20260901/replay_report.md`：上述 12 个 current-tree checkpoint/sidecar 的隔离 `is_training=0` 回放；12/12 contract 与 metric 通过，只验证评估链路。
- `../artifacts/exploratory/mamba_acf_channel_gate_replay_20260901/replay_report.md`：上述 18 个 gate checkpoint/sidecar 的隔离 `is_training=0` 回放；18/18 contract 与 metric 通过。
- `../artifacts/exploratory/mamba_acf_channel_gate_resources_20260901/`：三臂参数量、显存和 synthetic device-resident latency；不等同端到端训练速度。
- `../artifacts/exploratory/mamba_acf_anchor_copy_resources_current_20260902/`：direct-anchor on/off 的当前源码 synthetic resource JSON/CSV；不含数据集训练结果。
- `research/mamba_acf_repr_fusion_screen_readout_20260901.md`：raw-source ACF representation fusion 的 12-run screen；当前效果近零，门控保持接近关闭。
- `research/mamba_acf_variable_memory_relation_screen_plan.json`：上述 screen 的冻结启动前合同。其 `execution` 字段保留启动前状态；实际执行以三个 phase 的 manifest、provenance 和 `phase_validation.json` 为准。
- `research/mamba_acf_sidecar_replay_readout.md`：绑定 provenance snapshot 的 18-run sidecar 持久化/合同 readout；指标来自快照训练过程，仍仅作 exploratory evidence。
- `research/mamba_acf_sidecar_replay_audit.md`：sidecar readout 的人工复核附录，记录 immutable source snapshot 漂移、strict 18/18 合同审计，以及 standalone replay 与 current-tree 训练之间的边界。
- `../artifacts/exploratory/mamba_acf_data_bound_replay_20260902_v2/replay_report.md`：三阶段历史 checkpoint 的最终 schema-4 data-bound 回放；18/18 contract 与 metric 通过，并在执行前核对数据集、provenance、配置、全部输入工件哈希和回放工具 SHA-256。它仍是评估路径证据，不是 current-tree 重训或 confirmatory 结果。
- `research/mamba_acf_live_screen_current_validator_audit_20260902.md`：当前 validator 对旧三阶段 phase 的只读复核；单独列出 source-drift 和旧 relation sampler telemetry 缺失，不修改历史工件。
- `../artifacts/exploratory/mamba_acf_sidecar_standalone_replay_20260831_audit/replay_report.md`：当前 loader 对 18 个历史 checkpoint+sidecar 的 `is_training=0` 独立加载复核；同目录的 `replay_manifest.json` 与 `replay_results.jsonl` 提供启动前冻结和逐 job 审计；只验证评估路径，不升级训练或论文证据等级。
- `research/mamba_acf_sidecar_replay_plan.json`：上述 replay 的独立计划与输出坐标；它不替代未授权的 884-job 确认性方案，也不与旧 screen 数值混合。
- `research/mamba_acf_confirmation_protocol.md`：四数据集、多 horizon 的 launch-time 确认性方案草案；当前未授权、未启动。
- `research/mamba_architecture_survey_20260831.md`：外部 Mamba 时间序列结构的源码核验、迁移边界、优先级和公平实验合同。
- `research/mamba_architecture_external_audit_20260903.md`：按固定官方仓库 commit 核验 MambaTS、TimeMachine、CMamba、ms-Mamba、S-Mamba、MambaMixer、Mamba4Cast 的源码路径、证据等级，以及与当前 AutoCon/ACF 接口的具体迁移边界；不含本仓库新训练结果。
- `research/mamba_timemachine_baseline_screen_readout_20260907.md`：新增 TimeMachine 风格 Mamba-1/Mamba-2 的代码合同、真实 CUDA smoke、结构资源读数和未授权训练边界。
- `research/mamba_timemachine_baseline_screen_plan_20260907.json`：5 数据集、11 模型臂、3 seed、H96 的 165-job fresh baseline screen；当前仅 validate-only，未授权、未训练。
- `../artifacts/exploratory/mamba_timemachine_resource_smoke_20260907/`：Mamba2、MambaTM2、PatchTST 的同形状 prediction-only 资源预筛查；不含精度或端到端速度结论。
- `research/mamba_patch_token_pilot_readout_20260907.md`：`MambaACFAutoCon` patch-token off/on 的 ETTh1/Traffic 单 seed validation-only 工程 pilot；包含精度信号、参数/延迟代价和 relation active-anchor 退化诊断。
- `../artifacts/exploratory/mamba_patch_token_pilot_20260907/`：上述 pilot 的 checkpoint、loss telemetry、配置和 prediction-only resource JSON/CSV；不含 test 指标，不进入论文主表。
- `research/mamba_patch_token_contrastive_pilot_readout_20260907.md`：打开 `hard_selected + lag_mask` 后的有效 relation-learning 配置复验；ETTh1/Traffic 单 seed validation-only，不能与前一份 all-candidate pilot 混合。
- `../artifacts/exploratory/mamba_patch_token_contrastive_pilot_20260907/`：上述修正 pilot 的 checkpoint、lag-pair exposure、loss telemetry 和配置；显式跳过 test。
- `research/mamba_patch_token_contrastive_screen_plan_20260907.json`：下一轮 4 数据集、H96、3 seed、patch off/on 的 24-job 未授权筛选草案；已 validate-only 通过，未启动训练。
- `research/mamba_patch_token_contrastive_screen_preflight_readout_20260907.md`：上述 24-job screen 的 8 条 prediction-only 资源预检；四数据集 patch-on median latency 增加约 `40--68%`，训练仍未授权。
- `../artifacts/exploratory/mamba_patch_token_contrastive_screen_20260907/`：上述资源预检的 JSON/CSV 工件；不含训练 checkpoint、精度或端到端速度结果。
- `research/mamba_variable_group_pilot_readout_20260907.md`：Traffic/C862 上 variable-token 分组 `1/4/8/16/32/64` 的资源扫描与 `1/16` 配对训练；分组降低峰值显存但没有降低延迟，已停止扩展。
- `../artifacts/exploratory/mamba_variable_group_pilot_20260907/`：上述分组 pilot 的 checkpoint、ACF telemetry 和资源 JSON/CSV。
- `research/mamba_variable_repr_fusion_pilot_readout_20260907.md`：ETTh1/Traffic、H96、单 seed 的 variable-token 表示融合 off/on pilot；两个数据集均无 validation 改善，prediction-only 延迟方向不一致，已停止扩展。
- `../artifacts/exploratory/mamba_variable_repr_fusion_pilot_20260907/`：上述 pilot 的 checkpoint、relation telemetry、配置和 prediction-only/representation-requested 资源 JSON/CSV。
- `research/baseline_traffic_pilot_readout_20260907.md`：Traffic/C862/H96、单 seed 的 DLinear/NLinear/PatchTST/TimesNet/iTransformer/Mamba-1/Mamba-2 native baseline pilot；用于基线入口和资源覆盖，明确不能与 AutoCon/MeanStd 候选直接排名。
- `../artifacts/exploratory/baseline_traffic_pilot_20260907/`：上述七个模型的训练日志、checkpoint、配置和统一 GPU resource JSON/CSV。
- `research/baseline_native_pilot_summary_20260907.md`：ETTh1/ETTh2/ETTm1/ETTm2/Traffic 五数据集 native baseline 汇总；包含 validation、训练 wall time 和 GPU resource 边界，不能替代 normalization-parity 确认实验。
- `../artifacts/exploratory/baseline_etth1_pilot_20260907/`、`../artifacts/exploratory/baseline_etth2_pilot_20260907/`、`../artifacts/exploratory/baseline_ettm1_pilot_20260907/`、`../artifacts/exploratory/baseline_ettm2_pilot_20260907/`：四个 ETT 数据集七模型 pilot 的原始工件。
- `research/mamba_meanstd_baseline_pilot_20260907.json`：四数据集、七模型、H96、三 seed 的 MeanStd 公平性 pilot 注册计划；84 个作业已完成 validation-only 训练，测试集保持关闭。
- `research/mamba_meanstd_baseline_pilot_readout_20260907.md`：上述 pilot 的核心 validation/资源表和人工复核结论。
- `../artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/summary/meanstd_baseline_pilot_readout.md`：上述 84 个作业的三 seed validation、训练 wall time 和 prediction-only GPU 资源汇总；这是短预算 exploratory 读数，不能替代正式论文矩阵。
- `../artifacts/exploratory/mamba_meanstd_baseline_pilot_20260907/summary/paired_deltas.csv`：同一数据集/seed 下 Mamba-1/2 相对五个非 Mamba 基线的 paired delta 与胜负计数。
- `scripts/summarize_meanstd_baseline_pilot.py`：从 manifest、checkpoint `loss_logs.json` 和资源 CSV 重建上述汇总的只读工具。
- `research/mamba_classical_baseline_extension_pilot_20260907.json`：Weather/Exchange-Rate 两个新增数据集、五个经典模型（Autoformer/FEDformer/ETSformer/Transformer/Crossformer）加七个核心模型的 72-job H96 MeanStd pilot 计划。
- `../artifacts/exploratory/mamba_classical_baseline_extension_pilot_20260907/summary/classical_extension_pilot_readout.md`：上述 72 个 validation-only 作业的三 seed MSE、配对 delta、训练耗时和 24 条 prediction-only 资源记录；不含 test 指标。
- `scripts/summarize_classical_extension_pilot.py`：从扩展 pilot manifest、checkpoint loss trajectory 和资源 CSV 重建汇总的只读工具。
- `research/mamba_classical_baseline_core_extension_pilot_20260907.json`：ETTh1/ETTm1/Electricity/Traffic 四个核心数据集上五个经典模型的 60-job H96 MeanStd pilot 计划，用于补齐六数据集基线覆盖。
- `../artifacts/exploratory/mamba_classical_baseline_core_extension_pilot_20260907/summary/classical_extension_pilot_readout.md`：上述 60 个 validation-only 作业的三 seed MSE、训练耗时和 20 条 prediction-only 资源记录；不含 Mamba 配对 delta 或 test 指标。
- `research/mamba_normalization_parity_pilot_20260907.md`、`research/mamba_normalization_parity_pilot_readout_20260907.md`：ETTh1/Traffic 五模型、MeanStd、两 seed 的 parity pilot 协议与读数；AutoCon relation 已执行但平均增益接近零，仍非确认性结果。
- `../artifacts/exploratory/mamba_normalization_parity_pilot_20260907/`：上述 20 个训练作业和 10 份 prediction-only resource 工件。
- `research/mamba_variable_parity_pilot_20260907.md`、`../artifacts/exploratory/mamba_variable_parity_pilot_20260907/`：variable-token + AutoCon 的四作业 paired pilot；Traffic 两 seed 改善但推理代价显著，待多 seed 复验。
- `research/mamba_variable_parity_replication_20260907.md`、`research/mamba_variable_parity_replication_readout_20260907.md`、`../artifacts/exploratory/mamba_variable_parity_replication_20260907/`：补充 seed 2023 的四作业 replication；合并三 seed 后 Traffic 为 3/3 改善，但两数据集 prediction-only latency 仍增加约 69%，不支持效率主张。
- `research/results_evidence_readout.md`：冻结结果与新 MambaACF pilot 的一页式证据分级。
- `research/corrected_training_plan.json`：当前核心复验计划（输出在 `artifacts/paper_final/corrected/`）。
- `research/final_training_plan.json`：历史 strict 计划；结果属于修正前 pilot。
- `research/mamba2-autocon-protocol.md`：历史 strict 方法协议与模型配置；当前 corrected 参数以 `corrected_training_plan.json` 为准。
- `research/provenance-and-reproducibility.md`：源码快照、数据哈希和可复现性规则。
- `research/paper-statistics.md`：确认性结果的配对统计、Holm/FDR 校正和分析代码哈希。
- `research/resource-benchmark.md`：参数量、显存、推理吞吐/延迟和 manifest 训练耗时的独立采集流程。
- `research/resource-v3-current-traffic-report.md`：当前源码在 Traffic/C862/H96 下统一 50/200 计时协议的资源复测；只支持 synthetic forward 负载结论。
- `research/highdim-extension-protocol.md`：Electricity/Traffic 高维扩展的注册矩阵、数据合同、gate 和当前预检边界。
- `research/extension_baseline_plan.json`：新增高维纯预测基线的最小探索性注册矩阵；包含 DLinear、NLinear、PatchTST、TimesNet、Mamba1、Mamba2，结果不自动进入冻结主表。
- `research/benchmark-matrix.md`：数据集、模型、计划和结果覆盖的人工复核说明；由 `scripts/audit_benchmark_matrix.py` 生成 JSON/CSV/Markdown 清单。
- `research/dataset_baseline_inventory_20260903.md`：2026-09-03 数据集、模型、实际结果坐标、NLinear 缺口、资源边界和建议的最小公平基线矩阵；只读人工复核稿。
- `research/benchmark_audit_current_20260907.md`：当前 53 个计划、144 个 phase、3,009 条 metric artifact 的只读覆盖审计；明确 NLinear、四数据集 plain baseline、TimeMachine 和 patch-token screen 的未完成状态。
- `research/mamba_multidataset_plain_baseline_expansion_plan_20260903.json`：新增 ETTh1、ETTm1、Electricity、Traffic 的六模型（DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2）三 horizon、三 seed validation-only 基线矩阵；216 个作业仍未授权、未训练。
- `research/acf-channel-ordering.md`：`input`/`strength`/`similarity` 通道顺序的 train-only 语义、复杂度和 permutation 风险。
- `research/optional_array_dataset_adapter_20260903.md`：Solar/PEMS 可选 NPZ/TXT 数据适配器的格式、路径、通道和缺失文件规则；不包含下载数据或训练结果。
- `research/benchmark_inventory_current_20260903.md`：修正日期后缀计划扫描后的当前 35-plan/102-phase 库存、可用指标覆盖和候选晋级门槛；先读此页可避免把旧审计快照当成当前状态。
- `scripts/attest_selection.py`：对已完成 selection 的 validation trajectory、冻结 lambda 和 provenance 做事后审计；新的确认性阶段还会在启动前再次 replay 该侧车。
- `reports/strict-evidence-status.md`：历史六阶段的完整性状态。
- `reports/*.md`：按阶段生成的详细统计报告。
- `artifacts/mamba_acf_selector_replication_20260831/`：Traffic/H96 可用性加权 ACF memory 的 3-seed exploratory 配对工件。
- `reports/archive/`、`research/archive/`：仅作历史记录，不进入主结论。
- `references/`：论文 PDF 与本地文本，仅用于文献核对。

## 运行入口

计划校验：

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/corrected_training_plan.json --validate_only
```

新增 screen 仅做合同/缺产物检查：

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_acf_anchor_copy_screen_plan_20260902.json --validate_only
conda run --no-capture-output -n mamba python scripts/analyze_mamba_acf_anchor_copy_screen.py \
  --plan docs/research/mamba_acf_anchor_copy_screen_plan_20260902.json
```

第二条命令在未授权、未训练状态下应返回 `NOT_RUN`（退出码 `3`），不会创建结果文件。

Mamba 依赖检查：

```bash
conda run --no-capture-output -n mamba python scripts/smoke_mamba1.py
conda run --no-capture-output -n mamba python scripts/smoke_mamba2.py
conda run --no-capture-output -n mamba python scripts/smoke_mamba_acf_forecast_relation.py
```

代码回归测试（`pytest` 已安装在 `mamba` 环境）：

```bash
conda run --no-capture-output -n mamba python -m pytest -q
```

正式结果必须经过 `run_training_plan.py`、`validate_training_phase.py` 和报告脚本；不要直接把单次 `run.py` 输出写入论文。

Benchmark 覆盖审计（只读）：

```bash
conda run --no-capture-output -n mamba python scripts/audit_benchmark_matrix.py \
  --plan-dir docs/research --artifact-root artifacts \
  --output-dir artifacts/benchmark_audit
```

## 结果边界

`artifacts/paper_final/strict/` 是历史 pilot，`artifacts/paper_final/corrected/` 是独立复验。两者使用不同 provenance，禁止混合统计。详细日志、checkpoint 和 manifest 不复制到 `docs/`，以免产生第二份结果来源。
