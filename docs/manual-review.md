# 人工复核核心文档

> **现行状态（2026-10-09）：** 本页以下内容是历史阶段索引，不能覆盖当前
> local-cluster 主证据线。当前人工复核入口为
> [主干选择](research/final_backbone_selection_20261009.md)、
> [证据绑定论文骨架](research/local_cluster_manuscript_outline_20261009.md) 和
> [提交交接](research/mambatsv3_acf_infonce_submission_handoff_20261008.md)。
> 可运行 `python scripts/verify_final_local_cluster_evidence.py` 重建派生审计；
> 该命令不训练、不读取新的测试数据。

**项目：** Mamba-1/Mamba-2 + AutoCon 自监督对比预测  
**更新：** 2026-09-10  
**用途：** 人工核对当前证据边界；不是投稿定稿，也不替代冻结工件。短版阶段顺序和论文门槛见 [`research/mamba_paper_execution_protocol_20260903.md`](research/mamba_paper_execution_protocol_20260903.md)。

若只需要一页判断，请先读 [`core-review.md`](core-review.md)；本页保留完整的历史结果、工件 provenance 和人工 gate。

> 最新优先阅读：[`research/mamba_current_status.md`](research/mamba_current_status.md)。本页保留历史阶段索引；fair baseline、variable-token relation null、raw relation-target、representation fusion、channel-gate、low-rank head、relation-coverage、direct-anchor、单尺度/多尺度和逐步可靠性 screen 都是探索性证据，不应与旧 screen 或后验高维表混合。

## 当前判断

2026-09-22 的快速 MambaTS v3 H96 screen 已完成 `24/24` 个作业，后续 H336/H720 三臂 screen 也已完成 `72/72`。严格的 AutoCon null 对照中，`lambda=0.001` 相对 `lambda=0` 没有稳定改善，长预测八个数据集-horizon 单元的变化仅 `-0.005%` 至 `+0.110%`；不能写成 AutoCon 增益。v3 backbone 本身在 H96 的 Electricity/Traffic 上相对同坐标最佳外部 baseline 分别改善约 `6.11%/11.52%`，但 ETTh1 仍高 `3.55%`。同日 v4 ACF memory screen 在 Traffic 退化约 `0.64%`，因此 ACF memory 暂不升级为主线。人工复核入口为 [`mamba_ts_v3_multihorizon_autocon_screen_20260922.md`](research/mamba_ts_v3_multihorizon_autocon_screen_20260922.md)。

同日新增的 prediction-connected AutoCon 融合 screen 完成 `24/24`：active `lambda=.003` 对同构 null 在 Electricity 改善 `-0.666% (3/3)`、Traffic `-0.055% (3/3)`，但 ETTh1/ETTm1 分别退化 `+0.229%/+0.081%`。这证明旧结构的“表示不参与预测”确实是断点，但新全局表示只形成高维数据候选信号，不能作普遍结论。通道保留版本因 Traffic 训练超过 7 分钟、patch 优化后仍为全局前向延迟的约两倍而停止；详见 [`mamba_ts_v3_repr_forecast_fusion_screen_20260922.md`](research/mamba_ts_v3_repr_forecast_fusion_screen_20260922.md)。

随后完成的 train-only MambaTS variable permutation screen 为 `36/36`（四数据集、H96、三 seeds）：在同一排列结构中，AutoCon active 相对 null 的验证 MSE 变化为 ETTh1 `+0.220%`、ETTm1 `-0.182%`、Electricity `-1.111%`、Traffic `-0.037%`。高维 Electricity 三个 seed 全部退化，所以此项只能作为负消融，不能作为鲁棒性、精度或速度卖点；完整人工复核记录见 [`mamba_ts_v3_variable_permutation_screen_20260922.md`](research/mamba_ts_v3_variable_permutation_screen_20260922.md)。

最新的 direct forecast ACF consistency clean rerun 完成 `24/24`：训练期用 train-only signed ACF 选择 lag，直接正则预测序列的时间相关性，不增加任何推理参数。ETTh1/ETTm1 为 `-0.031%/-0.082%`，Electricity/Traffic 为 `+0.051%/+0.029%`，没有跨数据集一致性，不能进入主模型。一次并发执行产生了重复 manifest 和覆盖哈希，已被严格 validator 拒绝并在新输出根单 runner 重跑；只有 clean rerun 可读，详见 [`mamba_ts_v3_forecast_acf_consistency_screen_20260922.md`](research/mamba_ts_v3_forecast_acf_consistency_screen_20260922.md)。

全局融合随后完成 `60/60` 个预注册 λ/warm-up 校准作业。`.0003/.001/.003/.003-warm4` 均未通过“全数据集无平均退化且至少三个数据集多数 seed 改善”的选择规则；warm-up 虽改善 ETT，却使 Traffic 退化。因此当前全局损失的单纯 λ 调参已停止，不能从 Electricity 单项结果推导论文贡献；完整审计见 [`mamba_ts_v3_global_repr_autocon_calibration_20260922.md`](research/mamba_ts_v3_global_repr_autocon_calibration_20260922.md)。

另有一轮独立的 48-job MambaACF 候选 screen 已完成：四个数据集、H96、三 seed，四个 phase validator 均通过，测试工件为 0。`MambaACFAutoCon(lambda=0.001)` 在 Electricity/Traffic 的 H96 低于同坐标外部基线最佳均值，但在 ETTh1/ETTm1 仍落后 PatchTST；相对同架构 `lambda=0` 的增量接近零。因此这轮只能作为候选筛选证据，不能升级为对比学习收益、全 horizon 优势或论文主结果。计划 immutable snapshot 仍为 `authorized=false`，也不能追溯性地写成预注册确认实验。人工复核入口为 [`mamba_acf_multidataset_candidate_screen_20260910.md`](research/mamba_acf_multidataset_candidate_screen_20260910.md)。

2026-09-10 已完成独立的六基线 full screen：四个数据集、H96/H336/H720、三个 seed，
共 `216/216` 个 validation-only 作业，phase validator 为 `valid=true` 且无失败记录。
它补齐了 NLinear 与多数据集同来源对照，但结果明确不是 plain Mamba 的普遍胜出：
ETTh1 由 PatchTST 主导，Electricity 由 TimesNet 主导，Mamba-2 只在部分高维坐标
优于线性基线。请以 [`mamba_multidataset_plain_baseline_full_screen_20260910.md`](research/mamba_multidataset_plain_baseline_full_screen_20260910.md)
复核完整表；该 screen 没有测试指标，不能作为投稿表格。

代码合同更新（2026-09-07）：普通 Mamba-1/Mamba-2 及其 AutoCon 变体现在真正响应 `mamba_causal_embedding=true`，显式开启时使用左填充 `CausalDataEmbedding`；默认关闭时仍走旧的 circular embedding。该修复已通过回归测试，但尚未重新训练，不能写成精度提升。

修正后的 ETT S/M 阶段和高维 `_v3` 工件均已完成并通过各自 validator，但高维 selection 是后验 replay（`posthoc_verified`，`publication_ready=false`），不能当作原始 launch-time 预注册确认。最新 current-tree variable-token fair screen 的 30/30 作业、五个 validator 和 30/30 sidecar replay 均通过；同架构 `lambda=.03` 相对 `lambda=0` 的 MSE 为 `-0.979%`（5/6），memory on/off 为 `-1.168%`（6/6），但仍只有 H96、每数据集 3 seed，属于探索性证据。fresh direct-anchor held-out replication 为 ETTh1 `-0.563%`（5/5）和 Traffic `-0.978%`（4/5），但后续 24-job anchor x memory 交互 screen 的方向在两数据集相反、pooled CI 跨零，故没有协同结论，也不得将组合升级到 test 选择；详见 [`interaction readout`](research/mamba_acf_anchor_memory_interaction_screen_readout_20260904.md) 与 [`erratum`](research/mamba_acf_anchor_memory_interaction_screen_erratum_20260907.md)。与独立 fair baseline 的描述性均值相比，variable calibrated 在 ETTh1 比 PatchTST 高 `1.188%`，在 Traffic 低 `0.829%`；shared fair calibrated 的 `+3.207%/+1.329%` 是另一组 provenance，不得混写。因此尚不能写成“精度不低于基线”。普通低秩 channel gate、ACF-conditioned gate、low-rank head、relation-coverage、raw target、direct-anchor 和 NLinear 缺口 screen 的既有边界仍然有效。以上结果不能写成最终论文结论。

本轮修复了 Mamba kernel 独立导入、`Decomp` 早期拒绝、relation sampler 与损失目标的一致性，以及非整除多尺度池化/时间 mark 对齐；未启动任何新训练。多尺度 mark 从块首改为块尾、部分 relation 配置从 lag-pair 改为 shuffle 均属于 source-contract 改动，因此已有 accuracy 工件只能按其原始 immutable snapshot 解读，不能写成当前源码重新训练的结果。

本轮新增的 `mamba_variable_repr_fusion` 是一个默认关闭的 AutoCon 候选：它把已有 variable-token 状态摘要注入 relation representation，并已登记独立的 12-job off/on 未授权计划；实现、资源 proxy、严格因果性限制和 paired-contract 校验均已落盘，但没有新精度结果，不能写成已改善 Traffic 或基线。

三阶段 live screen 的新版 data-bound `is_training=0` 回放已完成：[`mamba_acf_data_bound_replay_20260902_v2`](../artifacts/exploratory/mamba_acf_data_bound_replay_20260902_v2/) 为 `18/18` contract pass、`18/18` metric match。回放前冻结并核对了 phase provenance、immutable source snapshot、plan、实际数据集 SHA-256、source artifact hashes、job/config identity 和回放工具 SHA-256；这只证明历史 checkpoint 的评估加载与输入绑定，不是当前源码重训结果。用当前 validator 对旧 phase 做只读复核时，plain/memory 仍有 runner source-drift 错误，relation 的 6 个旧记录还缺少新 sampler requested/effective/reason、selected-lag support 和 lag-pair telemetry，因此不能进入严格 relation aggregate。

**本次代码审计的结果边界：** `corrected_*` 工件来自较早的 immutable snapshot；它们已经包含局部 logits 轴修正，但不包含当前的 legacy ACF 身份/表外 lag mask、AutoCon projection contract 和 CI 逐通道 global relation 修正。因此文件的 `450/450`、`270/270` 只证明旧协议的完整性与可复核性。plain forecasting 行的预测函数未因这些辅助目标修正而改变；`AutoConNetAutoCon`、`Mamba1AutoCon`、`Mamba2AutoCon` 以及任何 `AutoConCI` 行必须按当前代码重新做 selection 和 confirmatory training，不能把旧表的 AutoCon 增益直接写入新论文。

当前新增的 scale conditioning 与 CLAM 仍只有实现合同和 synthetic 资源证据，没有 current-tree 预测精度结果。请先看 [`research/mamba_acf_scale_conditioning_review_20260902.md`](research/mamba_acf_scale_conditioning_review_20260902.md)；prediction-only 资源读数显示 scale 开销约为 `+18.4%/+18.8%`，CLAM 约为 `+23.8%/+21.1%`，因此不能把它们描述为速度优化。当前源码与 DLinear、PatchTST、Mamba-1、Mamba-2 的资源基线见 [`research/mamba_acf_resource_baseline_current_20260902.md`](research/mamba_acf_resource_baseline_current_20260902.md)。CLAM 的短版语义和最小 screen 建议见 [`research/mamba_acf_clam_review_20260902.md`](research/mamba_acf_clam_review_20260902.md)。两项训练均未获授权，也没有新训练工件。
同一资源页还登记了单尺度 `[1]` 候选：在 ETTh1/Traffic 的严格匹配协议下比 `[1,2]` 分别快 `38.69%/39.19%`，但尚无精度结果，不能据此修改默认架构或宣称整体优于基线。

逐步可靠性候选已单独登记为 [`research/mamba_acf_step_reliability_screen_plan_20260903.json`](research/mamba_acf_step_reliability_screen_plan_20260903.json)，人工复核稿见 [`research/mamba_acf_step_reliability_screen_readout_20260903.md`](research/mamba_acf_step_reliability_screen_readout_20260903.md)。它只切换静态 raw ACF memory 的 reliability gate，保留同一 `MambaACFAutoCon`、global signed relation、数据划分、seed 和预算；12 个 job 已 validate-only 通过，`execution.authorized=false`，没有精度结果。训练日志还必须报告 `acf_effective_memory_to_forecast_rms`，否则不能判断 ACF residual 是否在预测尺度上真正起作用。

单尺度/多尺度精度筛选计划见 [`research/mamba_acf_single_vs_multiscale_screen_plan_20260902.json`](research/mamba_acf_single_vs_multiscale_screen_plan_20260902.json)：`12` 个 job 已通过 validate-only，但 `execution.authorized=false`，尚未启动训练。计划只比较 scales，固定数据哈希、seed、ACF memory、global relation、AutoCon 权重和训练预算。

历史 `strict` 结果属于数值稳定化修正前的 pilot，只能按其冻结状态解读，禁止与 `corrected` 结果合并统计。详见 [`reports/strict-evidence-status.md`](reports/strict-evidence-status.md)。

## 当前确认性执行锁

新的 MambaACF 确认性方案仍是 `draft_requires_prelaunch_review`，没有正式确认结果。runner 在当前计划上独立复算出 selection `144`、confirmation `740`、training `884` 个作业；缺少注册计数、计数被篡改、未授权状态、缺少 launch sidecar 或 sidecar 哈希/审批/门禁证据不一致，都会在真实启动前返回错误。当前对该草案执行非 `dry_run` 的启动尝试已按预期被拒绝，未创建训练工件。

统计 analyzer 还会要求所有 test-producing phase 齐全，重新核对注册计数、source/analyzer 快照、manifest retry 记录、selection 绑定，以及 MambaACF memory 的 train-peak/fixed-placebo lag 来源和 fallback 状态。固定 `[5,11,17]` placebo 的 availability profile 未匹配 peak arm，已从 primary statistical gate 移除，只能做描述性诊断；未来若恢复该归因，必须先审批 availability-matched sham。只有人工复核 1% 非劣门槛、Bonferroni/配对统计规则、容量公平性、先验 sidecar 和 launch sidecar 后，才可以把计划副本改为授权状态。

当前工作树需以本轮最终回归输出为准；测试数字不等于论文结果，也不替代确认性训练。当前完整回归为 `828 passed, 45 warnings`。另外，当前 `mamba_ssm=2.3.1` 环境下 Mamba-1/Mamba-2 及 MambaACF/MambaACF1 forecast-view 的 CUDA shape/finite/backward smoke 均通过，可用 [`smoke_mamba_acf_forecast_relation.py`](../scripts/smoke_mamba_acf_forecast_relation.py) 重跑；这只是实现级证据，没有产生精度结果。MambaTS v5 的 16-job signed-ACF-anchor screen 和六基线 24-job execution smoke 均已归档为 validation/执行证据，不能替代完整基线矩阵或测试集确认。警告主要来自旧数据加载器的 pandas API、小样本配对统计 precision-loss 提示和一个预期的 Mamba headdim 兼容性提示，不代表训练或统计通过。

确认计划的 accuracy claim contract 已明确：primary non-inferiority 只覆盖 PatchTST 与 Mamba2；DLinear、NLinear、TimesNet、iTransformer、Mamba1 必须保留为描述性基线，但当前统计门不支持“相对全部基线不劣”。若要升级为全基线 headline，必须新建并复核 multiplicity family，而不能仅追加结果行。

新增实验合同 guard：MambaACF AutoCon relation loss 在 `local/both` scope 下禁止 `mamba_token_stride>1`，因为表示插值后的 token 坐标不能直接解释为原始时间 lag；`global` scope 仍可用，单位 stride 不受影响。

`resource_profile()` 已把 raw ACF context 的 lag-index 因果性与完整表示网格的严格因果性拆开。池化/插值、双向扫描、非因果 embedding、窗口级 gate、当前全窗口 `LastVal/Mean/ReVIN/MeanStd` 归一化，或 residual/trend 居中变换时，不能把表示学习写成逐位置严格因果；该限制不改变已有预测数值路径。

## 本轮核心审计结论

- `AutoCon_lambda=0.03` 不能被当作跨数据集的通用默认值。现有 synthetic probe 在共享 trunk 上得到 `0.03 * ||g_aux|| / ||g_mse||` 约 `0.071--0.124`，梯度方向近乎正交；应先做 validation-only 的配对 lambda 校准。
- MambaTS v3 的新 lambda 校准已经完成：seed 2022/2023 两个冻结 validation-only 阶段共 64 个 coordinate-lambda rows，`lambda=0.001` 相对同结构零权重控制为 `11/16` 改善、平均 `-0.0095%`，AutoCon loss 非零；`lambda=0.03` 首轮平均 `+0.3142%`。这只能支持低权重的“近似不扰动预测”探索性表述，不能写成 AutoCon 增益或确认性非劣效；详见 [`research/mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md`](research/mamba_pems_ts_v3_autocon_lambda_calibration_20260910.md)。
- 固定 `lambda=0.001` 的 calibrated v3 AutoCon 与 TimesNet 精确配对复核为 16 个坐标中 `10/16` 改善；H=12 `8/8`、平均 `-19.50%`，H=96 `2/8`、平均 `+6.74%`。这支持短期竞争力，不支持“全跨度不低于基线”；PEMS03/04 H=96 的失败模式必须在论文中保留。详见 [`research/mamba_pems_ts_v3_autocon_strong_baseline_20260910.md`](research/mamba_pems_ts_v3_autocon_strong_baseline_20260910.md)。
- 默认 `mamba_acf_repr_fusion=false` 时，relation loss 作用于表示分支，不能写成直接监督 ACF forecast memory 或 gate。representation fusion 若启用，必须作为独立消融归因。
- 当前 `sign_hinge` 只占 auxiliary 数值约 `0.04%`；`sign_accuracy` 约 `91%` 不足以证明强 signed supervision。不要在没有负相位/合成控制的情况下放大 `sign_weight`。
- 普通 `DataEmbedding` 使用 circular token convolution；严格左因果的全模型表述至少需要 `mamba_causal_embedding=true`，并同时满足单位 token stride、单尺度、非双向扫描、无窗口级 gate/非 raw 居中变换，以及 prefix-causal normalization。当前 `LastVal/Mean/ReVIN/MeanStd` 不满足最后一项。`wo_freq` 的输入原地污染已修复，并有直接输入和多尺度 view 回归测试。

- 当前新增显式 `MeanStd` 模式，使用 `sqrt(population_var + 1e-5)`，与 PatchTST forecast 的窗口标准化逐元素对齐；旧归一化模式没有改写。对应的 [`mamba_normalization_parity_screen_plan_20260903.json`](research/mamba_normalization_parity_screen_plan_20260903.json) 登记了 ETTh1/Traffic、H96、三 seed、Mamba2/MambaACF/AutoCon 配对，共 36 个未授权验证作业。它是公平性控制，不是已完成的精度结果。
- double `F.normalize` 已验证近似幂等（synthetic loss 差 `<=3.6e-7`、梯度相对差 `<=2.4e-7`），当前保留以维护 loss API 与历史 trainer 语义；这不是待宣称的改进点。
- relation loss 的 global/local 幅值 CE 与已加权 signed-hinge contribution 已写入 `loss_logs.json`；后者包含 `sign_weight`，四个字段可按 scope 重构 relation 均值，旧日志缺失时保持 unavailable。

详细推导见 [`mamba_acf_loss_scale_audit_20260902.md`](research/mamba_acf_loss_scale_audit_20260902.md)；未授权的 validation-only 校准登记见 [`mamba_acf_lambda_calibration_screen_plan_20260902.json`](research/mamba_acf_lambda_calibration_screen_plan_20260902.json)。

## Traffic 高维逐通道审计

后验 `_v3` 的 Traffic 862 通道描述性汇总显示，Mamba2 相对 PatchTST 的 scaled-MSE 差为 H96 `+29.205%`（8/862 通道更好）、H336 `+20.611%`（8/862）、H720 `+19.833%`（7/862）；trimmed 与 leave-one-out 检查仍保持同一方向。该结果不能把通道当作独立重复，也不能绕过 `posthoc_verified`/`publication_ready=false` 边界，但它支持一个结构诊断：Traffic 的主要风险是共享低维状态对异质通道的压缩，单独加入 AutoCon 没有修复这一点。逐通道/分组 lag bank 或 channel-preserving 分支必须与同预算基线配对验证。

## 反事实依赖审计

[`research/mamba_acf_counterfactual_audit_20260903.md`](research/mamba_acf_counterfactual_audit_20260903.md) 对 ETTh1/Traffic 的 6 个已有 checkpoint 做了只读 memory on/off 重放。相同 checkpoint 下，ETTh1 的均值 MSE 为 `0.393167`（on）与 `0.457892`（off），Traffic 为 `0.564118` 与 `0.624881`；所有 seed 的 off 都更差，说明该已训练预测函数确实使用了 ACF residual branch。该表的 estimand 是“固定权重函数对分支屏蔽的敏感性”，不是重新训练两臂的处理效应；窗口和 origin 也不是独立重复。此前无法与本 phase source/config/loader 同时对齐的 ETTh1 约 `0.444` 即时值不进入主表。

## 本轮新增的未运行项

- `mamba_acf_anchor_copy_screen_plan_20260902.json` 已登记 12 个配对作业（ETTh1/Traffic、H96、3 seeds、8 epochs）。两臂固定同一 variable-token MambaACF AutoCon、raw train-only ACF sidecar、hidden gate 和 signed global relation；唯一变化是独立的 per-channel 最大 `|ACF|` 合法滞后 signed direct-anchor residual。启用臂只有 1 个 ReZero 标量，采用严格 `L-1+h-lag` 索引和 invalid mask；计划未授权，尚无精度结果。
- [`analyze_mamba_acf_anchor_copy_screen.py`](../scripts/analyze_mamba_acf_anchor_copy_screen.py) 在缺少 phase validation/manifest 时返回 `NOT_RUN`，不创建 readout。计划 validate-only 已展开为 `12` jobs。
- direct-anchor 资源 smoke 已单独保存到 [`mamba_acf_anchor_copy_resource_smoke_20260902.md`](research/mamba_acf_anchor_copy_resource_smoke_20260902.md) 及其原始 JSON/CSV 目录；当前源码指纹下的 50/500 测量，ETTh1 median 开销约 `+4.04%`，Traffic 约 `+4.35%`，并分别增加 `0.0160/1.9730 MB` static buffer，记录含 model/tool SHA-256。它只支持结构资源复核，不支持精度、端到端速度或论文主张。
- NLinear 缺口另有 [`mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json`](research/mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json) 的 6-job 未授权登记。历史 NLinear 记录预算和 provenance 不匹配当前短 screen，不能直接填入现有基线表。
- 共同来源的高维 plain baseline 扩展计划 [`research/extension_baseline_plan.json`](research/extension_baseline_plan.json) 已 validate-only 展开为 `180` 个 job，但仍是 `0/180` 训练结果；基于历史 manifest 的串行耗时粗估约 `13.9` 小时，详见 [`research/experiment_runtime_estimate_20260903.md`](research/experiment_runtime_estimate_20260903.md)。该估计不是承诺的 wall-clock，也不构成授权。
- 新的 forecast-level causal lag-bank 修订计划 [`research/mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json`](research/mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json) 已通过 `validate_only`（12 jobs），但仍未授权、未训练；它明确只检验 forecast-memory support，不把 `LastVal` 下的 AutoCon representation 写成 prefix-causal。该计划关闭 `stream_metrics`，主指标读取 checkpoint `loss_logs.json:val_loss`，避免把跳过 test 的训练流程与 test-only stream 工件混用。Traffic `C=862` 的 50/200 forward 资源 smoke 见 [`mamba_acf_channel_lag_bank_resource_current_20260903`](../artifacts/exploratory/mamba_acf_channel_lag_bank_resource_current_20260903/)，只支持结构开销判断。

## 已冻结的核心阶段

| 阶段 | 范围 | 状态 | 人工复核入口 |
|---|---|---|---|
| `corrected_selection_S` | ETT，S，630 jobs | 已完成 | `artifacts/paper_final/corrected/selection_S/` |
| `corrected_selection_M` | ETT，M，630 jobs | 已完成 | `artifacts/paper_final/corrected/selection_M/` |
| `corrected_confirmatory_S` | ETT，S，450 jobs | 已验证 | `artifacts/paper_final/corrected/confirmatory_S/` |
| `corrected_confirmatory_M` | ETT，M，450 jobs | 已验证 | `artifacts/paper_final/corrected/confirmatory_M/` |
| `highdim_selection_M` | Electricity/Traffic，378 jobs | 已验证；selection sidecar 为后验 attestation | `artifacts/paper_final/corrected/highdim_selection_M_v3/` |
| `highdim_confirmatory_M` | Electricity/Traffic，270 jobs | 已验证；完整后验 replay，非 publication-ready 正式确认 | `artifacts/paper_final/corrected/highdim_confirmatory_M_v3/` |

共同估计量是同一 dataset、horizon 与随机 seed 下的配对差：

```text
delta = MSE(Mamba + AutoCon) - MSE(Mamba)
```

`delta < 0` 表示该 seed 的误差降低。推断重复单位是 seed；窗口和通道不是独立样本。

## M 阶段复核结果

- [`phase_validation.json`](../artifacts/paper_final/corrected/confirmatory_M/phase_validation.json)：`valid=true`，`expected_jobs=450`，`manifest_unique_jobs=450`，`status_counts.ok=450`，无 malformed manifest、record error 或 validator error。
- [`provenance.json`](../artifacts/paper_final/corrected/confirmatory_M/provenance.json)：`provenance_id=8bb6120ba4d85497`；包含 source snapshot、数据哈希、计划副本和 selection binding。绑定哈希为 `615b31637653cd0eb0ad7037a5adfb75071e0b5503e388b2ca063e45de5799d1`。
- [`paper_statistics.json`](../artifacts/paper_final/corrected/confirmatory_M/statistics/paper_statistics.json)：450 个已验证配置/结果、10 个 MSE 配对单元；使用 seed-level 双侧 paired Student-t，按 metric family 做 Holm-Bonferroni 与 BH-FDR。
- 描述性 MSE：`Mamba1AutoCon - Mamba1` 在 8/10 单元改善，平均相对变化 `-0.77%`，Holm/FDR 显著单元均为 0；`Mamba2AutoCon - Mamba2` 在 7/10 单元改善，平均相对变化 `-2.38%`，Holm 与 FDR 各有 1/10 单元通过。完整表和原始配对差在 [`report/`](../artifacts/paper_final/corrected/confirmatory_M/report/) 与 [`statistics/`](../artifacts/paper_final/corrected/confirmatory_M/statistics/)。

**必须保留的选择限制：** M selection sidecar 的模式是 `posthoc_verified`，且 `publication_ready=false`。确认性 M 的 provenance 记录的是 `prelaunch_replay_of_posthoc_attestation`；它证明了绑定/replay 一致性，但不能追溯性地把 selection 说成原始 launch-time 的预注册 gate。

S 阶段同样应只作描述性解读：Mamba-2 + AutoCon 在 8/10 个 MSE 单元方向改善（平均约 `-5.00%`），Mamba-1 + AutoCon 在 4/10 个单元改善（平均约 `+1.24%`）；Holm/FDR 后没有显著单元。请以其冻结统计文件为准。

## 高维 v3 复核结果

- [`highdim_selection_M_v3/phase_validation.json`](../artifacts/paper_final/corrected/highdim_selection_M_v3/phase_validation.json)：`valid=true`、`expected_jobs=378`、`manifest_unique_jobs=378`、`status_counts.ok=378`。
- [`highdim_confirmatory_M_v3/phase_validation.json`](../artifacts/paper_final/corrected/highdim_confirmatory_M_v3/phase_validation.json)：`valid=true`、`expected_jobs=270`、`manifest_unique_jobs=270`、`status_counts.ok=270`，无 malformed manifest、record error 或 validator error。原始 manifest 有 280 条尝试，其中 10 条路径名过长失败记录均被同一 job ID 的后续成功记录覆盖；推断输入是最终去重后的 270 个作业。
- [`selected_hyperparameters.provenance.json`](../artifacts/paper_final/corrected/highdim_selection_M_v3/selected_hyperparameters.provenance.json) 明确为 `posthoc_verified`、`publication_ready=false`；confirmatory provenance 的 binding mode 为 `prelaunch_replay_of_posthoc_attestation`。高维表因此只能作为完整后验复核/冻结读数，而非原始 launch-time 预注册确认。
- 结果和 seed-level 统计见 [`highdim-confirmatory-M-v3.md`](reports/highdim-confirmatory-M-v3.md) 与 [`paper_statistics.md`](../artifacts/paper_final/corrected/highdim_confirmatory_M_v3/statistics/paper_statistics.md)。

## 当前 MambaACF v3 机制 screen

[`mamba_acf_v3_relation_screen_readout.md`](research/mamba_acf_v3_screen_readout.md) 是在各自 immutable source snapshot 下可复核的短预算读数：五个 phase 共 `30/30` 个 successful run，五个 `phase_validation.json` 均为 `valid=true`。每个 dataset/arm 只有 3 个 seed、H96、8 epochs；它是机制筛选，不是架构或超参数选择，更不是论文确认性结果。

旧的三套 MambaACF exploratory screen 都是在 sidecar contract 引入前的 immutable snapshot 上执行的；对应旧 screen root 均没有 `acf_prior.npz`，run config 也没有 `mamba_acf_prior_artifact`/SHA 绑定。该限制按 arm 解释：variable-mixer screen 的 plain/variable 两臂均关闭 memory/relation，缺 sidecar 不改变其 forecast path；旧 v3 中的 available-memory、hard/soft relation 及 memory+relation 臂，以及旧 variable-memory screen 的 memory 和 memory+relation 臂，均在旧代码中从 train split 即时拟合 prior，不能仅凭 checkpoint+config 由当前 runner 独立 replay。它们只能在原始 train split 与匹配 source snapshot 下重算；这种操作应标为 historical re-run，不应称为 sidecar-backed standalone replay。新的 sidecar 持久化合同 readout 已单独列出，不能与旧 screen 数值混合。

| arm | ETTh1 MSE mean +/- SD | Traffic MSE mean +/- SD |
|---|---:|---:|
| plain MambaACF | 0.389664 +/- 0.002807 | 0.570445 +/- 0.003733 |
| train-only available ACF memory | 0.389005 +/- 0.001678 | 0.555458 +/- 0.005670 |
| hard non-overlap relation | 0.391957 +/- 0.002591 | 0.565820 +/- 0.008408 |
| soft non-overlap relation | 0.392103 +/- 0.001567 | 0.568679 +/- 0.004372 |
| memory + soft relation | 0.395972 +/- 0.005431 | 0.551952 +/- 0.004095 |

可用 ACF memory 相对 plain 的 MSE delta 为 ETTh1 `-0.000659`（2/3 seed）和 Traffic `-0.014987`（3/3 seed，约 `-2.63%`），因此它是下一轮确认实验的候选。soft 相对 hard 没有稳定增益（ETTh1 `+0.04%`、Traffic `+0.51%`）；memory+soft 相对 memory 在 ETTh1 变差约 `+1.79%`，Traffic 改善约 `-0.63%`。所以 signed relation 仍必须保留为必要的表示学习消融臂，但不能被宣称为通用精度提升。报告中的 pooled `ALL` 行不是六个独立重复，p 值仅为描述性筛选信息。

## 已完成的资源与数据 gate

- [`resource_benchmark/`](../artifacts/paper_final/corrected/resource_benchmark/) 已完成针对已验证 M 阶段的资源采集：45 个去重 architecture/resource record、90 个训练耗时分组，固定 `batch=32`、25 次 warm-up、100 次计时；环境为 RTX 5090 D、PyTorch `2.8.0+cu128`、CUDA 12.8。它是资源负载测量，不是预测效果证据，也不能单独证明容量公平。
- 当前源码的统一资源复测见 [`resource-v3-current-traffic-report.md`](research/resource-v3-current-traffic-report.md)：Traffic/C862/H96/B8、50 次 warm-up/200 次计时下，MambaACF plain 中位延迟 `0.684640 ms`，PatchTST `0.885824 ms`，但 memory 分支为 `0.808272 ms`。这仍只反映 device-resident synthetic forward，不支持端到端训练速度结论。
- 当前可复现实验环境为 Python `3.9.25`、`mamba-ssm 2.3.1`、`causal-conv1d 1.6.2.post1`、Triton `3.4.0`；Mamba-1/Mamba-2 的 `features=MS` GPU 资源合同 smoke 均为 `status=ok`，工件见 [`ms_contract_smoke/`](../artifacts/paper_final/corrected/resource_benchmark/ms_contract_smoke/)。
- [`highdim_data_quality/`](../artifacts/paper_final/corrected/highdim_data_quality/) 已通过 Electricity（26,304 行、321 通道）和 Traffic（17,544 行、862 通道）的本地质量 gate：哈希、小时 cadence、重复时间戳、有限值和常量列检查均无 error。数据合同与后续边界见 [`research/highdim-extension-protocol.md`](research/highdim-extension-protocol.md)。
- [变量 token screen readout](research/mamba_acf_variable_mixer_screen_readout.md) 已完成并通过独立 readout：ETTh1/Traffic 各 3 个 seed，MSE 分别为 `-0.93%` 和 `-2.49%`（variable minus plain）；训练耗时增加约 `31.1%`/`14.1%`。这是 cross-variable branch 的 exploratory 结果，两个臂都关闭 ACF memory/relation，不能写成 ACF 增益。
- [变量 token + memory + relation screen readout](research/mamba_acf_variable_memory_relation_screen_readout.md) 已完成 `18/18` 个作业并通过三个 phase validator。相对 variable plain，memory 的 MSE 变化为 ETTh1 `-0.045%`（1/3 seed）和 Traffic `-2.282%`（3/3）；在 variable+memory 条件下加入 hard signed relation 的增量为 ETTh1 `-0.802%`、Traffic `-0.655%`（均 3/3）。这些是 H96、8 epoch、每数据集 n=3 的机制筛选读数，不是确认性效应；relation 仅使用 train-only effective lags `[96,120]`，objective v3 coverage 为非零。这里所指的是 sidecar 合同之前的旧 screen，其 memory/relation 的 12 个 checkpoint 缺少 prior sidecar，不能与后续 sidecar 持久化 screen 混写。
- [sidecar 持久化合同 readout](research/mamba_acf_sidecar_replay_readout.md) 已完成 `18/18` 个作业；三个 phase 的 exploratory validator 均为 `valid=true`，当前严格 artifact 审计也通过 `18/18`，且每个 job 均有完整 `acf_prior.npz` 与 SHA 绑定。raw ACF 下，memory 相对 variable plain 的 MSE 变化为 ETTh1 `-0.567%`、Traffic `-1.618%`（各 3/3）；relation 在 memory 之上的增量为 ETTh1 `-0.529%`、Traffic `-0.550%`（各 3/3）。这些是 provenance snapshot 训练过程中拟合 prior 后保存 sidecar 的探索性指标，不是 current-tree 训练结果；仍只有 H96、n=3，且 global relation coverage 约 3%，不能写成确认性增益，也不能与旧 screen 数字合并。
- [三臂 channel-gate screen](research/mamba_acf_channel_gate_screen_readout_20260901.md) 已完成 `18/18` 个作业并通过三个 validator。ordinary low-rank gate 相对 no-gate 的总体 MSE 为 `-0.452%`，其中 Traffic 为 `-0.752%`（3/3），ETTh1 近零；ACF-conditioned gate 相对 ordinary gate 为 `+0.347%`，Traffic 为 `+0.720%`。checkpoint 尺度显示两种 gate 都实际打开，因此当前不支持把 ACF 条件化继续叠加。它仍是 H96、n=3/dataset 的 exploratory evidence。
- [low-rank head screen](research/mamba_acf_low_rank_head_screen_readout_20260901.md) 已完成 `12/12` 作业、两个 validator 均为 `valid=true`。rank-16 使预测头参数从 `18,624` 降到 `6,336`，但 MSE 平均增加 `1.095%`（`0/6` 配对点更低），elapsed 仅减少 `0.977%` 且方向混合；实际 early-stop 总轮数为 full `41`、rank-16 `39`，因此不能将 elapsed 差异解释成纯架构速度。该读数来自 provenance snapshot，且没有 standalone replay 工件。
- [relation-coverage screen](research/mamba_acf_relation_coverage_screen_readout_20260901.md) 已完成 `18/18` 作业、三个 validator 均为 `valid=true`。balanced pairing 将 `[96,120]` 的 exposure span 压到 `0--0.135%`，但 MSE/MAE 分别增加 `0.239%`/`0.249%`；active-anchor normalization 的 MSE/MAE 变化仅 `+0.013%`/`+0.018%`。reference/active_norm 的 ETTh1 实际轮数为 `7/5/5`，balanced 为 `5/6/6`，Traffic 各为 8；elapsed 仍只是短预算训练耗时。该 screen 只定位 coverage 机制，不支持精度或速度主张。
- [current-tree raw relation-target screen](research/mamba_acf_relation_target_screen_readout_20260901.md) 完成 `12/12` 作业，两个 phase validator 均为 `valid=true`，并完成 `12/12` standalone sidecar replay。soft05 相对 hard 的 pooled MSE 为 `+0.146%`（`2/6` seed 更低）；ETTh1 为 `-0.045%`（`2/3`），Traffic 为 `+0.279%`（`0/3`）。soft05 将 positive/anchor 从 `0.923` 增至 `1.470`，独立 high-ACF 排除率从 `78.47%` 降至 `66.62%`，但没有精度收益，已触发停止规则；不再扩展 soft 阈值、memory 或更长 horizon。该 screen 只检验 target-support 语义，不能写成 false-negative 修复或 ACF 确认性收益。
- [current-tree variable-token fair screen](research/mamba_acf_variable_token_fair_screen_readout_20260901.md) 完成 `30/30` 作业，五个 phase validator 均为 `valid=true`，独立 replay 为 `30/30` contract/metric pass、drift `0`。同一 `MambaACFVariableAutoCon` 的 `lambda=.03 - lambda=0` MSE 为 `-0.979%`（5/6，ETTh1 2/3、Traffic 3/3）；memory on/off 为 `-1.168%`（6/6），rank-2 calibration 为 `-0.025%`（3/6）。它仍是 H96、每数据集 3 seed 的 exploratory screen；candidate 在 ETTh1 不及 PatchTST，不能替代确认性矩阵或速度结论。
- [channel-order/conditioning screen](research/mamba_acf_channel_order_conditioning_screen_readout_20260902.md) 完成 `18/18` 作业和 replay；strength order pooled MSE `+0.113%`、ACF conditioning `+0.598%`，均未达到扩展门槛。其 immutable snapshot 与当前工作树有四个 source 文件漂移，不能称为 current-tree 重训。
- 当前树新增 `mamba_acf_channel_order=similarity`：基于 train-only signed selected-lag profile 的无矩阵 cosine-seriation；已完成模型/CLI/runner/资源/回归接入，但尚无训练精度结果，且只应作为 order ablation。对应语义与风险见 [`acf-channel-ordering.md`](research/acf-channel-ordering.md)。
- 新增 [`mamba_multidataset_plain_baseline_expansion_plan_20260903.json`](research/mamba_multidataset_plain_baseline_expansion_plan_20260903.json)：ETTh1/ETTm1/Electricity/Traffic × DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2 × H96/336/720 × 3 seeds，共 `216` 个 validation-only job；`execution.authorized=false`，没有新的结果。
- [memory-gate mode screen plan/review](research/mamba_acf_memory_gate_mode_screen_review_20260902.md) 仍是未授权的 12-job 草案；`hidden`/`variable_token` 目前只有 synthetic resource 对照，没有预测精度结果，不能提前启动或写入论文主表。
- 其只读分析器 [`analyze_mamba_acf_memory_gate_mode_screen.py`](../scripts/analyze_mamba_acf_memory_gate_mode_screen.py) 在缺少完整训练工件时返回 `NOT_RUN`，不会写出伪造的 readout；当前状态正是如此。
- [channel-gate replay](../artifacts/exploratory/mamba_acf_channel_gate_replay_20260901/replay_report.md) 对 18 个 checkpoint/sidecar 做了隔离 `is_training=0` 回放，`18/18` contract 和 metric 均通过；[resource directory](../artifacts/exploratory/mamba_acf_channel_gate_resources_20260901/) 记录了参数量与 synthetic latency，不能替代端到端速度。
- sidecar readout 的 source drift 和证据边界详见 [`mamba_acf_sidecar_replay_audit.md`](research/mamba_acf_sidecar_replay_audit.md)：18 个指标绑定旧 provenance snapshot，当前严格审计只验证 sidecar 合同，不把旧快照结果升级为 current-tree 训练或确认性结果。
- 已在隔离输出根完成 18-job `is_training=0` standalone sidecar replay（详见 [`replay_report.md`](../artifacts/exploratory/mamba_acf_sidecar_standalone_replay_20260831_audit/replay_report.md)；同目录含启动前冻结的 `replay_manifest.json` 和逐 job `replay_results.jsonl`）：`18/18` 合同通过、`18/18` 指标在冻结容差内匹配、sidecar hash `18/18` 一致，MSE 最大绝对差 `3.3e-9`、MAE 最大绝对差 `2.2e-9`。这证明当前评估加载路径可重现历史 checkpoint，不代表 current-tree 重训或确认性效果。
- 训练/验证边界回归见 [`tests/test_split_and_prior_leakage.py`](../tests/test_split_and_prior_leakage.py)：修改验证/测试行不会改变 train-only prior，且所有 forecast label 均留在对应 split 内；确认计划已固定 raw ACF，以避免现有 global residual/trend prior 与 per-window memory 分解不一致。该选择不消除 circular embedding 等其他方法风险。
- 高维合同 gate [`validate_highdim_contract.py`](../scripts/validate_highdim_contract.py) 会固定质量报告的稳定内容哈希（不含生成时间/本机路径），并已在当前计划上以 strict 模式通过；后续训练计划遇到合同缺失、CSV 替换或质量内容变更会在启动前停止。
- 历史冻结 M 阶段曾记录 `134 passed`；那是旧测试集合的回归数字，不代表当前工作树。当前回归包含变量 alias、benchmark alias、split/prior leakage、memory-lag contract、strict sidecar shape/lag checks、launch authorization、channel-gate、low-rank head、relation-target、relation-coverage 和 confirmation-statistics checks；最终通过数以本轮完整 pytest 输出为准。冻结 M 阶段用兼容默认值重检后没有 record/artifact/config 错误，仅保留其 immutable source snapshot 与当前 validator/runner 的预期差异。
- 当前回归包含 channel resource profile、stepwise ACF reliability、静态 ACF memory cache、AutoCon local relation geometry cache、低秩 ACF channel gate、optimizer active-parameter contract、CUDA 输出合同以及 v3 relation objective contract。MambaACF shared/plain 配置中未参与 forward 的 `channel_pool` 与关闭 memory 时的 `acf_gate` 已冻结，因此旧 resource artifact 的 trainable 参数列只能作为历史快照，正式报告须重跑并区分 active trainable。AutoCon local cache 只复用无梯度的距离/ACF 查表，global origin relation 仍按 batch 重算。
- AutoCon 的三条 forecasting head 路径（AutoConNet、AutoConCI、Mamba2AutoCon）现在对偶数 moving-average kernel 使用非对称端点补齐，保证多尺度分解不再静默缩短一个时间步；奇数 kernel 的既有数值保持不变。该修复是边界合同，不构成新的精度结果。
- 新增 `mamba_acf_memory_gate_mode` 的默认值 `hidden` 保持旧路径；`variable_token` 仅完成结构/资源检查，尚未有预测精度结果。canonical synthetic forward 的 Traffic median 为 `1.5957 -> 1.6704 ms`（约 `+4.68%`），ETTh1 为 `1.4372 -> 1.4395 ms`（约 `+0.16%`）；token gate 的输出和 MAC 随通道数放大，因此不能先写成速度优化。详见 [`mamba_acf_memory_gate_mode_resources_20260902.md`](research/mamba_acf_memory_gate_mode_resources_20260902.md)。
- 真实 GPU smoke 已验证 hidden/variable-token 两种 gate 的 forward、backward、输出形状和有限梯度；这是实现合同检查，不是数据集训练或精度证据。variable-token gate 取 variable branch 表示，在 ReZero 初始点会改变该分支的梯度路径，后续比较必须按功能容量/优化路径消融解释。
- 高维 synthetic resource benchmark 仅说明当前实现能够承载相应通道规模；它不构成真实 Electricity/Traffic 的预测证据。
- 四数据集确认性方案见 [`research/mamba_acf_confirmation_protocol.md`](research/mamba_acf_confirmation_protocol.md)；JSON 仍为 `draft_requires_prelaunch_review`，884 个作业只完成 validate-only 展开，未授权启动。变量 token 组合、low-rank head 和 relation-coverage screen 已执行完成；它们均为 H96、每数据集 3 seed 的探索性工件，不能替代确认性矩阵。
- 预测路径 relation view 已实现，计划见 [`mamba_acf_forecast_relation_view_screen_plan_20260903.json`](research/mamba_acf_forecast_relation_view_screen_plan_20260903.json)。prediction-only 结构增量为零，但 global loss 只读预测窗口的跨通道均值；该 24-job 四臂计划未授权、未训练，零权重臂也不是吞吐对照。
- AutoConCI 的时间标记对齐回归见 [`tests/test_autocon_ci_contract.py`](../tests/test_autocon_ci_contract.py)；修复前的 `B>1,C>1` 结果应视为待复核。
- 真实数据的一 epoch GPU smoke 已完成（321/862 通道，流式 validator `errors=[]`），仅证明数据加载、CUDA、ACF 初始化和工件链路可运行；详情见 [`highdim_smoke/README.md`](../artifacts/paper_final/corrected/highdim_smoke/README.md)，不得当作效果结果。

## 尚未完成，不能越界

- 高维 `_v3` 的 selection/confirmatory 结果已经产生并通过 validator，但其 selection sidecar 是 `posthoc_verified` 且 `publication_ready=false`；因此可用于完整的后验复核和描述性读数，不能追溯性地写成 launch-time 预注册确认。
- 高维 selection 的旧预检目录 [`highdim_selection_M/`](../artifacts/paper_final/corrected/highdim_selection_M/) 与 `_v2` 目录 [`highdim_selection_M_v2/`](../artifacts/paper_final/corrected/highdim_selection_M_v2/) 仍是诊断工件（分别为 4/378 和 1/378），不应与已完成的 `_v3` 结果混用。
- 仍缺随机/打乱 ACF、residual 与 raw、局部与全局等机制消融。
- 仍需将参数量、FLOPs、显存、吞吐和训练时间用于容量公平性解释；不能把 backbone 差异自动归因于 AutoCon。
- 历史 pilot 的 plain/AutoCon `weight_decay` 和 scales 并不总是一致；同时默认 circular embedding、固定 `seq_len`、prior/窗口分解边界和 relation loss 的 local/global/per-lag coverage 仍需在正式协议中单独核对。
- `repr_head` 现在只在 `AutoCon=True` 时是可训练辅助投影；普通预测模式保留兼容 state-dict 但冻结该层。修复后的 AutoCon 结果必须建立新的 provenance，不与旧投影未使用的结果混表。
- Weather 有时间戳异常，未纳入高维扩展；不应以它代替已注册的 Electricity/Traffic。

## 最小人工核对顺序

1. 打开 M 的 validator、provenance 和统计 JSON，确认 ID、哈希、450/450 覆盖与 sidecar 限制一致。
2. 打开 S/M 的 `report/` 和 `statistics/`，核对 seed 是唯一推断单位，表格没有混入 `strict` 输出。
3. 打开高维 `_v3` 的两个 validator、selection sidecar 和统计文件，确认 378/378、270/270 与后验 replay 限制没有被混淆。
4. 打开 MambaACF v3、low-rank head 和 relation-coverage readout，确认每个 dataset 仅 `n=3`，以及 memory、relation、压缩和采样结论没有被混合或提升为确认性证据。
5. 打开变量 token + memory + relation 的 18-run readout，再打开 channel-gate 三臂 readout，核对增量对比是配对 screen，且 ACF-conditioned gate 的负结果没有被隐藏。
6. 打开 gate replay、low-rank head contract 与资源工件，确认 replay/参数合同不被写成端到端速度。
7. 在独立的 launch-time 预注册高维协议、机制消融和容量公平性分析完成前，不提升论文主张；`_v3` 的 270/270 后验 replay 不替代这一要求。

更短的状态摘要见 [`reports/corrected-evidence-status.md`](reports/corrected-evidence-status.md)。训练计划、统计方法和可复现性说明分别见 [`research/corrected_training_plan.json`](research/corrected_training_plan.json)、[`research/paper-statistics.md`](research/paper-statistics.md) 与 [`research/provenance-and-reproducibility.md`](research/provenance-and-reproducibility.md)。

Mamba/AutoCon/ACF 的逐层代码复核见 [`research/autocon_mamba_acf_core_review.md`](research/autocon_mamba_acf_core_review.md)；旧 MambaACF 组合仍是单 seed exploratory pilot，新 `train_peaks_available` 复核虽有 3 个 seed，也只覆盖 Traffic/H96/3 epochs；stepwise gate 的重跑同样只是单 seed 实现验证，不应替代冻结确认性工件。

## 本轮人工复核新增项

- `is_training=0` 的 plain `MambaACF` ACF-off 回放现在跳过无用的 train-split prior 拟合；即使残留或显式 sidecar 路径存在也不会加载/重生成，并保留已有 sidecar/resolution provenance；AutoCon 已挂载的 sidecar 则只做 hash/state metadata 绑定而不重复 loader；`--itr` 复用的 argparse namespace 也会清理上一副本的派生 lag 字段。回放合同还要求 required 分支加载 sidecar、任何分支都不重新拟合 prior，并比较规范化 runtime/config identity。对应回归覆盖已纳入当前 `601 passed`。
- 历史 replay JSON 中 ACF-off 记录的 `sidecar_load_message=true` 反映旧 loader 行为；新的回放合同按 live ACF 分支条件化加载要求，不把该历史日志当作当前实现的必要条件。
- 当前代码的三个单 job sidecar identity smoke（[`ACF-off`](../artifacts/exploratory/mamba_acf_replay_identity_smoke_off_current_20260903/replay_report.md)、[`plain required`](../artifacts/exploratory/mamba_acf_replay_identity_smoke_on_current_20260903/replay_report.md)、[`AutoCon required`](../artifacts/exploratory/mamba_acf_autocon_replay_identity_smoke_current_20260903/replay_report.md)）均为 `contract=pass`、`metric=match`；off 无 load/fit，两个 required 分支有 sidecar load 且无 fit。它们是评估合同回归，不是新的训练或精度样本；旧目录仍作为历史哈希报告保留。
- 阅读 [`mamba_acf_reliability_interpolation_screen_plan_20260902.json`](research/mamba_acf_reliability_interpolation_screen_plan_20260902.json)，确认 off/on 两臂除 `mamba_acf_reliability_interpolation` 外完全一致，且 `execution.authorized=false`。
- 可靠性插值的 canonical GPU 资源配对已记录在 [`mamba_acf_resource_baseline_current_20260902.md`](research/mamba_acf_resource_baseline_current_20260902.md)：ETTh1/M/H96、B=32、50 次 warm-up/200 次计时下 median 为 `1.399312 -> 1.506592 ms`（约 `+7.67%`），只增加 1 个参数；这是单次 device-resident forward 负载测量，没有精度或端到端训练结论。
- 在 [`models/MambaACF.py`](../models/MambaACF.py) 核对候选只使用 detached static raw prior 的 `step_reliability`/`aggregate_reliability`，零初始化时保持嵌套，未引入当前窗口或目标统计。
- 核对新资源 signature 不与旧 signature 混合；目前只有结构/资源契约和回归测试，没有该候选的精度结果。
- 当前源码的 data-dependent channel mixer spot check 已单独保存为 [`mamba_data_dependent_channel_mixer_resource_current_b8_20260903`](../artifacts/exploratory/mamba_data_dependent_channel_mixer_resource_current_b8_20260903/README.md)：ETTh1/Traffic 均为 `B=8`，四条记录是 plain `MambaACF`、`AutoCon=false`、`mamba_use_acf_memory=false` 的 rank-4 on/off isolation；median 比分别为 `1.0671x`/`1.0811x`，p95 比为 `1.0724x`/`1.0545x`。其余显式 CLI defaults 也不等于预注册候选，故这是每配置一次构造的 device-resident synthetic forward 测量，不能替代 Mamba1/Mamba2/PatchTST/DLinear 的统一训练基线，也不能写成端到端速度优势；旧版工件绑定不同源码/协议，未混合统计。
