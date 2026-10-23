# MambaACF 确认性实验协议草案

**状态：** `draft_requires_prelaunch_review`。这是待人工复核的 launch-time 方案草案，不是对已经完成实验的追溯性预注册；本次只生成计划和验证它的 job expansion，不启动训练。

**计划文件：** [`mamba_acf_confirmation_plan.json`](./mamba_acf_confirmation_plan.json)

## 1. 研究问题和可声称范围

问题限定为：在 ETTh1、ETTh2、Electricity 和 Traffic 上，使用当前仓库实现的共享 Mamba-2 `MambaACF`，训练集拟合的有符号 ACF 仅用于合法的 causal memory 和 AutoCon relation representation，是否能在多个 horizon 上相对预先指定的 PatchTST 与 Mamba2 不劣，并且 ACF 的收益不是任意 lag memory 的副作用。DLinear、NLinear、TimesNet、iTransformer 和 Mamba1 仍是必须报告的描述性基线，但不由当前 primary gate 支持“对所有基线不劣”的结论。

结论只能绑定到本仓库的实现、数据哈希、划分、硬件、优化器和训练预算。即使通过，也不能写成“所有 Mamba 都更准确”或“所有场景都更快”。

## 2. 预先冻结的设计

- 数据：四个固定 CSV，`features=M`，`seq_len=96`，horizon 为 `96/336/720`；计划文件登记了路径、列数、时间列、目标列和 SHA256。
- 选择阶段 seeds：`1101,1102,1103`，只运行 H96 验证集，命令带 `--skip_test`。选择阶段的测试 loader 不得被实例化或读取。
- 确认阶段 seeds：`2021,2022,2023,2024,2025`。每个 dataset-horizon-seed 是一个配对单位；滑动窗口和通道不当作独立样本。
- 公共容量合同：`d_model=32`、`d_ff=64`、一层 Mamba、`d_state=16`、`d_conv=4`、`expand=2`、Mamba-2 `headdim=16`、batch `32`、cosine 学习率、50 epoch 上限、patience 10。这与已完成 v3 screen 的 MambaACF 容量相同；所有模型共享数据、训练预算、优化器和正则化合同。OOM 只能按 launch sidecar 中预先写明的规则降低 batch，不能在看过测试结果后换宽度或结构。
- 数据质量、GPU smoke、streaming 等门槛必须在 launch 前完成并留下哈希；这些门槛不是本 JSON 的训练作业。
- `registered_job_counts` 是不可省略的矩阵合同：runner 会按当前 phase 配置重新展开并逐 phase 校验，同时复算 `selection_total=144`、`confirmation_total=740` 和 `training_total=884`；缺失、额外或不一致的键都会阻止 launch。

### 变量-token 分支的地位

已完成的 ETTh1/Traffic、H96、三 seed variable-token screen 只提供方向性机制证据。它登记在独立的 [`mamba_acf_variable_token_fair_screen_plan.json`](./mamba_acf_variable_token_fair_screen_plan.json) 中，属于 exploratory candidate，不进入本确认矩阵，也不改变本计划的 `144` 个 selection、`740` 个 confirmation 或 `884` 个总作业计数。只有在单独审阅、冻结并重新哈希的 replication plan 获批后，变量-token 臂才可进入确认性结果表。runner 中的 `MambaACFVariable` 和 `MambaACFVariableAutoCon` 是可审计的 arm aliases：分别调用 `MambaACF`/`MambaACFAutoCon`，并通过 `mamba_variable_mixer=true` 覆盖实现；它们不是新增的独立神经网络类，alias、executable model 和 resolved overrides 必须保存在 manifest/config hash 中。

该独立 screen 预注册三个 phase：variable plain、variable + `train_peaks_available` memory、variable + memory + global `hard_selected` relation（`relation_min_lag=96`、lambda `0.03`），均为 ETTh1/Traffic、H96、seed `2021/2022/2023`、8 epochs/patience 3，共 `6+6+6=18` 个 exploratory jobs。它只用于判断是否值得另行设计变量-token replication；不允许用 screen test 指标回填或修改本确认计划。

### 选择和确认必须分离

Phase A 先运行三部分：强基线/ACF-off screen（96 jobs）、ACF memory screen（12 jobs）和 AutoCon memory+relation 的 lambda grid（36 jobs），共 **144 个验证作业**。强基线含仓库最低集合 `DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2`，另外保留 iTransformer；用 best validation scaled MSE 的 mean relative criterion 选择 lambda，并生成 `selected_hyperparameters.json` 及其 provenance sidecar。

只有在 Phase A 的 manifest、source snapshot、数据哈希和 validation trajectory 通过严格 validator 后，才能冻结配置并打开确认阶段。Phase B 的 740 个作业固定配置，在四个数据集和三个 horizon 上使用五个新 seed；不能用确认测试集重选模型、lambda、lag、order 或 epoch。

当前确认配置固定使用 `AutoCon_acf_source=raw`。训练先验是在完整 train split 上估计，而推理记忆在每个 encoder window 内计算；现有 residual/trend 分解在窗口边界会产生不同数值，不能支持严格的先验归因。若以后要恢复 residual/trend，必须先实现窗口一致的变换并新建、复核、哈希一份计划。

这里的 lambda 选择以 combined memory+relation arm 为准，relation-only arm 使用同一个冻结 lambda。这样 relation-only 是增量分解，不会被单独调参后再与 combined arm 做不公平比较。MambaACF trunk、ACF source、lag 数量、relation scope 和否证臂均在 Phase A 前冻结；Phase A 只选择 lambda，不从确认测试结果选择结构。

Relation selection 还要通过可行性门槛：每个 cell 的 `mamba_acf_relation_effective_lags` 必须非空、在 train-only ACF 测量范围内，并且是解析出的真实 train-peak lag 子集；如果代码退回到仅由 `relation_min_lag` 构造的 fallback lag，该 cell 不能支持“peak relation”归因。global candidate/active-anchor coverage 也必须是有限且非零值。

### 容量公平的精确定义

“相同 `d_model`”不等于参数量匹配。该方案把公平性分成两层：所有模型共享数据、训练预算、优化器和正则化；Mamba 架构归因则只以同宽度/深度的 Mamba2 为 capacity-matched comparator。launch 前必须运行 `scripts/benchmark_model_resources.py` 的参数审计，并要求每个数据集 H96 下 combined `MambaACFAutoCon` 与 Mamba2 的 active trainable parameter ratio 不超过 `1.5`。超出门槛时，本计划不能把比较称为容量匹配，必须新建计划、调整宽度/深度并重新选择。

当前源码的只读 dry audit（`d_model=32`、H96）显示 combined/Mamba2 active parameter ratio 约为 `1.35`（7 通道 ETT）、`1.01`（Electricity）和 `0.90`（Traffic），因此门槛看似可行；这些数字不是 launch-time 证据，实际 launch 必须从不可变 source snapshot 重算。PatchTST、TimesNet、iTransformer、DLinear 和 NLinear 保留为必需的公开预算 accuracy baseline，但不被虚称为参数匹配；最终表必须给出 total 和 active 参数量。

## 3. 臂和对照

确认矩阵最多 **740 个作业**：

| 臂 | 配置 | 目的 |
|---|---|---|
| 强基线 | DLinear、NLinear、PatchTST、TimesNet、iTransformer、Mamba1、Mamba2 | 满足仓库最低基线集合，并提供线性、patch/frequency、inverted Transformer 和两个 Mamba 版本的容量/精度锚点 |
| ACF-off | `MambaACF`，关闭 memory、relation、channel ACF adapter | 同一 MambaACF trunk 的无 ACF 路径对照 |
| ACF peak memory | `train_peaks_available` causal memory | 主 ACF 预测路径；只读 train-only signed raw ACF，并按 horizon 合法可用性加权 |
| 固定 lag placebo | 同一 memory 路径，固定 `[5,11,17]`，独立于训练峰值排序 | 仅作描述性诊断；其 horizon-availability profile 未与 peak arm 匹配，不能支持 peak-selection 归因 |
| AutoCon relation | `MambaACFAutoCon`，global signed relation，无 forecast memory | 必须保留的 AutoCon representation/relation 分支，隔离其增量作用 |
| AutoCon memory+relation | memory 加 global signed relation | 主要组合方法 |
| soft sensitivity | H96、`soft_multi_peak`、四数据集五 seed | 只作预先声明的语义敏感性，不得据此把 soft mode 晋升为主方法 |

ACF-off 在训练阶段仍可为 provenance 保存 train-only prior，但当前 `is_training=0` 评估路径不会重新拟合或挂载它；因此评估 null 不依赖 sidecar，且不把 sidecar 计算成本误算为预测路径成本。固定 lag placebo 也不是有效的 peak-selection 确认性对照：在 `seq_len=96, horizon=96` 时，`[5,11,17]` 只覆盖约 5.2%、11.5%、17.7% 的 forecast steps，而 availability-weighted peak lags 可能覆盖大部分甚至全部 steps。它保留在矩阵中用于描述性诊断，但已从 primary statistical gate 移除。它同时不是 sign-shuffled 或 channel-permuted sham；未来若要恢复 peak-versus-placebo 归因，必须先实现、测试并冻结一个 availability-matched non-peak lag matcher 及其误差容忍度。

relation 臂统一使用 `global`、`relation_min_lag=96` 和 `hard_selected`：下界排除输入窗口重叠，避免把局部复制 shortcut 当作表示学习。soft 臂仅为敏感性分析，因为现有 v3 screen 没有显示稳定精度增益。

## 4. 统计门槛

主指标是 train-standardized test MSE；同时报告 original-scale MAE/MSE/RMSE、每通道误差和 origin-stride diagnostics。所有比较以相同 dataset、horizon、seed 配对，报告 raw MSE delta 和 `log(MSE_treatment/MSE_reference)`。

预先指定的 primary 比较包括（固定 placebo 不在其中）：

1. combined AutoCon memory+relation 对 PatchTST 和 Mamba2（这是当前计划唯一支持的 baseline non-inferiority claim）；
2. ACF peak memory 对 ACF-off；
3. combined 对 memory-only，判断 relation 是否提供增量价值。

计划上每个 primary 比较在 12 个 dataset-horizon cell 上做 seed-level log-MSE ratio 的 one-sided paired t interval，并用 exact Wilcoxon 做敏感性检查。当前已实现的 [`scripts/analyze_mamba_acf_confirmation.py`](../../scripts/analyze_mamba_acf_confirmation.py) 对每个比较固定使用 `alpha=0.05/12` 的 Bonferroni adjusted-alpha simultaneous upper bound；它同时报告跨 12 个 cell 的 Holm-adjusted one-sided p 值，但该区间不是 Holm step-down 反演区间。四个 primary comparisons 已在 JSON 中写成显式 `phase:model::phase:model` 坐标，避免同名模型在多个 phase 中产生歧义；其中只有 combined 对 PatchTST/Mamba2 是 baseline accuracy gate，DLinear/NLinear/TimesNet/iTransformer/Mamba1 仅作为完整描述性行。固定 placebo phase 仅保留为描述性诊断。该 analyzer 已通过单元测试并纳入 runner 的 source snapshot；状态仍为 `implemented_pending_human_review`，因此在人工复核区间构造、Wilcoxon 零差异约定、配对规则和 launch-sidecar 哈希前，不得宣称确认性非劣/优越结果。五个 seed 足以做配对不确定性报告，但仍然是小样本，不能把“不显著”写成等价。

- **非劣门槛（由 analyzer 计算、待人工批准）：** log-MSE ratio 的 Bonferroni adjusted-alpha 单侧区间上界不超过 `log(1.01)`（1% multiplicative MSE margin）。
- **优越门槛（由 analyzer 计算、待人工批准）：** log-MSE ratio 的 Bonferroni adjusted-alpha 单侧区间上界小于 `log(1.00)`。
- 任一 cell 缺少五个有效 seed、存在非有限指标或 provenance 不一致，均阻止该 cell 的确认性汇总；不能静默删掉失败作业。

ACF 的解释分两层：若 peak memory 不满足对 ACF-off 的非劣，不能声称 ACF 带来预测收益；固定 lag placebo 只允许报告覆盖率与误差的描述性差异，不能据此声称 peak selection 或 generic lag memory 的因果归因。若 memory+relation 劣于 memory-only，应报告 AutoCon relation 在该合同下没有确认的额外精度价值。

## 5. 资源和效率

资源测量不混入训练矩阵，由 `scripts/benchmark_model_resources.py` 在 Phase A 冻结后、确认测试解释前执行。PatchTST、TimesNet、Mamba1、Mamba2 和冻结的 MambaACF 候选必须使用同一 GPU/driver、dtype、batch、loader、序列长度、horizon、warmup 和 timed iterations；至少 50 次 warmup、200 次计时，三次重复，报告 median/p90。

必须同时报告 total parameters、active trainable parameters、allocated/reserved peak GPU memory、device-resident latency、throughput 和包含 data loading 的端到端 epoch time。只有在精度非劣门槛通过且对指定 nonlinear comparator 的 median epoch time 每个声称 cell 至少低 10% 时，才允许写 speed advantage。DLinear/NLinear 是精度锚点，不是 nonlinear speed comparator。

## 6. Launch gate 和停止规则

人工批准前不得启动本计划。runner 对声明 `execution` 的计划强制要求真实 launch 时 `execution.authorized=true`、并要求 `--launch_sidecar` 绑定计划绝对路径、SHA256、审批人、UTC 审批时间，以及每个 `required_before_launch`/`prelaunch_gates` 的 SHA256 证据。确认性计划还要求顶层 `status` 与 `execution.status` 一致且为 `approved_for_launch`（或 `launch_authorized`）。runner 会把 sidecar 原文复制到每个 phase 的 provenance 目录并记录副本 hash；无 `execution` 的旧 exploratory 计划保持兼容。批准时应把 JSON、源代码快照、环境（PyTorch、CUDA、`mamba_ssm`、driver）、四个数据哈希和资源脚本哈希写入 sidecar。任何代码、数据、硬件、选定配置或统计规则变化都必须新建 plan/output root，重新做 Phase A。统计 analyzer 也拒绝未授权确认计划；它只接受带相同 launch sidecar hash 的 phase provenance。统计 analyzer 已实现、测试并写入 runner source snapshot；当前状态为 `implemented_pending_human_review`，仍需人工审阅后才能授权 launch。

选择阶段的 `--skip_test` 是技术上的 no-test gate。确认阶段则会在每个冻结作业结束后生成 test artifact；这不等于允许边跑边看结果。必须等所有确认 phase 的 manifest、artifact hash 和 provenance 通过 validator 后，才能做汇总、检验或调整论文叙述；任何中途失败都按预先的停止规则处理。

`--validate_only` 和 `--dry_run` 仍可对草案做 job expansion，但不会创建 provenance 或调用训练。真实 launch 必须使用经人工批准、重新哈希且带 `--launch_sidecar` 的副本，并遵循 selection/provenance gate；不能把本草案直接当作已授权训练命令。sidecar 的 `kind` 可使用 `training_plan_launch_authorization`（兼容旧确认草案 kind `mamba_acf_confirmation_launch_authorization`），并应包含 `prelaunch_gate_evidence_hashes` 对象。

下列任一情况出现时，交付物应是可复核的消融/否证报告，而不是组合方法的发表性结论：

- Phase A 未完整通过验证，或 AutoCon lambda 没有独立 attestation；
- 任一确认 cell 少于五个有效 seed；
- combined 臂未通过对预先指定 comparator 的非劣门槛；
- peak memory 与 ACF-off/placebo 的方向不支持预先声明的 ACF 解释；
- 资源协议不完整，或没有达到逐 cell 的 10% epoch-time 门槛；
- 发现测试集被用于选择，或任一 provenance/data/source hash 改变。

## 7. 当前作业规模

计划登记 **884 个训练作业**：144 个 validation-only selection + 740 个 frozen confirmation。资源测量、数据质量和 smoke test 不计入训练矩阵。这个规模是上限；早停可以减少实际 epoch 数，但不能减少预先登记的 seed/cell，也不能在查看结果后删臂。

本文件和 JSON 只完成方案整理与 `validate_only` 校验，未启动任何正式训练。
