# MambaACF 核心状态（更新于 2026-09-02）

## 一句话判断

当前方案可以作为一个有明确研究问题的探索性原型，但还不能作为“普遍优于基线”或“已达到高水平论文证据”的结论。纯 Mamba 的效果明显依赖数据集：冻结高维读数中，Mamba-2 在 Electricity/H96 的 MSE 为 `0.183130`，Traffic/H96 为 `0.648306`，同期 PatchTST 分别为 `0.184448` 和 `0.501767`。Traffic 上的差距约为 `+29.2%`，说明仅替换 Mamba 编码器不能解决跨变量和局部模式问题。

当前最合理的定位是：**Mamba 时间编码器 + train-only signed ACF 先验 + AutoCon 表示学习**。ACF memory 是预测分支，ACF relation 是辅助表示分支；两者不是同一条路径，不能把 relation loss 的结果直接解释为 memory 的预测收益。

## 最新可复核实验

### 当前代码的变量-token公平基线与关系null screen

这是当前代码下最完整的一组机制筛选：计划 [`mamba_acf_variable_token_fair_screen_plan.json`](mamba_acf_variable_token_fair_screen_plan.json) 注册了五个 phase、ETTh1/Traffic、H96、三个 seed、共 30 个作业。所有 phase 均为 `valid=true`、`6/6` 坐标完整；严格分析器读数在 [`mamba_acf_variable_token_fair_screen_readout_20260901.md`](mamba_acf_variable_token_fair_screen_readout_20260901.md)，独立 `is_training=0` 回放为 `30/30` contract pass、`30/30` metric match、drift `0`，环境为 Python `3.9.25`、PyTorch `2.8.0`、`mamba_ssm 2.3.1`。

| 配对对比（当前代码） | n | MSE delta | 相对变化 | MSE 改善点数 | 解释 |
|---|---:|---:|---:|---:|---|
| variable memory on - off | 6 | `-0.005546` | `-1.168%` | 6/6 | train-only raw ACF memory 的描述性增量 |
| `MambaACFVariableAutoCon` `lambda=.03` - 同架构 `lambda=0` | 6 | `-0.004630` | `-0.979%` | 5/6 | 唯一可解释的 relation-loss 机制对照 |
| horizon calibration - relation | 6 | `-0.000119` | `-0.025%` | 3/6 | 基本中性，不支持继续增加校准器复杂度 |

关系 null 对照按数据集为 ETTh1 `2/3`、Traffic `3/3` 个 seed 改善；描述性 paired t 的 MSE `p=0.0848`，不能当作确认性显著性。relation coverage 约 `2.98%`、sign accuracy 约 `91.3%`，说明当前损失只约束少量合法窗口对，不能写成“完整学习了 ACF 相位关系”。

### 与同协议外部基线的窄比较

单独的 42-job 公平基线 screen（[`mamba_acf_fair_baseline_screen_plan.json`](mamba_acf_fair_baseline_screen_plan.json)）在相同 H96/3 seed/8 epoch 条件下比较 DLinear、PatchTST、Mamba-1、Mamba-2 与 MambaACF memory/relation/calibration。其 readout 位于 [`mamba_acf_fair_baseline_screen_readout_20260901.md`](mamba_acf_fair_baseline_screen_readout_20260901.md)。这组数据与上面的 variable-token screen provenance 独立，不能合并做配对推断。特别是 fair screen 的 `relation - memory` 同时切换了 `MambaACF` 到 `MambaACFAutoCon` 可执行分支并启用 AutoCon 目标，属于混杂的 staged comparison；不能解释为纯 relation-loss 增量。

| 数据集 | DLinear | PatchTST | Mamba-1 | Mamba-2 | MambaACF memory (shared) | MambaACF relation (shared) | MambaACF calibrated (shared) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1/H96 | 0.395581 | **0.380848** | 0.401598 | 0.413333 | 0.386960 | 0.393167 | 0.393062 |
| Traffic/H96 | 0.680158 | **0.555538** | 0.651591 | 0.654318 | 0.567329 | 0.564118 | 0.562918 |

上表最后三列来自独立 fair screen 的 shared-token MambaACF 臂，不是下面 30-job variable-token screen 的结果。shared calibrated mean 在 ETTh1 比 PatchTST 高约 `3.207%`，在 Traffic 高约 `1.329%`；因此这组候选尚不能声称“精度不低于基线”。它明显优于本轮短预算的 DLinear/Mamba-1/Mamba-2，但这不能替代多 horizon、5-seed、容量匹配的确认矩阵。

当前 variable-token screen 的自身均值为：

| 数据集/H96 | PatchTST（独立 fair screen） | variable relation | variable calibrated | 相对 PatchTST（relation / calibrated） |
|---|---:|---:|---:|---:|
| ETTh1 | 0.380848 | 0.384496 | 0.385372 | `+0.958% / +1.188%` |
| Traffic | 0.555538 | 0.552047 | 0.550932 | `-0.628% / -0.829%` |

这张表跨越两个独立 provenance，只作描述性上下文；它不能替代同一计划内的配对基线，也不能把 Traffic 的单一 H96 优势外推到其它 horizon。

### 资源边界

当前 calibrated variable-token `MambaACFAutoCon` 的 device-resident synthetic forward 读数为：ETTh1 参数 `58,953`、median `1.557 ms`；Traffic 参数 `169,248`、median `1.669 ms`（B=32，10 warm-up/30 timed，RTX 5090 D）。与独立 fair-baseline 资源工件中同一 10/30 协议的 PatchTST 读数（Traffic `4.057 ms`、ETTh1 `0.283 ms`）相比，candidate 只在 Traffic 该输入形状更快，在 ETTh1 更慢且参数更多。该结果只能说明一个高维 Traffic 的 forward 负载方向，不能写成普遍端到端训练/部署速度优势。资源工件见 [`mamba_acf_variable_token_resources_20260901`](../../artifacts/exploratory/mamba_acf_variable_token_resources_20260901/) 和 [`mamba_acf_fair_baseline_resources_20260901`](../../artifacts/exploratory/mamba_acf_fair_baseline_resources_20260901/)。

新加入的 memory-gate mode 目前只有资源证据，没有精度结果。canonical 测量使用不读数据的 deterministic synthetic ACF prior（`source_attestation.verified=false`），不是 train-only raw 结果。它显示 `hidden` gate 输出为 `[B,H,1]`，`variable_token` 输出为 `[B,H,C]`；ETTh1/C=7 的 median 为 `1.4372 -> 1.4395 ms`（约 `+0.16%`），Traffic/C=862 为 `1.5957 -> 1.6704 ms`（约 `+4.68%`）。两者参数量相同；本次 allocator 峰值增量恰好相同，但不能据此认为内存成本相同，token 模式的结构 gate MAC 从 `3,072` 增到 `2,648,064`/batch item（Traffic），激活也按通道数放大。资源表和非配对旧记录的排除说明见 [`mamba_acf_memory_gate_mode_resources_20260902.md`](mamba_acf_memory_gate_mode_resources_20260902.md)；该模式尚未进入训练或论文主表。
追加的只读 GPU smoke（RTX 5090 D、`mamba_ssm=2.3.1`、合成静态 prior、ETTh1/C=7/H96/B=2）验证了两种 gate 的真实 forward/backward：预测均为 `[2,96,7]`，AutoCon 表示为 `[2,96,32]`，输出/梯度有限且 `acf_gate` 与输入梯度非零。该 smoke 不读取数据集、不训练、不产生结果工件；它只证明实现和梯度合同。由于 variable-token gate 从 variable branch 取源，它在 ReZero 初始点仍会改变该 branch 的梯度路径，故后续精度 screen 必须按功能容量/优化路径消融解释。

### Direct-anchor 分支：已实现，尚未训练

代码中新增了默认关闭的 `mamba_acf_anchor_copy`。启用后，它从同一个 raw、static、train-only ACF sidecar 为每个 `(horizon, channel)` 选择可用的最大 `|ACF|` lag，读取严格因果位置 `L-1+h-lag`，乘以 ACF 符号，再以一个初始化为零的 `tanh` ReZero 标量作为独立残差注入预测。它不复用现有 soft-memory 的共享 gate/reliability，也不生成 `[B,H,C,K]` 候选张量；资源 profile 标记为 1 个新增可训练参数和零 dense selection MAC。无效 lag 在 argmax 前后均被 mask，因此不会循环复制或读到未来值。

该实现只证明结构、梯度和 sidecar 合同，不证明精度提升。专门的 [`mamba_acf_anchor_copy_screen_plan_20260902.json`](mamba_acf_anchor_copy_screen_plan_20260902.json) 注册了 12 个 H96 配对作业：同一 variable-token `MambaACFVariableAutoCon`、hidden gate、raw sidecar、global signed relation 和训练预算，仅切换 anchor flag；计划仍为未授权草案。只读入口 [`analyze_mamba_acf_anchor_copy_screen.py`](../../scripts/analyze_mamba_acf_anchor_copy_screen.py) 在没有完整训练工件时返回 `NOT_RUN`，所以当前没有可报告的 anchor accuracy 数字。

独立资源 smoke 已在同一 RTX 5090 D 上按当前 `MambaACF.py` 源码重测，记录见 [`mamba_acf_anchor_copy_resource_smoke_20260902.md`](mamba_acf_anchor_copy_resource_smoke_20260902.md)。50/500 合成 device-resident forward 的 median 开销为 ETTh1 `+4.04%`（`1.2759 -> 1.3274 ms`，B=2）和 Traffic `+4.35%`（`1.3049 -> 1.3616 ms`，B=1）；参数各增加 1 个标量。相对 model-plus-input 基线的 peak increment 相同，但 static buffer 分别增加 `0.0160 MB` 和 `1.9730 MB`，不能把后者省略。JSON/CSV 同时记录 model source（`41281fc7...5510de16`）与 benchmark tool 的 SHA-256。该结果只证明额外资源负载较小，不能写成端到端速度优势，也不能替代 anchor 精度训练。

NLinear 也被发现是现有短 screen 的基线缺口。其独立 6-job 登记见 [`mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json`](mamba_acf_nlinear_baseline_gap_screen_plan_20260902.json)；旧 NLinear 记录使用 3 epochs 或不同 batch/provenance，不能直接拼接到当前矩阵。

### 配对 relation null control

计划和工件：[`mamba_acf_relation_null_control_plan.json`](mamba_acf_relation_null_control_plan.json)、[`analysis.json`](../../artifacts/paper_final/mamba_acf_relation_null_control_20260901/analysis.json)、[`readout`](mamba_acf_relation_null_control_readout_20260901.md)。

两个臂使用相同的 MambaACF variable-token AutoCon、数据、seed、训练预算、train-only ACF sidecar、关系目标和 lag-pair sampler，唯一变化是 relation-loss 权重：`lambda=0.0` 对 `lambda=0.03`。两阶段各 `6/6` 作业通过 validator，合计 `12/12`。

| 对比（0.03 - 0.0） | n | 平均 delta | 相对变化 | 较低误差次数 | 双侧 p |
|---|---:|---:|---:|---:|---:|
| MSE | 6 | `-0.003404` | `-0.732%` | 4 | `0.0454` |
| MAE | 6 | `-0.003140` | `-0.821%` | 4 | `0.0980` |

按数据集拆分，ETTh1 MSE 为 `-0.208%`（1/3 seed），Traffic 为 `-1.101%`（3/3 seed）。这支持“在这个 H96 配对矩阵中，非零 relation 梯度可能有贡献”的窄结论；它不支持跨数据集、跨 horizon 或 SOTA 结论。`n=3` 的数据集内结果尤其不能用来宣称稳定性。

### 当前代码的 raw relation-target screen

计划、逐 job 工件和读数：[`mamba_acf_relation_target_screen_plan.json`](mamba_acf_relation_target_screen_plan.json)、[`mamba_acf_relation_target_screen_readout_20260901.md`](mamba_acf_relation_target_screen_readout_20260901.md)。这是当前代码、raw train-only sidecar、variable-token MambaACF AutoCon、global relation、memory 关闭、H96、三 seed/数据集的 12-job 配对筛选；唯一干预是 `hard_selected` 对 `soft_multi_peak(abs(ACF)>0.5)`，selected-lag boost 为 0。独立 high-ACF telemetry 阈值固定为 `0.1`。

| 对比（soft05 - hard） | n | 平均相对 MSE | MSE 较低次数 | 平均相对 MAE | 独立 high-ACF 排除率（pooled） |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 3 | `-0.045%` | 2/3 | `-0.033%` | 见下方分数据集诊断 |
| Traffic | 3 | `+0.279%` | 0/3 | `+0.380%` | 见下方分数据集诊断 |
| pooled | 6 | `+0.146%` | 2/6 | `+0.160%` | `78.47% -> 66.62%` |

soft05 确实增加了 positive/anchor（`0.923 -> 1.470`）并把独立 high-ACF 条件排除率降低约 `15.1%`；按数据集分别计算，ETTh1 为 `81.48% -> 67.31%`，Traffic 为 `75.46% -> 65.94%`。这些覆盖变化没有带来预测收益；Traffic 为 `0/3` seed 改善，触发预注册停止规则。该结果只能否定“扩大 strong-ACF target 在当前协议下自动提高精度”的假设，不能证明所有 soft target 都无效，也不能称为 false-negative 修复或确认性结果。12/12 phase validator 通过，12/12 standalone sidecar replay 通过；回放只验证加载/指标链路。

### 关系目标的实际覆盖

`residual` source 的 relation null control 中，hard-selected/global 平均 global valid fraction 为 `2.977%`、active-anchor fraction 为 `91.447%`、positive/anchor 为 `0.923`，条件 high-ACF 排除率为 `76.063%`。这个数字只适用于该 residual-source null-control 工件；当前 raw screen 的对应 hard/soft05 排除率为 `78.47%/66.62%`，不能跨 source 合并。lambda=0.03 并未提高 weighted sign accuracy（约 `91.25%`，null 约 `91.61%`）。因此当前增益不能被描述为“模型学会了全部 ACF 相位关系”。

另一个统计边界是 global loss 对无 positive anchor 的行返回零，随后直接对 batch 求均值；hard/soft target 的 active-anchor 和 positive 密度不同，会改变相同 nominal lambda 的实际梯度权重。比较 target mode 时必须报告 active fraction、relation-loss scale，最好再做 active-anchor 归一化或梯度尺度控制。

### Sidecar 独立回放

null-control 的 12 个 checkpoint/sidecar 已在隔离目录以 `is_training=0` 回放：[`replay_report.md`](../../artifacts/exploratory/mamba_acf_relation_null_control_replay_20260901/replay_report.md)。结果为 `12/12` contract pass、`12/12` metric match、`12/12` sidecar hash 相同；最大绝对差约为 MSE `7.7e-10`、MAE `5.5e-10`。这证明评估加载和 train-only sidecar 复现链路可靠，不等于 current-tree 重训确认，也不是新的统计重复。

针对三阶段 live screen 的新版数据绑定回放见 [`mamba_acf_sidecar_replay_audit.md`](mamba_acf_sidecar_replay_audit.md) 和 [`mamba_acf_data_bound_replay_20260902_v2`](../../artifacts/exploratory/mamba_acf_data_bound_replay_20260902_v2/)。18 个历史 checkpoint 全部 `18/18` contract pass、`18/18` metric match；回放前额外核对 immutable source snapshot、plan/provenance/phase-validation 哈希、实际数据集 SHA-256、每个输入工件哈希和 job/config identity，并固定回放工具自身 SHA-256。它提高了评估路径的可追溯性，但不把历史训练结果升级为 current-tree 或 confirmatory 结果；当前 validator 对旧 relation 日志仍会因缺少新 sampler telemetry 而拒绝严格关系聚合。

### 三臂低秩 channel gate screen

计划、分析和回放：[`mamba_acf_channel_gate_screen_plan.json`](mamba_acf_channel_gate_screen_plan.json)、[`analysis.json`](../../artifacts/paper_final/mamba_acf_channel_gate_screen_20260901/analysis.json)、[`readout`](mamba_acf_channel_gate_screen_readout_20260901.md)、[`replay_report.md`](../../artifacts/exploratory/mamba_acf_channel_gate_replay_20260901/replay_report.md)。三臂均使用 raw train-only sidecar、同一 variable-token AutoCon、同一 global relation target、同一 8-epoch 预算和同一 3-seed/dataset 矩阵；共 `18/18` 作业通过 validator，`18/18` replay contract/metric 检查通过。

| 对比 | n | MSE 平均 delta | 相对变化 | MAE 平均 delta | 读法 |
|---|---:|---:|---:|---:|---|
| ordinary gate - no gate | 6 | `-0.002091` | `-0.452%` | `-0.000129` (`-0.034%`) | Traffic MSE `-0.752%`（3/3），ETTh1 基本不变 |
| ACF gate - ordinary gate | 6 | `+0.001601` | `+0.347%` | `+0.000783` (`+0.206%`) | Traffic MSE `+0.720%`（3/3），抵消 ordinary gate |
| ACF gate - no gate | 6 | `-0.000489` | `-0.106%` | `+0.000653` (`+0.172%`) | 没有一致的整体收益 |

这组结果的关键是门控确实被优化：ordinary gate checkpoint 的 gate residual scale 平均绝对值约 `0.0813`（最大 `0.408`），ACF-conditioned 臂的 ACF latent scale 平均约 `1.03`。因此 ACF gate 的负结果不能归因于“模块保持零开关”。在当前 relation sampler、`seq_len=96` 和 raw source 下，ACF 通道摘要没有改善 ordinary gate，Traffic 上反而造成退化；暂不应继续叠加该分支。

资源 benchmark（同一 RTX 5090 D、B=32、10 warmup/30 timed、synthetic device-resident forward）显示：Traffic ordinary gate 参数 `169,242 -> 191,663`（约 `+13.2%`），median latency `1.462 -> 1.544 ms`（约 `+5.6%`）；ACF-conditioned gate 为 `1.743 ms`（约 `+19.2%`）。这些是架构资源读数，不是端到端训练速度或预测证据。资源文件在 [`mamba_acf_channel_gate_resources_20260901`](../../artifacts/exploratory/mamba_acf_channel_gate_resources_20260901/)。

### 低秩预测头筛选

计划和读数：[`mamba_acf_low_rank_head_screen_plan.json`](mamba_acf_low_rank_head_screen_plan.json)、[`mamba_acf_low_rank_head_screen_readout_20260901.md`](mamba_acf_low_rank_head_screen_readout_20260901.md)。rank-16 将两个 `96x96` 预测投影改为 `96x16x96` 因子化投影；AutoCon 使用的表示张量不变，因此该对比主要检验预测头容量。两臂均为 raw train-only sidecar、variable-token、memory、global hard relation、H96、每数据集 3 seed，`12/12` 作业和两个 phase validator 均通过。

相对 full head，rank-16 的 MSE 平均增加 `1.095%`（`0/6` 配对点更低），MAE 增加 `0.199%`；短训练 elapsed 平均减少 `0.977%`，但只有 `3/6` 更快且没有统一端到端计时协议。预测头参数从 `18,624` 降到 `6,336`（`-66.0%`），但总参数仅 ETTh1 `-20.85%`、Traffic `-7.26%`，所以没有形成可宣称的速度收益。结论是保留 full-rank head；rank-16 作为压缩负结果归档，不进入主模型。

### 关系覆盖与归一化筛选

三臂读数：[`mamba_acf_relation_coverage_screen_plan.json`](mamba_acf_relation_coverage_screen_plan.json)、[`mamba_acf_relation_coverage_screen_readout_20260901.md`](mamba_acf_relation_coverage_screen_readout_20260901.md)。三臂共 `18/18` 个作业通过验证，比较 random pairing、least-exposed-lag balanced pairing 和 active-anchor normalization。balanced 将两条 effective lag（`[96,120]`）的 aggregate exposure span 压到 `0--0.135%`，而 random 已为 `0.124--0.951%`；但 balanced 相对 random 的 MSE/MAE 分别为 `+0.239%`/`+0.249%`。active-anchor normalization 的 MSE/MAE 变化仅 `+0.013%`/`+0.018%`，没有可见精度改善。

这组结果把当前瓶颈定位得更清楚：不是两条 lag 的频率失衡，也不是把 global loss 从全 batch 均值改为 active-anchor 均值。raw screen 虽能减少 hard target 对独立 high-ACF candidate 的排除，却没有提升预测精度；下一轮不再扩展 soft 阈值网格，应优先处理跨变量 lag coherence、horizon-specific memory 校准和容量公平性。

### 通道顺序与 ACF conditioning screen

最新的 18-job screen（[`mamba_acf_channel_order_conditioning_screen_readout_20260902.md`](mamba_acf_channel_order_conditioning_screen_readout_20260902.md)）只改变 train-prior 的通道顺序或 ACF token conditioning，其余为同一 variable-token、global-relation、raw-sidecar 配置。strength order 相对 input control 的 pooled MSE 为 `+0.113%`（3/6 改善；ETTh1 0/3、Traffic 3/3）；conditioning 相对 input control 为 `+0.598%`（2/6；Traffic 0/3）。两项均未达到预注册停止规则，已停止继续扩展。18/18 合同/指标 replay 通过，但 immutable snapshot 与当前工作树存在四个 source 文件漂移，故不能称为 current-tree 重跑。

### 本轮 current-source 合同修复（2026-09-02）

- Mamba-1 与 Mamba-2 改为独立可选导入；缺少其中一个 kernel 不会阻断另一个。`MambaACF + AutoCon_wnorm=Decomp` 现由运行、计划和 phase validator 在启动前拒绝，因为该归一化分支与 MambaACF 的预测合同不兼容。
- lag-pair sampler 现在仅在 relation objective 实际使用全局窗口关系、且 selected lag 确实定义正样本时启用：hard target 必须有 `lag_mask=true`，soft target 必须有正的 selected-lag boost，`relation_scope=local` 一律使用普通 shuffle。解析后的 requested/effective sampler 和原因写入 `config.json` 与 `loss_logs.json`。旧的 false-mask hard 或 local-only 工件保留其原始快照，不可与此语义下的重跑结果合并。
- 多尺度值池化对非整除尾块只平均真实 token，时间 mark 改为同一池化块的最后一个真实时间点。可整除块的数值路径保持不变，但默认 `seq_len=96, scale=2` 的 calendar mark 语义已从块首变为块尾；历史 checkpoint 的输出可能因此改变，必须重训/重评估后才可称为 current-source 结果。
- `MambaACF.load_acf_prior` 现在在 sidecar 同时提供 compact/full ACF 表时逐元素核对 `acf_full_values[:, lags] == acf_values`，并验证宽度、lag 唯一性和有限值；仅有 compact 表的旧 sidecar 仍保持兼容。该检查防止 sidecar 的选择 lag 与实际系数列脱钩。
- 静态 ACF cache 仅为启用的预测分支编译，eval telemetry 默认不再强制 GPU-to-CPU 同步。这些是实现/计时合同改进，不是新的精度实验。上述准确率屏幕仍按各自 immutable source snapshot 解读；本轮没有启动新的训练作业。

## 代码审计结论

1. `MambaACF` 的主预测路径先形成 Mamba hidden，再经过 direct multi-horizon head、linear residual、可选 variable-token 分支和 ACF memory。relation 表示默认来自独立 `repr_head`，历史实现中并未自动看到 memory/variable forecast 分支。
2. 新增的 `mamba_acf_repr_fusion` 是默认关闭的 ReZero 适配器，仅在 AutoCon 模式下把 train-only ACF 的四个摘要特征注入 hidden 表示。raw-source 三 seed screen 的 MSE 变化仅 `-0.019%`，checkpoint 门控尺度接近零，当前不构成有效改进证据。
3. ACF 表示融合的 lag index 不读未来位置，但 residual/trend source 使用 centered full-window smoother；只有 `AutoCon_acf_source=raw` 才能把该变换标为严格因果。最新 gate/repr screen 使用 raw，但这只支持 raw 分支的因果声明，不能外推到 residual/trend。
4. `mamba_acf_relation_scope=global` 将时间网格池化为窗口级表示；在 `seq_len=96`、`min_lag=96` 时 local target fraction 为零。当前关系学习是窗口级 pooled regularization，不是局部 token-level 预测约束。
5. variable-token mixer 扫描的是输入 CSV 通道顺序，不是时间轴。ETTh1/Traffic 的变量 mixer 参数量分别比 shared plain 高约 `52.8%`/`12.8%`；没有 parameter-matched control 时，不能把增益归因于 Mamba 或 ACF 单一组件。低秩 channel gate 是另一条 `O(C·rank)` 的跨变量路径，Traffic 上已显示比 ACF 条件化更值得保留的方向，但仍需容量匹配。
6. 现有 memory 选择 `[72,96,120]`/relation `[96,120]` 受 `seq_len=96` 的可用性限制，无法直接覆盖训练 ACF 中的 168/336/504/672 等长周期。要利用这些周期，必须增加 context 或使用经过审计的历史缓存，不能在窗口内循环复制不可用值。

## 如何更有效利用 ACF

- 只用 train split 估计 signed ACF，并把完整表、compact lag、选择来源和 hash 一起保存。
- 将 forecast memory 的 `valid` mask 与 horizon 对齐；不可用 lag 直接屏蔽，不做周期回填。
- 表示学习使用同一 train-only relation 表，并记录每个 lag 的 exposure；本轮已证明 balanced sampler 在 `[96,120]` 上只改善 telemetry，不改善精度。
- active-anchor normalization 已做配对筛选，效果近中性；后续应把计算资源转向 high-ACF false-negative、soft candidate 定义、signed sign accuracy、sign hinge 和每通道统计。
- `lambda=0`、同 sampler、同 AutoCon head 的 raw-source null control 已完成；当前结果支持保留 signed relation 作为候选辅助目标，但不能把它写成普遍精度收益。下一轮应在冻结矩阵中复验，而不是继续在短 screen 上搜索更多 lambda。
- 对 Traffic 重点加入 channel-independent 或 parameter-matched cross-variable/local branch；单纯增加 Mamba depth 不是针对当前失败模式的充分修复。
- 当前架构筛选优先级是 ordinary low-rank channel gate > ACF-conditioned channel gate；后者已完成同协议三臂 screen，但暂不进入下一轮主模型。

## 证据等级与下一步

当前证据等级是“实现完整、sidecar 可复现、机制筛选有方向性”，不是 confirmatory/publication-ready。冻结高维 `_v3` 的 selection 是 `posthoc_verified` 且 `publication_ready=false`；relation null control、raw relation-target screen、repr fusion 和 channel-gate screen 都是 exploratory。

进入论文主表前至少还缺：

1. raw-source、strict-causal 的预注册多数据集/多 horizon 矩阵；
2. high-ACF candidate/false-negative 控制，以及跨变量 lag coherence；
3. parameter-matched baseline、端到端训练时间/峰值显存和统一计时协议；
4. 5 个以上 seed 或预先定义的统计功效与多重比较规则；
5. Traffic 的 channel-level/trimmed/leave-one-channel-out 分析，并对 ordinary gate 做 parameter-matched local/channel-preserving 对照。

当前完整回归为 `511 passed, 36 warnings`（`mamba` 环境，约 21.8 秒；仓库 `pytest.ini` 显式加入 `pythonpath=.`）；本轮新增模型身份回归覆盖 plain/variable MambaACF 别名分离、跨 collector 一致性、资源记录恢复，以及 ACF-off/required sidecar replay 的配置身份和 prior-fit 合同。其中 warning 来自旧 pandas API、小样本配对统计的 precision-loss 和预期的 Mamba headdim 兼容性提示，不是训练失败。Mamba-1/Mamba-2 独立 smoke 也分别通过，输出形状为 `(2,24,1)` 与 `(2,96,16)`。实验环境的 `mamba_ssm` 版本为 `2.3.1`，Python `3.9.25`，PyTorch `2.8.0+cu128`。
