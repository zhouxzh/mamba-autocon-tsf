# AutoCon、Mamba 编码器与 ACF 的核心复核

## 结论先行

Mamba 作为时间编码器做预测是**条件可行**的：它适合长输入、需要线性时间扫描、变量关系不是唯一主要难点的任务；它不是所有多变量数据集的默认最优模型。当前冻结高维结果中，名为 `Mamba2` 的 arm 是“无 relation loss 的 Mamba-2 编码器 + AutoConNet-compatible forecasting head”，不是裸 Mamba forecaster；它在 Electricity 的三个 horizon 接近 PatchTST（H96 `0.183130` 对 `0.184448`，H336 `0.211477` 对 `0.207953`，H720 `0.241182` 对 `0.252428`），但在 Traffic 明显落后（H96/336/720 为 `0.648306/0.623489/0.659914`，PatchTST 为 `0.501767/0.516942/0.550696`）。因此合理定位是“数据集依赖的 Mamba 时间编码器”，而不是“纯 Mamba 普遍优于基线”。

这些数值直接从高维 `_v3` 的 `stream_metrics.json` 重新聚合；每个模型-数据集-horizon 有 5 个 seed。它们是文件校验完成后的后验描述性读数，不是 launch-time 预注册的确认性结果。按匹配 seed 计算，Traffic 上无 relation-loss 的 Mamba2-compatible-head 相对 PatchTST 的 MSE 增量为 H96 `+29.205%`、H336 `+20.611%`、H720 `+19.834%`；Mamba2AutoCon 的增量仍为 `+29.497%/+20.549%/+19.544%`。因此 AutoCon 在这批高维设置中没有解除 Mamba 编码器路径的主要瓶颈。

### 高维神经网络结果（scaled MSE，5 个匹配 seed 的均值）

下表从同一批 `stream_metrics.json` 按 `dataset × horizon × seed` 聚合；数值越低越好。`AutoCon` 后缀表示对应的辅助关系目标已打开。表中结果仍受后验 selection provenance 限制，只用于当前架构判断。

| 数据集 | 模型 | H96 | H336 | H720 |
|---|---|---:|---:|---:|
| Electricity | PatchTST | 0.184448 | 0.207953 | 0.252428 |
| Electricity | Mamba1 | 0.184586 | 0.211008 | 0.244689 |
| Electricity | Mamba1AutoCon | 0.180798 | 0.204395 | 0.246006 |
| Electricity | Mamba2 | 0.183130 | 0.211477 | 0.241182 |
| Electricity | Mamba2AutoCon | 0.182321 | 0.210460 | 0.241146 |
| Traffic | PatchTST | 0.501767 | 0.516942 | 0.550696 |
| Traffic | Mamba1 | 0.650336 | 0.623198 | 0.661425 |
| Traffic | Mamba1AutoCon | 0.640048 | 0.619543 | 0.655154 |
| Traffic | Mamba2 | 0.648306 | 0.623489 | 0.659914 |
| Traffic | Mamba2AutoCon | 0.649773 | 0.623169 | 0.658317 |

在 Electricity，Mamba2 相对 PatchTST 的 matched-seed MSE 变化为
`-0.715%/+1.697%/-4.455%`（H96/H336/H720）；在 Traffic 为
`+29.205%/+20.611%/+19.834%`。Mamba2AutoCon 相对 Mamba2 的变化为
`-0.440%/-0.485%/-0.012%`（Electricity）和
`+0.227%/-0.051%/-0.242%`（Traffic），说明 AutoCon 不是稳定的主效应，
更不能单独解释为跨数据集精度提升。

## AutoCon 原始代码的真实路径

历史 [`models/AutoConNet.py`](../../models/AutoConNet.py) 有三段：

1. `DataEmbedding` 将多变量窗口映射到隐藏空间；原始 `TokenEmbedding` 使用 circular `Conv1d`。
2. `DilatedConvEncoder` 提取 forecasting backbone 表示；在 AutoCon 模式下，独立 `repr_head` 输出对比学习使用的投影表示。
3. 预测端不是编码器直接解码，而是多尺度趋势分解、`length_mlp`、通道 MLP，以及对短期残差的线性 seasonal head。

历史 [`models/Mamba1AutoCon.py`](../../models/Mamba1AutoCon.py) 和 [`models/Mamba2AutoCon.py`](../../models/Mamba2AutoCon.py) 只替换第 2 段的特征提取器；第 3 段仍是 AutoConNet 预测头。历史表中的 `Mamba1/Mamba2` 只是将 `AutoCon=false`，仍复用该预测头；`Mamba1AutoCon/Mamba2AutoCon` 再打开关系目标和表示路径。因此四种历史 arm 都不能被称为“纯 Mamba forecasting”结果。

代码审计还发现两项会影响复核的实现问题，现已修正并加回归测试：历史 [`models/AutoConCI.py`](../../models/AutoConCI.py) 把 `[B,C,T]` 展平为 `B*C` 流时，曾用交错 batch 的方式复制时间标记，导致 `B>1,C>1` 时通道流使用了别的样本的时间特征；现在统一使用 batch-major 的 `repeat_interleave`，并兼容 `[B,T,F]`、`[B,1,T,F]` 和无时间标记输入。其次，`AutoConNet`、`AutoConCI` 与 Mamba-2 兼容头中声明的 `repr_head` 曾未进入计算图；现在只有 `AutoCon=True` 时才把其输出交给对比目标，预测头仍使用未投影的 backbone 表示，普通预测模式则冻结该兼容层。由此，旧 AutoConCI 数值至少应做修正后敏感性重跑，旧 AutoCon 增益不能与新投影契约的结果混合。

原始 [`layers/losses.py`](../../layers/losses.py) 的 legacy `AutoCon` 目标按 ACF 为每个位置选 top-1 正样本，使用绝对 ACF，并把超过表范围的距离 clamp 到最后一个元素。它能复现旧论文方法，但有三个解释限制：符号信息丢失、远距离关系被伪造为同一个末端值、普通随机 batch 可能几乎没有候选关系。当前工作树已对 legacy `AutoCon`/`AutoConCI` 做最小语义修正：身份只由距离矩阵的对角位置定义，表外 lag 从候选分母和正样本集合同时 mask。`ACFRelationLoss` 的 v2 目标（objective version 2）把选定 lag 定义为正样本，同时保留同一 ACF 表范围内的非身份 pair 作为 denominator candidates；这只在同一 anchor 存在其他 in-range candidate 时避免 selected-lag batch 退化为单候选、零梯度目标，不能保证每个 batch 都有有效负例。

这项修正不会把旧工件自动变成新结果。若 ACF 表覆盖了所有窗口距离且只有 lag 0 精确等于 1，数值目标在理想浮点运算下基本保持不变；若出现非零 lag 的 ACF=1，旧实现会把它误当作身份并当作负例，新实现允许它成为正候选；若距离超出表范围，旧实现会复用末端值，新实现不再产生该关系的梯度。历史日志因此仍按原代码口径保留，涉及这些边界情形或 AutoConCI 的比较应在修正后重新训练，不能与旧结果混合。

### ACFRelationLoss v3（当前代码）

当前 [`ACFRelationLoss`](../../layers/losses.py) 的 `objective_version=3` 将候选集和目标集显式分开：候选是 ACF 表范围内、非 identity 且满足 `min_lag` 下界的所有窗口 pair；`selected_lags` 不再自动删除未选中的候选。目标模式由 `mamba_acf_relation_target_mode` 固定为三者之一：

- `hard_selected`：提供 selected lags 时，只有这些 lag（含 tolerance）是正目标；这是 v2 兼容模式。未提供 selected-lag mask 时保留 legacy 的 all-candidate fallback，不应把两种配置混合比较。
- `soft_multi_peak`：所有 `abs(signed_acf) > soft_min_abs` 的候选都是正目标，selected lags 再乘以 `1 + selected_lag_boost`；这是一个语义/机制控制，用于避免把未选但强相关的周期当作假负例，不是已经验证的精度推荐配置。
- `soft_full`：所有非零 ACF 候选都是正目标，是用于敏感性分析的宽松控制。

配置语义需要同时看 `lag_mask`。当 `lag_mask=false` 且
`target_mode=hard_selected` 时，初始化逻辑会把 selected lag 集合置空，
`ACFRelationLoss` 实际退化为所有 in-range candidate 的目标，sampler 也不会
按 selected peak 配对；这应标记为 all-candidate ablation，不能写成
selected-peak supervision。正式日志必须同时报告 candidate fraction、每个
anchor 的 positive count、至少含一个 in-range negative 的 contrastive-active
anchor fraction、退化 positive anchor fraction 和每 lag exposure。

强度项拟合 ACF 幅值，独立的 sign hinge 拟合正负方向。`min_lag` 同时作用于分母候选和正目标，不改变 forecast memory 的 lag 集；将它设为 `seq_len` 并使用 `relation_scope=global` 可构造无重叠窗口控制。`relation_scope=local`/`both` 仍可运行，但 local 分支只在窗口内部位置有候选，短窗口下长 lag 可能完全无效。v3 对 local loss 按 active temporal positions 归一化，避免大量零位置稀释梯度。

v3 日志必须带有 `acf_relation_objective_version=3`、`acf_relation_target_mode`，以及 global candidate/active-anchor/positive-count/contrastive-active-anchor/degenerate-positive-anchor/high-ACF-excluded 和 local target/active-position 诊断。缺少这些字段的旧日志不能与 v3 结果合并。

## 当前 MambaACF 的数据流

[`models/MambaACF.py`](../../models/MambaACF.py) 将预测和表示学习分开：

- 共享时间 Mamba 扫描产生 hidden state，直接经过 multi-horizon head 预测；
- 可选 variable-token 分支让一次 Mamba 扫描建模跨变量关系；
- 可选 patch-token 分支保留通道内局部片段；
- 默认 `mamba_acf_relation_representation=encoder` 时，`hidden_full -> repr_head -> ACFRelationLoss`，避免对比目标直接压缩 forecasting state；
- 受限的 `mamba_acf_relation_representation=forecast` 可将最终归一化预测 `[B,H,C]` 交给 global relation loss；该 loss 会先沿 `H` 平均，所以它只监督窗口级跨通道关系，不是逐预测步的时序 ACF 约束；
- ACF memory 是额外 residual，不是未来值生成器。

训练器 [`exp/exp_long_term_forecasting_with_AutoCon.py`](../../exp/exp_long_term_forecasting_with_AutoCon.py) 与普通 Mamba 路径都从 train split 拟合 signed ACF prior；探索性配置默认 `AutoCon_acf_source=residual`，而严格确认配置固定为 `raw`。`_memory_signal` 保证一次运行内 prior 与 memory 使用同一信号族，但不能弥合整段 train 分解与逐窗口分解在 residual/trend 边界处的数值差异。`ACFLagPairBatchSampler` 只按窗口 origin 配对，不读取未来目标。

这里有四个不能省略的实验边界。第一，`MambaACF` 默认仍使用
[`layers/Embed.py`](../../layers/Embed.py) 中带 `padding_mode='circular'` 的
`TokenEmbedding`；因此只有 ACF memory 的索引是严格因果的，整个模型不是严格左因果模型。
第二，先验是在完整 train 序列上做 moving-average 分解，而推理时在每个窗口上独立分解并复制窗口端点；窗口边界附近的 residual 并不与先验拟合时完全同分布，长 lag/短窗口时尤其要做敏感性分析。第三，当前 head 和 `_memory_signal` 依赖固定 `seq_len`，变长输入不是已承诺的模型合同。第四，历史扩展 pilot 中 plain 路径省略 `weight_decay` 时使用 `0`，AutoCon 路径默认使用 `1e-5`，且有 `[1]` 与 `[1,2]` scales 的差异；这些工件不能用于归因 ACF。正式比较必须显式固定 optimizer、scales、宽度和训练预算。

归一化名称必须绑定数值合同。当前历史 `AutoConNet`、`AutoConCI`、共享
Mamba-1/2 AutoCon head 与 `MambaACF` 的 `ReVIN` 统一使用 PyTorch 默认的
unbiased sample standard deviation；PatchTST 公平比较使用独立的
`MeanStd`（`sqrt(var_population + 1e-5)`）模式。两种模式不能混称或把旧日志
拼接到新表中；若论文需要其他 denominator，必须先注册并重新训练。

## 当前精度瓶颈与下一步

默认路径中，ACF 对 forecast 仍是弱耦合：`_fuse_acf_representation` 只修改送入 AutoCon 的 `hidden_full`；预测使用的 `hidden_base` 和后段 ACF memory residual 是两条独立路径。因此默认 encoder relation loss 对 memory/gate 的梯度只能通过共享编码器间接传递，不能据此声称“对比目标直接训练了 ACF 预测修正”。新增的 forecast view 是受限例外：它让 global relation loss 直接反传到 forecast head 和已启用的 forecast residual/gate，但不是新的预测算子，也不是逐步预测动态监督。它仅允许 `MambaACFAutoCon/MambaACF1AutoCon`、relation loss、global scope、raw ACF、`features=M`、`c_out=enc_in`、至少两个通道且关闭 encoder-only fusion；目前没有该模式的训练结果。

静态 memory 的 lag 系数目前为 `sign(prior) * softmax(log|prior| / temperature)`；绝对 ACF 强度主要进入 reliability，进入 seasonal correction 的是归一化后的相对权重。强周期与弱周期可能因此得到相近的 correction 幅度。已有 horizon-calibration 开关可以恢复部分幅度信息，但现有 screen 近似无收益，不能预先把它写成改进方案。

relation objective 的有效信号也很稀疏：现有旧日志中 global candidate fraction 约 `3%`、positive active-anchor 约 `0.92`，所以它更接近弱正则而非主要 forecast signal。这个旧 active-anchor 字段只表示有正目标，不能证明存在分母负例；当前日志新增 contrastive-active-anchor 和 degenerate-positive-anchor 两项，必须以它们判断幅值对比项是否真正有梯度。扩大 candidate coverage 的已有 screen 没有稳定改善；下一轮应优先报告每个 lag 的有效 pair、符号准确率、两项新 telemetry 和 relation-loss/forecast-loss 比例，再决定是否改变目标，而不是盲目提高 `AutoCon_lambda`。当 `mamba_token_stride>1` 时，插值表示的 token 坐标不再等于原始时间坐标；当前 runner 已禁止 `local/both` relation scope 的该组合，正式 relation 实验固定 `stride=1` 或只使用 global scope。

在 Traffic 等高维数据中，shared temporal path 先将 `C` 个通道压到 `d_model`，relation pooling 再做 channel mean/max-abs；相反相位通道可能相互抵消。这是比继续增加 temporal scale 更直接的精度假设。当前 compact channel lag bank 只把 per-channel ACF 压缩为共享的小型 lag bank，并不是完整的 per-channel lag forecaster；可行的下一候选是 train-only ACF-strength weighted pooling 或小型、保留通道身份的 summary，但必须单独注册容量和表示合同，不能把它混入当前单尺度速度筛选。

训练期 telemetry 采用 detached scalar 的批量设备搬运：MambaACF forecast-path
诊断和 `ACFRelationLoss` 覆盖/符号诊断在设备上先归约，读取一个 batch 快照时
按设备成组搬运。这样保留原有 `loss_logs.json` 字段，同时避免每个字段各触发
一次 GPU-to-CPU synchronization。评估默认不收集这类诊断；任何速度比较仍须
固定诊断开关，并分别报告 prediction-only、prediction-plus-representation 和
端到端训练时间。

可重复微测脚本为 [`scripts/benchmark_acf_telemetry_sync.py`](../../scripts/benchmark_acf_telemetry_sync.py)。在本机 RTX 5090 D、20 个标量、
200 次迭代的读数中，批量路径为旧逐字段路径的 `0.359x`；这是 telemetry
同步开销的算子级观察，不是训练吞吐或预测延迟结论。

## ACF 应该怎样高效使用

推荐把 ACF 用作三类信息，而不是硬编码周期复制：

### 1. 结构配置

用 train-only FFT 计算每通道 signed ACF，按绝对值选稀疏、分离的 peak，记录符号和置信阈值。当前共享模型将这些统计压缩为稳定的 common-lag/compact bank；这不是完整的 per-channel lag 建模。这样把 `O(T log T)` 的统计计算从每个 batch 移到训练初始化。对直接 forecast memory，新增 `train_peaks_available` 选择器，把训练峰值强度乘以当前 `seq_len/pred_len` 下的合法步比例，并排除完全不可用的 lag；这比固定把 `[24,168,336]` 塞进 H96 更符合因果数据流。该选择是 horizon-specific，不能和不同 horizon 的结果混用。

### 2. 表示学习关系

新 [`ACFRelationLoss`](../../layers/losses.py) 先按 lag 查 signed relation，再用 `abs(cosine)` 拟合强度，用独立 hinge 项拟合正/负方向，并对超出 train prior 的 lag 做 mask。v2 中 selected lags 是正样本集合，而同一表范围内的非身份 pair 是分母候选；v3（`objective_version=3`）进一步把 `min_lag` 应用于候选分母和目标，并提供 `hard_selected`、`soft_multi_peak`、`soft_full` 三种目标模式。训练器把 objective version、target mode、候选覆盖率、positive active-anchor、positive-count、contrastive-active-anchor、degenerate-positive-anchor、high-ACF 排除比例和 local active-position 诊断写入 `loss_logs.json`。global scope 对任意输入表示都会先池化其时间/预测步轴；因此 forecast view 是窗口级 relation routing，而不是 token-level 或 horizon-step-level forecasting relation。新旧 relation 结果不能混合，旧 screening 的 loss 曲线只代表修复前目标。

在真实 ETTh1/Traffic train split、`batch=32`、selected lags=`[72,96,120]`、`max_lag=720` 的 v2 epoch-0 审计中，Traffic 每个 anchor 平均有 `4.891` 个 denominator candidates、`0.940` 个 positives、`3.950` 个 non-positive candidates，`90.79%` 的 anchors 至少有一个 non-positive candidate；ETTh1 对应为 `5.850/0.942/4.908` 和 `94.64%`。这说明 v2 已有真实负样本，但约 `91.5%` 的 Traffic anchors 仍只有一个正对。v3 的 soft target 会把强但未选中的 ACF 峰纳入正目标；它的覆盖率不能与 v2 的 selected-positive fraction 直接比较，也不能预设为精度增益。训练日志仍应分别报告 global 与 local 覆盖率；`T=96` 时 local scope 对 lag 96/120 没有有效位置。

当前刻意配对的 lag 72 低于 `seq_len=96`，两窗口共享 `24/96=25%` 的输入历史；ETTh1/Traffic 的 lag-72 配对约占所有刻意配对的三分之一。为排除 representation 从共享观测中获取捷径，新增 `mamba_acf_relation_min_lag`：它过滤 relation 的 lag-pair sampler，并在 v3 中同时过滤 loss 的候选监督，不改变 forecast memory。将其设为 `seq_len` 并使用 `relation_scope=global` 可形成无重叠控制；该字段已纳入运行签名、训练计划、资源清单和 phase validator。若该下界超过测得的 ACF 表范围，正式配置应拒绝，避免辅助目标静默变成 no-op。

### 3. 因果 memory residual

对预测步 `h` 和 lag `k`，只读取 `L-1+h-k` 位于输入窗口内的历史值；不可用项使用 validity mask，不循环复制、不使用模型已经预测的未来值。signed ACF 的符号决定方向，幅值和可用比例决定可靠性，隐藏状态 gate 决定 residual 注入强度。长 horizon 下该 mask 尤其重要：`seq_len=96` 时，lag 168 在 H96 只有 24/96 步可用，lag 336 在 H96 完全不可用。当前实现同时支持跨 horizon 汇总 gate 和 `mamba_acf_step_reliability=true` 的逐预测步 gate；后者只对每个 `h` 的合法 lag 加权，避免有效早期步和无效后期步共用一个系数。该开关已通过 shared/CI CUDA 前向与反向合同测试，正式实验仍应把它作为独立 ablation。

### 计算路径的低风险优化

当前 `ACFRelationLoss` 对 local 表示关系增加了一个无梯度缓存：固定的 `T×T` 距离、signed-ACF relation 和 candidate/selected mask 按 device、步数及 ACF tensor version 复用；global pair 因 origin 每 batch 变化，仍然实时计算。训练器现在在初始化时一次性把 relation loss 的静态 buffers 移到目标设备，避免每个 batch 重复复制完整 ACF 表。缓存不改变 loss 数值，也会在 train-only ACF buffer 原地更新或设备迁移时失效。在 RTX 5090 D、`B=32,T=96,D=64` 的工程微测中，200 次 local loss 由强制 cache miss 的 `84.56 ms` 降至 cache hit 的 `70.29 ms`（约 `16.9%`）；这是算子级观察，不是端到端训练加速或精度证据。

## 当前证据应该怎样读

Traffic、H96、seed 2021、3 epochs 的探索性 MambaACF pilot：

| 结构 | MSE | 相对 plain |
|---|---:|---:|
| plain shared temporal | 0.600947 | -- |
| MambaACF1 plain (Mamba-1) | 0.608056 | +1.18% |
| causal ACF memory | 0.596415 | -0.75% |
| variable-token | 0.587315 | -2.27% |
| ACF relation loss | 0.606501 | +0.92% |
| variable-token + relation | 0.590902 | +0.61% vs variable-only |
| patch-token（旧 all-candidate pilot） | 0.611913 | +1.82% |
| causal embedding + memory | 0.615312 | +2.39% |

在同一 Traffic/H96、同一 3-epoch 预算下，`train_peaks_available` 的训练集选择器解析为 `[96,120,72]`。三 seed 与完全匹配的 plain 对照如下：

| seed | selector MSE | fixed-memory MSE | plain MSE | selector vs fixed | fixed vs plain |
|---:|---:|---:|---:|---:|---:|
| 2021 | 0.586256 | 0.596419 | 0.600948 | 1.704% | 0.754% |
| 2022 | 0.590390 | 0.607324 | 0.611094 | 2.788% | 0.617% |
| 2023 | 0.587594 | 0.599909 | 0.605031 | 2.053% | 0.847% |
| mean | 0.588080 | 0.601217 | 0.605691 | 2.182% | 0.739% |

selector 相对 fixed-memory 的配对 MSE 差均值为 `-0.013137`（描述性 95% t 区间 `[-0.021732,-0.004542]`）；selector 相对 plain 的差均值为 `-0.017611`。按正式协议的双侧 exact Wilcoxon，三个比较均为 `p=0.25`；因为只有 3 个 seed，所以这只是机制筛选信号，不能写成确认性显著收益。工件见 [`artifacts/mamba_acf_selector_replication_20260831/`](../../artifacts/mamba_acf_selector_replication_20260831/)。

逐步 gate 的独立重跑（Traffic/H96、`d_model=32`、3 epochs、seed 2021）得到 scaled MSE `0.585388`、MAE `0.396185`，完整输出位于 [`artifacts/fair_pilot_20260831/rerun_memory_stepwise/`](../../artifacts/fair_pilot_20260831/rerun_memory_stepwise/)。它只用于验证实现和候选优先级；由于是单 seed、单 horizon、短预算，不能与三 seed selector 统计量合并。

这些是候选筛选和实现验证，不是确认性结果：三组仍只有 Traffic/H96、3 epochs，且不能与冻结 PatchTST 表直接比较。它支持把“训练峰值乘可用性”的 memory 增量纳入正式多数据集、多 horizon 消融；不支持声称 ACF 或 Mamba 已稳定提高精度。旧扩展 benchmark 的计时协议不一致，现已降级为历史工程读数；当前统一复测显示 MambaACF plain 的中位前向 `0.684640 ms`、memory-only `0.808272 ms`，PatchTST `0.885824 ms`。这仍只是 device-resident synthetic forward，不能等同于训练吞吐或总体效率优势。

2026-09-07 的 patch-token 修正 pilot 使用 `hard_selected + mamba_acf_relation_lag_mask=true`，因此真正启用了 selected-lag pairing；ETTh1/Traffic 的 patch-on best validation loss 相对 off 分别下降 `2.79%/3.24%`，global contrastive-active anchors 约 `87.0%/84.1%`。但它仍是单 seed、validation-only、4 epoch，训练 wall time 增加 `6.47%/13.57%`，synthetic prediction latency 增加 `40.49%/65.88%`，不能解释为速度收益。完整读数与下一轮 24-job 未授权 screen 见 [`mamba_patch_token_contrastive_pilot_readout_20260907.md`](mamba_patch_token_contrastive_pilot_readout_20260907.md) 和 [`mamba_patch_token_contrastive_screen_plan_20260907.json`](mamba_patch_token_contrastive_screen_plan_20260907.json)。

当前最有方向性的结构信号来自跨变量而非 relation loss：独立的
`MambaACFVariable - MambaACF` H96 screen（ETTh1/Traffic 各 3 个 seed，
未启用 ACF memory 或 relation）给出 MSE `-0.933%/-2.491%`，但训练 elapsed
增加 `31.08%/14.09%`，且依赖 CSV channel order。它支持“shared temporal
state 的跨通道瓶颈值得优先处理”这一假设，不支持把收益归因于 ACF，也不支持
普遍更快；完整逐 seed 读数见 [`mamba_acf_variable_mixer_screen_readout.md`](mamba_acf_variable_mixer_screen_readout.md)。

### 早期 v3 soft relation 机制 pilot（Traffic/H96，单 seed、3 epochs）

为验证目标语义而不是挑选最终模型，使用同一 `MambaACFAutoCon` 骨干、seed `2021`、global scope、未启用 ACF forecast memory 的配对 pilot。hard 行是 v2 兼容的 selected-positive 目标（`lag_mask=true`）；soft 行使用 `target_mode=soft_multi_peak`、`soft_min_abs=0.1`、`selected_lag_boost=1.0`（`lag_mask=false`，但 selected lags 仍作为 soft boost）。测试流指标和最后一个训练 epoch 的覆盖诊断如下：

| 目标 | scaled MSE | scaled MAE | global positive/anchor | candidate fraction | high-ACF excluded |
|---|---:|---:|---:|---:|---:|
| hard_selected (v3 compatibility) | 0.617183 | 0.400212 | 0.944 | 0.157 | 0.681 |
| soft_multi_peak (v3 semantic control) | **0.609619** | **0.397436** | 4.287 | 0.157 | 0.000 |

soft 相对 hard 的 scaled MSE 下降约 `1.23%`，但这只是单 seed、3 epochs、一个数据集/一个 horizon 的早期机制 pilot；两行都没有启用 forecast memory，不能与 memory-only 或旧 AutoCon 表比较。后续 30-run screen 中 soft 相对 hard 没有稳定精度增益，因此这组早期方向性差异不能据此声称 soft 目标有效。完整工件见 [`artifacts/mamba_acf_soft_relation_pilot_20260831/hard/`](../../artifacts/mamba_acf_soft_relation_pilot_20260831/hard/) 和 [`artifacts/mamba_acf_soft_relation_pilot_20260831/soft/`](../../artifacts/mamba_acf_soft_relation_pilot_20260831/soft/)。

### v3 30-run relation screen（五阶段均值，探索性）

新的 screen 共包含 `5` 个阶段、`2` 个数据集和 `3` 个 seed，即 `30` 个成功运行。每个
dataset-phase 单元的推断单位是 seed，`n=3`；`ALL` 合并行（见
[`mamba_acf_v3_screen_readout.md`](./mamba_acf_v3_screen_readout.md)）不是同一数据生成过程的
六次独立重复，也未做多重比较校正。下表是精确的均值 ± seed-level SD：

| 阶段 | ETTh1 MSE | Traffic MSE | ETTh1 MAE | Traffic MAE |
|---|---:|---:|---:|---:|
| plain_screen | 0.389664 ± 0.002807 | 0.570445 ± 0.003733 | 0.411648 ± 0.002041 | 0.389375 ± 0.002401 |
| available_memory_screen | 0.389005 ± 0.001678 | 0.555458 ± 0.005670 | 0.408272 ± 0.002361 | 0.384809 ± 0.003573 |
| hard_relation_nonoverlap_screen | 0.391957 ± 0.002591 | 0.565820 ± 0.008408 | 0.413486 ± 0.003919 | 0.384731 ± 0.004063 |
| soft_relation_nonoverlap_screen | 0.392103 ± 0.001567 | 0.568679 ± 0.004372 | 0.412781 ± 0.002968 | 0.388632 ± 0.008201 |
| available_memory_soft_relation_nonoverlap_screen | 0.395972 ± 0.005431 | 0.551952 ± 0.004095 | 0.414361 ± 0.006776 | 0.381977 ± 0.007107 |

相对 hard relation，soft relation 的 MSE 均值在 ETTh1 为 `+0.000146`（约 `+0.04%`），在
Traffic 为 `+0.002859`（约 `+0.51%`），方向并不一致且均未形成稳定收益。加入 soft relation
后再与 available memory 比较，ETTh1 为 `+0.006967`（约 `+1.79%`），Traffic 为
`-0.003506`（约 `-0.63%`）；Traffic 的方向有利，但 `n=3` 的 paired tests 仍不足以支持
稳定结论。该 screen 只能把 `soft_multi_peak` 定位为语义/机制控制，不能把它写成预测精度推荐。

资源工件中的 MambaACF 参数列是在 dead optional parameters 冻结前生成的历史快照；当前代码对 shared/plain 未使用的 `channel_pool`、关闭 memory 时的 `acf_gate` 不再计入 active trainable capacity。正式资源表需重新运行当前脚本，并区分 total 与 active trainable，不能把旧计数当作当前容量公平证据。

## Traffic 高维逐通道审计（后验描述性）

在 `_v3` 的 Traffic 862 通道汇总中，Mamba2 相对 PatchTST 的 scaled-MSE 差为 H96 `+29.205%`（8/862 通道更好）、H336 `+20.611%`（8/862）、H720 `+19.833%`（7/862）；5% trimmed 与 leave-one-out 仍保持劣势。Mamba2AutoCon 的方向也未改变。通道只是异质性描述，不是独立推断单位；该工件的 selection 为 `posthoc_verified`，所以它只能支持“共享 channel-mixing 表达存在系统性风险”，不能支持确认性显著性。ACF 后续若继续，应先验证逐通道/分组 lag bank 与 channel-preserving 表示，而不是继续增加共享时间深度。

## 已训练函数的 ACF memory 反事实依赖（只读）

在同一 checkpoint、同一测试流上仅屏蔽 `use_acf_memory` 的 source-snapshot 重放中，ETTh1 的 on/off 均值 MSE 为 `0.393167/0.457892`，Traffic 为 `0.564118/0.624881`；6 个 seed 的 off 均高于 on。输出差、gate 和 effective-memory RMS 均非零，因此可以确认这批已训练函数在预测时消费了 ACF residual branch。该结果不是 memory-on/off 独立重训的平均处理效应，不能替代配对重训、placebo 或确认性统计；完整哈希索引和逐 seed 结果见 [`mamba_acf_counterfactual_audit_20260903.md`](./mamba_acf_counterfactual_audit_20260903.md)。

## 论文门槛

在真实 Electricity/Traffic 上，用固定 validation gate 完成 plain、variable-token、memory、relation 四项消融，并同时报告：seed-paired delta、Holm/FDR 校正、通道级分布、参数量、峰值显存、端到端训练时间和 device-resident latency。当前统一资源复测（Traffic、B=8、H96、50 次 warm-up/200 次计时）中，plain MambaACF `0.684640 ms`、memory-only `0.808272 ms`、PatchTST `0.885824 ms`；它只说明 forward 负载，不能替代训练速度证据。变量 token screen 另见 [`mamba_acf_variable_mixer_screen_readout.md`](./mamba_acf_variable_mixer_screen_readout.md)，其结果仍是每数据集 3 个 seed 的 exploratory evidence。

完整 ACF baseline 定义与数据哈希见 [`acf-forecast-utility.md`](./acf-forecast-utility.md)；历史结果和 provenance 限制见 [`autocon_mamba_evidence_memo.md`](./autocon_mamba_evidence_memo.md) 与 [`mamba_current_status.md`](./mamba_current_status.md)。
## 当前核心实现补充：静态 ACF 可靠性插值

`MambaACF._build_static_acf_memory_terms` 已经把 train-only prior 编译为 signed coefficients、`step_reliability[H,C]` 和 `aggregate_reliability[C]`。新增候选在 `_static_acf_memory` 的 seasonal 累积之后应用一个有界的 `[H,C]` amplitude，仅使用这两个 detached 统计量和 causal validity；因此不会把当前输入或未来标签泄漏进 ACF 条件。

候选只有一个可训练标量 `acf_reliability_interpolation_scale`，以零初始化并通过 `tanh` 限幅。关闭时不创建有效参数；开启且标量仍为零时，seasonal memory 与旧实现保持嵌套。它与 stepwise reliability 互斥，避免两个分支同时重定义 reliability 语义；与 AutoCon relation loss 正交，relation 仍作用于表示空间。

实现和实验边界由以下契约共同约束：

- 模型、`run.py`、训练计划、阶段 validator 和资源 benchmark 都要求 static raw memory；
- 资源 architecture signature 显式记录 on/off，避免延迟记录合并；
- [`mamba_acf_reliability_interpolation_screen_plan_20260902.json`](mamba_acf_reliability_interpolation_screen_plan_20260902.json) 只登记未授权的 12-job H96 paired screen；
- 在训练完成前，不报告精度、显著性、泛化或发表结论。

## 损失与归因审计补充

当前 relation trainer 的双重归一化（trainer 外层与 `ACFRelationLoss` API 内层）在非零表示上近似幂等，现有 CPU/GPU probe 的 loss/梯度差异均低于 `4e-7`；保留两层是为了 API 边界和历史语义，不把它当作优化改进。真正需要校准的是 `AutoCon_lambda`：共享 trunk 上 `lambda=.03` 的 auxiliary 梯度约为 MSE 梯度的 `7.1%--12.4%`，方向接近正交；整模型梯度不能用于自动平衡，因为 forecast head 与 `repr_head` 的受监督参数集合不同。

现有日志里的 signed hinge 约占 auxiliary 数值 `0.04%`，所以 sign accuracy 不能支持“强 signed supervision”的机制表述。默认 `mamba_acf_repr_fusion=false` 时，relation loss 只看到 representation path，不能直接监督 forecast ACF memory/gate；representation fusion 必须单独做消融。普通 `DataEmbedding` 的 circular convolution 意味着默认配置不是严格全模型因果；即使启用 `mamba_causal_embedding=true`，还需单位 token stride、单尺度、非双向扫描、无窗口级 gate/非 raw 居中变换和 prefix-causal normalization。当前 `LastVal/Mean/ReVIN/MeanStd` 都从完整窗口取统计量，因此不能满足最后一项。完整数值和未授权验证计划见 [`mamba_acf_loss_scale_audit_20260902.md`](mamba_acf_loss_scale_audit_20260902.md) 与 [`mamba_acf_lambda_calibration_screen_plan_20260902.json`](mamba_acf_lambda_calibration_screen_plan_20260902.json)。
