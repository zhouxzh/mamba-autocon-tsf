# 外部 Mamba 时间序列结构审阅

**审阅日期：** 2026-08-31  
**用途：** 为本仓库的 Mamba + ACF + AutoCon 研究路线筛选可复核的设计先例；不是外部论文结果的复现或本仓库性能声明。  
**取证规则：** 仅使用论文原始页面、论文作者/项目页明确链接的 GitHub 仓库，或仓库自身明确标记为 official code 的来源。每个 GitHub URL 均固定到下表记录的 `HEAD` commit；外部声称的精度、效率未经本仓库同一数据合同复现前，不进入结果表。

**外部页面复核（2026-09-10）：** 本轮重新检查了作者仓库的公开 README。MambaTS 官方仓库仍将
`models/MambaTS.py`、`random_shuffle/unshuffle` 和 `mixer2_seq_simple.py` 分别作为架构、变量排列和
VAST 入口，并标记为 MIT；TimePro 官方仓库明确要求编译其 `selective_scan` 子目录；CMamba 官方仓库
标记为 MIT，并要求按其环境重新准备 ETT、Electricity、Traffic 和 Weather 数据。上述事实只支持
结构和复现依赖判断，不能把外部仓库的精度或速度数字填入本项目结果表。

## 结论

当前项目不应再把“普通的时间轴 Mamba + 线性头”当作候选论文主结构。现有受控结果已经表明：旧 Mamba 在 Electricity 上有一定竞争力，但在 Traffic 的三个 horizon 都明显落后于 PatchTST；而历史 `Mamba*AutoCon` 还保留了 AutoConNet 的预测头，不能称为纯 Mamba。

外部工作与该短板的直接对应关系如下：

1. **优先实现：低秩、数据依赖的跨变量调制。** CMamba 的 GDD-MLP、MambaMixer 的 channel mixer，以及 TimesMamba 的 series-as-token 都针对多变量关系；这是 Traffic 失败最有针对性的改进方向。
2. **第二优先：以 ACF 峰值控制多尺度，而不是复制不可观测的未来值。** ms-Mamba 的不同 `Delta` 分支提供了可借鉴的多尺度 SSM 形式。应由训练集的 signed residual ACF 选择/正则化尺度，且设置单尺度消融，不能先验宣称更快。
3. **AutoCon 保持为独立表示学习分支。** 预测状态与对比表征应分离，关系目标只从训练集 signed ACF 构造，且对无效 lag pair 做 mask。跨变量 mixer 不应替代这个研究要求。
4. **不直接移植外部源码。** 可直接移植的前提是许可证清楚、依赖兼容、没有测试集选择、并经本仓库 unit/smoke/资源基准验证。没有明确许可证的项目只可借鉴论文思想和重新实现。

### Mamba-1 / Mamba-2 / Mamba-3 的选择

对这个仓库，**Mamba-2 应继续作为候选主模型，Mamba-1 仅作为架构消融，Mamba-3 暂不进入正式矩阵。** 这是工程与证据边界的选择，不是关于任何版本普遍精度高低的断言。

| 版本 | 当前官方来源 / 本地可用性 | 本项目建议 |
| --- | --- | --- |
| Mamba-1 | [官方实现](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)，本环境 `mamba_ssm==2.3.1` 可直接导入 `Mamba` | 保留为历史可比、同配置的消融臂；不要继续以其为新增主干 |
| Mamba-2 | 同一官方实现；本环境可直接导入 `Mamba2`，已有 CUDA smoke 和现有模型接入 | 作为 P0--P5 的共享时间主干；Mamba-2 的 head dimension 和 expand 兼容性必须写入 config/provenance |
| Mamba-3 | 官方源码在上述 commit 已含 `Mamba3`，但官方 README 明确要求从最新源码安装；当前环境没有该 export，且本项目 Python 3.9 不满足官方 README 的 Python 3.10+ 要求 | 不升级现有正式环境，不将其作为“更快”的替代品。只有在独立环境、版本锁定、完全相同数据合同、Mamba-2 容量/速度对照和多 seed 结果齐备后，才可作为探索臂 |

官方 README 给出的 Mamba-3 示例使用 `bf16`、`d_model=768`、`d_state=128`、MIMO state update；这与本仓库当前小宽度（如 `d_model=32/64`）的长预测设置没有直接可迁移的速度结论。因此不能因“版本更新”而跳过 Mamba-2 基线。

推荐的下一架构是：

```text
training-only signed residual ACF
          |                         |
  selected lag/scale prior    lag-paired relation targets
          |                         |
input -> causal pointwise embedding -> shared temporal Mamba2 branches
                                      -> low-rank cross-variable gate
                                      -> direct multi-horizon head
                                      -> causal seasonal residual gate
                                          (only real historical values)
                                      -> forecast
                                      -> separate projection head -> ACFRelationLoss
```

这里的跨变量 gate 要以共享 summary 生成、逐变量小投影应用，避免 `C x C` 注意力或全变量 Mamba scan 在 Electricity/Traffic 上失去资源优势。ACF 只决定可解释的候选尺度、关系样本和季节残差可靠度；它不是把 `x[t-lag]` 在超出可用历史后重复/循环成未来观测。

## 当前仓库的对照点

| 现有实现 | 优点 | 需要补齐的部分 |
| --- | --- | --- |
| `Mamba1AutoCon` / `Mamba2AutoCon` | 已有 Mamba encoder、训练集 ACF prior、AutoCon trainer 合同 | 预测头仍是 AutoConNet trend/seasonal head；全局 ACF 关系并非逐通道且历史实现偏向 top-1 正样本 |
| `MambaACF` / `MambaACF1` | 共享时间扫描、直接 multi-horizon 头、signed prior、causal-indexed memory mask、预测/对比表征分离 | 尚无正式效果结果；默认通道混合仍是输入 embedding；当前多尺度是输入池化，不是 Delta-space 机制 |
| `ACFRelationLoss` | 可保留 signed ACF、屏蔽超出测量范围的 pair | 必须搭配连续时间 batch 或显式 lag-paired sampler；随机 batch 会大幅降低有效全局 pair 比例 |

因此，下表中的“迁移”均指在本仓库重新实现成小模块，不是复制项目整体。

**因果编码前置条件。** 当前共享 `DataEmbedding` 的 `TokenEmbedding` 是 kernel size 3、`padding_mode='circular'` 的卷积。它不读取预测区间的 label，但一个窗口的首端可以使用该窗口末端输入；因此推荐图中的 “causal pointwise embedding” 是待实现目标，不是当前实现。任何把 ACF memory/lag branch 解释为严格因果的正式实验，都必须同时采用点式线性 embedding 或左填充的 causal convolution，并把 circular-embedding 版本作为独立对照。

## 候选结构比较

| 方法 / 原始来源 | 固定版本与许可 | 输入、通道与时间结构 | 预测头 / 理论复杂度 | 可迁移模块 | 主要风险与本仓库判定 |
| --- | --- | --- | --- | --- | --- |
| **MambaTS** [论文](https://arxiv.org/abs/2405.16440), [官方仓库](https://github.com/XiudingCai/MambaTS-pytorch/tree/3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d) | `3b6797d`，MIT；仓库 README 明确标为 official code | 每变量 patch 后形成 `B x (C*P) x D` token 序列；VPT 在训练时随机排列变量，验证时以 ATSP 代价排序；VAST 把变量-时间 token 串行扫描 | flatten 全部 patch representation 后每变量线性到 horizon；选择性扫描对 token 长度近线性，但 token 数为 `C*P` | patch token 化；变量排列作为**单独、预注册的**鲁棒性消融 | 大 C 时 token 数暴涨，ATSP 排列引入复杂且可能不稳定的训练/推理差异；其 vendored Mamba 版本与当前 `mamba_ssm==2.3.1` 不同。**不作为第一实现。** |
| **S-Mamba** [论文](https://arxiv.org/abs/2403.11144), [作者仓库](https://github.com/wzhwzhwzh0921/S-D-Mamba/tree/e7e8bf04066135afa43d85b0a87afa97cda16e3f) | `e7e8bf0`，无明确开源许可证 | `DataEmbedding_inverted` 将每个变量视为 token，双向 Mamba 沿变量维扫描，FFN 处理 token representation | 逐变量 `D -> H` 线性头；每个 Mamba scan 随变量数近线性，不是沿历史时间作 scan | “变量作为 token”的 cross-channel 对照实验 | 当前仓库的问题是 Traffic 高维跨变量建模，值得作为对照；但 bidirectional variable scan 对变量排列敏感，且无许可证。**仅重实现最小变量 mixer 消融。** |
| **ms-Mamba** [论文](https://arxiv.org/abs/2504.07654), [作者仓库](https://github.com/meric-karadag/ms-mamba/tree/24c5e3eb5eff2bc968f37814a0e44e837d8302c1) | `24c5e3e`，无明确开源许可证；README 标为 Neurocomputing 2026 official implementation | 基于 S-Mamba 的变量 token 输入；并行分支使用相同输入但不同 SSM sampling rate `Delta`，输出均值融合 | 线性投影到 horizon；若有 `S` 个分支，主 SSM 工作约为单分支的 `S` 倍，不能直接主张加速 | **固定少量 Delta multiplier**，由 train-only ACF 峰/周期范围选取或正则化；使用 gated/weighted fusion 而非无条件均值 | 动态 Delta 额外 MLP 和多分支会增加训练成本；源码自己的 Mamba fork 与当前版本不兼容。**是多尺度 ACF-Mamba 的最强结构先例，但先做 2-scale 和单尺度消融。** |
| **CMamba** [论文](https://arxiv.org/abs/2406.05316), [官方仓库](https://github.com/zclzcl0223/CMamba/tree/78166596a235fa1565c5b84d034216dc8242c84b) | `7816659`，MIT | 每变量 patch + temporal Mamba；每层可将 `[B,C,P,D]` 输入 GDD-MLP，以 pooled feature 生成 data-dependent scale/shift 的跨变量调制；配合 Channel Mixup | flatten patch 后逐变量线性头；temporal scan 近线性，GDD-MLP 仍含随 `C` 的全连接，约 `O(C^2/r)` 参数/计算 | **低秩 global-summary -> per-channel FiLM/gate**；训练期 channel mixup 作为可关闭的鲁棒性消融 | 原版 GDD-MLP 在 Traffic `C=862` 时仍可能昂贵，且 `C//reduction` 有数据集特定形状。**优先改写成低秩/分组 gate，而不是复制全量 GDD-MLP。** |
| **MambaMixer / TSM2** [论文](https://arxiv.org/abs/2403.19888), [项目页](https://mambamixer.github.io/), [作者链接的代码仓库](https://github.com/MambaMixer/M2/tree/250a86b9ff8bbb986b132c4e07d7aa8fd8b83f67) | `250a86b`，无明确许可证；公开 repo 在该 commit 只有 README/图像，**没有可审阅的模型源码** | 论文：先后以 selective SSM 做 token mixer 和 channel mixer，并对早期层特征做加权平均；TSM2 的时间 mixer 采用单向因果 SSM | 论文声明 token/channel 均使用线性扫描；总开销取决于两个维度的 scan 和层间融合 | **双轴选择性混合的研究假设**：时间 Mamba + 小型数据依赖 channel gate；学习的层间融合 | 没有可核查源码与许可证，不能复制；对 `C=862` 做完整 channel scan 可能过慢且变量顺序敏感。**概念先例，不纳入直接基线。** |
| **TimeMachine** [论文](https://arxiv.org/abs/2403.09898), [官方仓库](https://github.com/Atik-Ahamed/TimeMachine/tree/5bf17a728349c674eceafc28a90f22b904e38c2d) | `5bf17a7`，Apache-2.0 | 将每变量完整 lookback 线性映射至两种宽度；四个 Mamba 分支交替处理嵌入和其转置，提供 channel-independent / mixed 选项 | 拼接两个 latent，再线性预测 horizon；四个 Mamba 分支，实际常数开销较高 | 双方向/双轴 residual 连接的消融思路；RevIN 对照 | 不是轻量主干，且四个 Mamba 可能损失速度优势；完整 lookback projection 强依赖 `seq_len`。**仅作为容量匹配的结构对照，不应作为效率方案。** |
| **TimesMamba** [仓库](https://github.com/shaowen310/TimesMamba/tree/f52ccaf34cf1de17e177483a246943909699f833) | `f52ccaf`，`NOASSERTION`；作者维护仓库，但未核实同行评审论文为同一版本 | RevIN；整条变量历史映射为 series embedding，即变量 token；双向 Mamba + FFN 在变量轴工作；可加入时间标记 | token `D -> H` 线性头；变量轴 scan 近线性 | temporal marks 作为外生变量的明确消融；变量 token mixer 的轻量对照 | 将时间压成一个 token 后不再通过 Mamba 建模时间细节；外部时标在部分数据集可有害。**可借鉴数据合同，不作为主结构。** |
| **Affirm** [AAAI 2025 论文](https://ojs.aaai.org/index.php/AAAI/article/download/35463/37618), [作者代码](https://github.com/zjuml/Affirm/tree/53cfa465e73dd57cdab29cd04d5e2ec69e1e57da) | `53cfa46`，无明确许可证 | patch embedding；Adaptive Fourier Filter 先做频率过滤，dual interactive Mamba 再做多粒度交互 | flatten patch 头；其实现带有较大外部工程和自定义 Mamba | 用频域噪声/季节性滤波作为**ACF 前处理的竞争对照** | 复杂度和工程依赖显著，且频域门控会与本文 ACF 贡献混淆。**不直接集成；可作为 ACF-vs-frequency-ablation 的外部基线。** |
| **AdaMamba** [预印本](https://arxiv.org/abs/2604.23239), [作者仓库](https://github.com/XDjiang25/AdaMamba/tree/8f10f4dc88a6d06dd3ac8354ca757fd2893773a4) | `8f10f4d`，无明确许可证；2026 preprint | 多尺度 patch；用频率门控 Mamba / correlation convolution | flatten patch 头；架构在频率状态中加入自适应门控 | “频率/周期先验进入 state update”这一**长期研究方向** | 预印本、未明确许可证、实现复杂且与 ACF 新颖性高度重叠。**暂不采用，论文相关工作中讨论即可。** |

### 未纳入可实现清单的来源

- **DTMamba** 有原始预印本 [arXiv:2405.07022](https://arxiv.org/abs/2405.07022)，但本次检索未找到作者公开、可验证的官方 GitHub 仓库；不把其代码/结构作为依赖来源。
- **How to Train Your Mamba for Time Series Forecasting** 已以 [KDD 2026 DOI](https://doi.org/10.1145/3770854.3780279) 发表，但截至本次审阅未发现作者公开源码。它应在论文写作时用于方法学背景核查，不可作为可复制实现。
- 外部模型声称的“更快”往往只在其特定 GPU、batch、lookback、precision 下成立。本项目只有在相同输入形状、同一 RTX 5090 D、固定 warmup/timed iterations 下由 `scripts/benchmark_model_resources.py` 测得后，才能写入资源结论。

## 选择理由和迁移边界

### 1. 为什么 CMamba 类 gate 是第一优先

当前 MambaACF 默认共享时间轴 Mamba scan，这在高维时避免了 `B*C` 次 scan，但输入 `DataEmbedding` 后缺乏显式、数据依赖的跨变量模块。Traffic 结果落后 PatchTST 的模式提示该缺口比再增加时间深度更可信。

建议实现的版本不是原始全连接 GDD-MLP，而是：

```text
z[b,c,t,d] -> pool_t,d -> summary[b,c]
summary -> low-rank adapter (C -> r -> C) -> scale[b,c], shift[b,c]
z' = (1 + alpha * tanh(scale)) * z + alpha * shift
```

约束：

- `r` 必须显式配置，并报告 `C,r`；高维默认应小，例如 8/16/32，而不是随 `C` 增长。
- `alpha` 零初始化或小值初始化，保证 plain MambaACF 是嵌套基线。
- 只能从当前输入窗口计算 summary；不能从 label、forecast target 或 batch 索引读取信息。
- 需要 plain / static channel MLP / low-rank data-dependent gate 三臂消融，才能证明“data-dependent”而不是“多参数”带来的差异。

### 2. ACF 如何控制 ms-Mamba 式多尺度

ms-Mamba 的关键思想是用不同 `Delta` 处理同一输入，而不是对输入做丢失信息的下采样。与本项目结合时，训练集 residual ACF 应提供**弱先验**而不是硬编码周期：

1. 在每个训练 split 单独拟合 signed residual ACF，排除 lag 0，记录峰值、置信阈值和 train hash。
2. 根据可靠峰值的短/中/长周期范围选择预注册的 `Delta` multiplier 候选，例如 `(1, 2)` 或 `(1, 2, 4)`；不得查看 test 后选取。
3. 先采用两条分支和可学习 softmax gate。只有在效果/效率都通过后，才研究三到四个分支。
4. 对长 horizon，ACF 主要指导状态尺度和关系学习；直接 seasonal-memory residual 只使用严格因果可用的历史索引。不可用的 lag 必须被 mask，而不是把模型已预测值或周期循环值伪装成观测。

当前 `MambaACF` 的输入池化尺度与这里的 `Delta` 分支不是同一件事，论文中必须分别命名并消融。

### 3. AutoCon/表示学习应如何保留

候选主模型必须同时输出：

```text
forecast_state -> forecast head -> y_hat
forecast_state -> projection head -> representation -> signed ACF relation loss
```

关系监督应遵守：

- 训练集 only 的 ACF prior；验证/测试段不能参与 peak 选择或关系 target。
- 保留 ACF 符号，正相与反相关系不能在 `abs()` 后混为同类。
- 使用 lag-paired sampler 或连续时间 mini-batch；随机 shuffle batch 中的远距离 pair 不应被 clamp 到最大 ACF lag。
- 每轮记录有效 pair coverage、local/global relation loss、forecast MSE；coverage 太低的结果不可归因于关系学习。
- 预测头、跨变量 gate、ACF memory、关系损失必须逐项消融，且所有模型使用相同训练预算与 early-stop 规则。

## 实施优先级

| 顺序 | 交付物 | 必要验证 | 通过条件 |
| --- | --- | --- | --- |
| P0 | 低秩数据依赖 cross-variable gate，嵌入 MambaACF | shape、梯度、C=1 等价/可用性、C=862 smoke | 默认 gate 初始化不劣化 plain 输出合同；无 `C^2` 激增 |
| P1 | 无 circular wrap 的 causal embedding | 端点扰动测试、输入可达域测试、与旧 embedding 形状兼容 | 输入第 0 步绝不依赖窗口末端；它是所有“因果 ACF”结论的共同前置条件 |
| P2 | 连续时间/lag-paired sampler + relation coverage 日志 | sampler determinism、所有 target lag 可追溯、loss mask 测试 | 训练 batch 内有效 pair coverage 接近设计值，绝不 clamp 越界 lag |
| P3 | 两分支 ACF-scale Mamba | 单分支等价、Delta 参数、Mamba1/Mamba2 GPU smoke | 在相同模型宽度下先报告参数、延迟和显存，不声称效率 |
| P4 | 预注册 selection + confirmatory 结果矩阵 | plan validator、source snapshot、5 seeds、资源脚本 | 相比强 baseline 在每数据集/horizon 报告配对差、Holm/FDR 与速度；不能只汇总平均数 |
| P5 | 外部结构对照 | 统一数据划分和调参预算 | 至少含 PatchTST、DLinear、TimesNet、plain Mamba2、Mamba2+AutoCon、候选主模型；Traffic/Electricity 必须纳入 |

## 最小对照矩阵

为区分每个贡献，正式训练至少应包括：

| Arm | 时间主干 | 跨变量机制 | 多尺度 | ACF 预测先验 | AutoCon/关系学习 |
| --- | --- | --- | --- | --- | --- |
| A | Mamba2 | 无 | 单尺度 | 无 | 无 |
| B | Mamba2 | 低秩 gate | 单尺度 | 无 | 无 |
| C | Mamba2 | 低秩 gate | ACF-scale | 无 | 无 |
| D | Mamba2 | 低秩 gate | ACF-scale | causal residual gate | 无 |
| E | Mamba2 | 低秩 gate | ACF-scale | causal residual gate | signed ACF relation loss |
| F | E | 同 E | 同 E | ACF prior 随机置换 | 同 E |
| G | E | 同 E | 同 E | 无/zero ACF | 同 E |

F/G 是必要的否证对照：若随机 ACF 与真实 ACF 同样有效，则不能声称模型利用了自相关结构。

## 可发表性边界

外部工作数量不构成论文贡献。能支持高水平投稿的最小证据不是“某个 Mamba 结构在一个数据集跑得更低”，而是：

1. 在 ETT、Electricity、Traffic 以及新增、数据合同清楚的独立数据集上，对强现代基线不差；
2. 在同一精度等级，报告可复现的端到端训练时间、峰值显存和推理吞吐优势，而非仅理论 `O(L)`；
3. signed ACF、因果 memory、跨变量 gate、对比表示学习各自有预注册消融和失败对照；
4. 多 seed 配对统计经多重比较校正后仍有稳定结果，且 selection/confirmatory/test 不混用；
5. 所有数据、源码快照、配置、环境和资源基准可以人工复核。

目前这五项尚未全部满足，因此本报告是架构筛选依据，不是发表准备完成声明。

## 版本取证附录

| 仓库 | 观察到的 HEAD | 最近推送/提交（UTC） | 许可状态 |
| --- | --- | --- | --- |
| [MambaTS](https://github.com/XiudingCai/MambaTS-pytorch) | `3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d` | 2024-10-02 | MIT |
| [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) | `e7e8bf04066135afa43d85b0a87afa97cda16e3f` | 2025-05-20 | 未声明 |
| [TimeMachine](https://github.com/Atik-Ahamed/TimeMachine) | `5bf17a728349c674eceafc28a90f22b904e38c2d` | 2024-07-27 | Apache-2.0 |
| [CMamba](https://github.com/zclzcl0223/CMamba) | `78166596a235fa1565c5b84d034216dc8242c84b` | 2025-01-06 | MIT |
| [MambaMixer/M2](https://github.com/MambaMixer/M2) | `250a86b9ff8bbb986b132c4e07d7aa8fd8b83f67` | 2024-03-31 | 未声明 |
| [TimesMamba](https://github.com/shaowen310/TimesMamba) | `f52ccaf34cf1de17e177483a246943909699f833` | 2025-05-02 | `NOASSERTION` |
| [ms-Mamba](https://github.com/meric-karadag/ms-mamba) | `24c5e3eb5eff2bc968f37814a0e44e837d8302c1` | 2026-07-30 | 未声明 |
| [Affirm](https://github.com/zjuml/Affirm) | `53cfa465e73dd57cdab29cd04d5e2ec69e1e57da` | 2025-01-16 | 未声明 |
| [AdaMamba](https://github.com/XDjiang25/AdaMamba) | `8f10f4dc88a6d06dd3ac8354ca757fd2893773a4` | 2026-05-05 | 未声明 |

“未声明”与 `NOASSERTION` 不等于可自由复制。它们只允许本项目进行独立的学术性结构比较；任何复制、派生或发布代码前都必须取得明确授权或重新实现不受版权表达保护的高层思想。

## 2026-09-10 MambaTS v5 anchor screen

为检验“直接利用 ACF 结果改善预测”这一方向，仓库新增了 `MambaTSv5AutoCon`。它严格嵌套在 v4：保留 patch/shared-scan、低秩跨变量 context、AutoCon 表示路径和 train-only ACF memory，只增加一个零初始化的 signed ACF anchor。每个预测步只从训练集 ACF 选出的可用最大绝对相关 lag 读取观测窗口中的历史值；不可用早期步被 mask，不使用预测值回填。

在四个 PEMS 数据集、H12/H96、两个 seed 的 16 个验证坐标上，v5 相对 v4 的 H12 胜率为 8/8，但平均变化仅 `-0.0080%`；H96 胜率为 4/8，平均变化为 `+0.0017%`。anchor gate 在 16/16 行非零，但 H12 平均绝对 gate 只有 `0.001569`。v5 平均每 job `18.199s`，v4 为 `18.089s`。因此，这个结构目前应作为已实现且可审计的中性消融，不应作为主论文贡献或效率改进证据。完整记录见 [`mamba_pems_ts_v5_anchor_screen_20260910.md`](mamba_pems_ts_v5_anchor_screen_20260910.md)。
