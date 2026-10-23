# Mamba 时间序列结构筛选报告

**审阅日期：** 2026-08-31  
**目的：** 为本仓库的 `Mamba + ACF + AutoCon` 路线筛选可复核的外部结构先例，并给出下一轮公平实验合同。本文不把外部论文报告的精度或速度直接当作本仓库结果。

## 先看结论

当前仓库的主要问题不是 Mamba 扫描本身太慢，而是高维多变量关系和局部片段表示不足。已有 5-seed 高维结果中，Mamba-2 在 Electricity 的 H96/H336/H720 为 `0.183130/0.211477/0.241182`，PatchTST 为 `0.184448/0.207953/0.252428`；Traffic 为 `0.648306/0.623489/0.659914`，PatchTST 为 `0.501767/0.516942/0.550696`。因此下一步应优先补充跨变量建模和 patch 表示，而不是继续堆叠时间轴 Mamba 层。

推荐的主线结构为：

```text
train-only signed residual ACF
       |                         |
  lag/scale prior          lag-paired AutoCon targets
       |                         |
causal pointwise embedding -> patch temporal Mamba-2
                            -> low-rank data-dependent channel gate
                            -> direct multi-horizon head
                            -> optional causal ACF residual gate
                            -> forecast
                            -> separate projection head -> AutoCon/ACF relation loss
```

跨变量模块必须是低秩或分组形式，避免在 Traffic (`C=862`) 上引入完整 `C x C` 矩阵。ACF 只作为训练集先验、尺度选择和关系目标；不能把不可观测的未来值复制成“季节记忆”。

## 外部结构核验

| 结构 | 可核验来源 | 核心设计 | 对本仓库的价值 | 采用边界 |
| --- | --- | --- | --- | --- |
| **MambaTS** | [论文](https://arxiv.org/abs/2405.16440)，[官方代码](https://github.com/XiudingCai/MambaTS-pytorch) | patch 后构造变量-时间 token；VAST 学习变量关系并在推理时解码扫描顺序；TMB 去掉对独立变量不必要的 causal convolution，并对 selective 参数 dropout | 直接针对 Traffic 的变量关系、扫描顺序和 circular/causal embedding 问题 | **P0 对照/重实现思想**。原仓库使用 vendored Mamba 和 ATSP，不能直接混入当前依赖；必须固定排序算法和训练期信息来源 |
| **TimePro** | [ICML 2025 论文/代码](https://openreview.net/forum?id=s69Ei2VrIW)，[官方代码](https://github.com/xwmaxwma/TimePro) | variable- and time-aware hyper-state；变量关系与时间特征共同调节 Mamba 状态，使用自定义 selective-scan/CUDA | 最贴合“高维变量异质性”问题；官方脚本覆盖 ECL、ETT、Weather、Exchange、Solar | **P0 外部强基线/概念参考**。依赖 `DCNv4` 和自定义 CUDA，不能直接宣称更快；先独立环境或重实现低秩 hyper-state |
| **CMamba** | [论文](https://arxiv.org/abs/2406.05316)，[官方代码](https://github.com/zclzcl0223/CMamba) | patch temporal Mamba + GDD-MLP；将 `[B,C,P,D]` 汇总后生成 data-dependent channel scale/shift，另有 channel mixup | 提供可解释的跨变量调制机制，直接对应 Traffic 的失败模式 | **P0 最小迁移**：改写为 `summary -> C->r->C -> scale/shift`，`r=8/16/32` 预注册；不复制原版可能随 `C` 增长的全连接 GDD-MLP |
| **Bi-Mamba+** | [论文](https://arxiv.org/abs/2404.15772)，[官方代码](https://github.com/take3210/Bi-Mamba4TS-for-time-series) | Mamba+ forget gate；输入 patch 的正向/反向双分支；SRA decider 在 channel-independent 与 channel-mixing 间选择 | 可检验“保留历史状态”和双向观察窗口是否帮助长 horizon | **P1 消融**。双分支约增加扫描和显存；反向只允许扫描已观测输入，不应解释成使用未来目标 |
| **S-Mamba** | [论文](https://arxiv.org/abs/2403.11144)，[作者代码](https://github.com/wzhwzhwzh0921/S-D-Mamba) | 每变量先线性 token 化；双向 Mamba 沿变量轴提取跨变量关系，FFN 学时间映射 | 简单的 variable-token cross-channel 对照 | **P1 对照**。变量顺序敏感且仓库未见明确许可证；不能替代时间轴 Mamba 主干 |
| **ms-Mamba** | [论文](https://arxiv.org/abs/2504.07654)，[项目说明](https://www.sciencedirect.com/science/article/pii/S0925231226006235) | 多个 Mamba 分支使用不同 `Delta`，并行建模不同时间尺度 | 提供比输入池化更严格的多尺度 SSM 先例，可由 ACF 峰值弱引导 | **P2 重实现**。公开代码可复核性不足；先做两个固定/可学习 Delta 分支，并与单尺度等预算比较 |
| **TimeMachine** | [论文](https://arxiv.org/abs/2403.09898)，[官方代码](https://github.com/Atik-Ahamed/TimeMachine) | 两种 latent 尺度、channel-mixing/independence 组合成四个 Mamba 分支，并用残差融合 global/local context | 说明多轴、多尺度 Mamba 的设计空间 | **结构参考，不作为速度主线**。四分支常数开销较大，且完整 lookback projection 依赖固定 `seq_len` |
| **MambaMixer/TSM2** | [论文/项目页](https://mambamixer.github.io/) | selective token mixer + selective channel mixer，并用层间加权融合 | 支持“时间 Mamba + 数据依赖通道模块”的研究假设 | **概念参考**。公开页面可核验论文主张，但当前作者仓库缺乏可审阅模型源码和明确许可证，不直接复制 |
| **Mamba4Cast** | [论文](https://arxiv.org/abs/2410.09385)，[官方代码](https://github.com/automl/Mamba4Cast) | 在合成数据上训练 PFN 式零样本模型，一次前向生成整个 horizon | 可作为将来 zero-shot/预训练方向的背景 | **不纳入监督基线矩阵**。训练范式、数据和目标与本仓库不同，外部速度不能迁移 |
| **DuoMamba** | [OpenReview 论文](https://openreview.net/pdf?id=9payEItr83) | 极简 decomposition-free Mamba-2，chunk-aligned patch token，专门检验是否需要复杂分解 head | 可作为“纯 Mamba-2 + patch head”负对照，帮助识别 AutoCon head 是否真正贡献 | **探索对照**。目前是近期预印本，未作为本仓库依赖或主论文证据 |

### 重要的外部实现事实

- MambaTS 官方 README 明确指出：架构在 `models/MambaTS.py`，VPT 在 `random_shuffle/unshuffle`，VAST 在 `mixer2_seq_simple.py`；这不是普通的单轴 Mamba。
- CMamba 的实际代码顺序是 `RMSNorm -> selective Mamba -> optional GDDMLP -> dropout -> residual`，GDD-MLP 接收 `[B,C,P,D]`，不是一个普通的静态 channel MLP。
- TimePro 的官方脚本使用 `d_model=32`、patch_len 8、stride 4，并编译自定义 `selective_scan`；它的速度必须在本地同硬件、同精度、同 warmup 协议重新测量。
- Bi-Mamba+ 的官方 README 声称在八个数据集上比较了 SRA、双向、forget gate 和 residual，但其环境要求单独编译 `mamba_plus`；不能直接与当前 `mamba_ssm==2.3.1` 的 Mamba-2 数字混表。
- 外部论文的“领先/更快”均是其作者的数据划分、超参数和硬件合同下的结果。进入本仓库主表前必须重新训练并保存配置、源码 commit、环境和资源计时。

### 版本取证（2026-08-31 查询）

下表只记录查询时 `main`/默认分支的 commit；后续复现实验必须把 commit、依赖锁定文件和编译器信息一并保存。`未声明` 或 `NOASSERTION` 不等于允许复制代码，本文只借鉴论文中的高层结构。

| 项目 | 查询到的 commit | 许可证/依赖备注 |
| --- | --- | --- |
| [MambaTS](https://github.com/XiudingCai/MambaTS-pytorch) | `3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d` | MIT；Python 3.11，vendored Mamba/ATSP |
| [CMamba](https://github.com/zclzcl0223/CMamba) | `78166596a235fa1565c5b84d034216dc8242c84b` | MIT；自带 selective-scan 实现 |
| [TimePro](https://github.com/xwmaxwma/TimePro) | `70a20e5a257b30eb026ee4316293cf4feeb92a1f` | 仓库未见明确许可证；`DCNv4` 与自定义 CUDA selective-scan |
| [Bi-Mamba+](https://github.com/take3210/Bi-Mamba4TS-for-time-series) | `17865287c8e7d765e6e9f62c279b2549a3267688` | 仓库未见明确许可证；需要单独编译 `mamba_plus` |
| [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) | `e7e8bf04066135afa43d85b0a87afa97cda16e3f` | 仓库未见明确许可证 |
| [TimeMachine](https://github.com/Atik-Ahamed/TimeMachine) | `5bf17a728349c674eceafc28a90f22b904e38c2d` | Apache-2.0 |
| [Mamba4Cast](https://github.com/automl/Mamba4Cast) | `f11d0df579fdcecdc69f463fdf7a8091eb0cf64d` | MIT；零样本合成数据训练，不能当监督基线 |
| [MambaMixer/M2](https://github.com/MambaMixer/M2) | `250a86b9ff8bbb986b132c4e07d7aa8fd8b83f67` | 公开仓库缺少可审阅模型源码/明确许可证 |

## 针对当前失败模式的落地优先级

### P0：低秩数据依赖跨变量 gate

从 CMamba/TimePro/MambaMixer 的共同思想抽取最小模块：先沿时间和 hidden 维做 summary，再用低秩映射生成每变量的 scale/shift：

```text
z[b,c,t,d] -> pool(t,d) -> s[b,c]
s -> Linear(C,r) -> GELU -> Linear(r,C)
z' = (1 + alpha*tanh(scale)) * z + alpha*shift
```

`alpha` 零初始化，使 plain MambaACF 成为严格嵌套基线；summary 只能来自当前输入窗口。Traffic 应先测 `r=8,16,32`，并报告参数量和额外 MAC，而不是只报告 MSE。

### P1：patch temporal encoder + causal pointwise embedding

MambaTS 的 patch token 和 TMB 对当前实现很关键。当前仓库默认 `TokenEmbedding` 使用 circular Conv1d，因此不能宣称整网严格因果。正式 ACF 实验应先切换到点式线性 embedding 或左填充 causal convolution，并加入端点扰动测试。

### P2：ACF 引导的两尺度 Delta 分支

用 train-only signed residual ACF 选出短/中周期候选，只决定两个预注册 Delta multiplier（例如 `(1,2)`），不硬编码未来观测。先比较：单尺度、固定双尺度、可学习 gated 双尺度；三/四分支和动态 Delta 只有在精度和资源都通过后再做。

### P3：AutoCon 表示学习保持独立

预测路径和表示路径应明确分离：

```text
forecast state -> direct horizon head -> prediction
forecast state -> projection head -> signed ACF relation loss
```

关系 loss 必须使用训练集 ACF、合法 lag mask 和 lag-paired sampler；每轮记录 global/local active pairs、每个 lag 的计数、forecast MSE。若随机 ACF 或 zero ACF 与真实 ACF 同样有效，就不能声称模型真正利用了自相关。

## 公平实验合同

### 数据集层级

第一层复核现有九个公开数据集：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic、Weather、Exchange、National Illness（ILI 可作为小样本压力测试）。第二层新增 PEMS03/04/07/08 或 Solar-Energy 时，必须保存原始文件 hash、时间频率、缺失值处理和官方 split；不能只下载结果表。

### 模型矩阵

| Arm | 时间主干 | 跨变量 | ACF 预测模块 | AutoCon |
| --- | --- | --- | --- | --- |
| A | Mamba-2 | 无 | 无 | 无 |
| B | Mamba-2 + patch | 无 | 无 | 无 |
| C | B | 低秩 data-dependent gate | 无 | 无 |
| D | C | 同 C | 固定双尺度 Delta | 无 |
| E | D | 同 C | causal residual memory/gate | 无 |
| F | E | 同 C | 同 E | signed ACF relation |
| G | F | 同 C | ACF lag 随机置换或 zero | 同 F |
| H | PatchTST / TimesNet / DLinear / iTransformer | 原实现 | 无 | 无 |
| I | MambaTS 或 TimePro（重实现/独立环境） | 外部方法 | 按原方法 | 无 |

G 是必要的否证对照，H 是强现代基线，I 是外部 Mamba 对照。Mamba-1 只保留为与 Mamba-2 同容量的消融，不在主结论中替代 Mamba-2。

### 固定参数与统计

- 长期预测先统一 `seq_len=96`、`pred_len={96,336,720}`；ETTm 可额外报告 `192`，但不能只挑有利 horizon。
- Mamba-2 固定 `d_state=16`、`d_conv=4`、`expand=2`，`d_model` 与 PatchTST/候选模型按 active trainable 参数或明确的两档预算匹配。
- 所有 arm 显式固定 optimizer、learning rate、weight decay、batch、epochs、early-stop 和 temporal marks；不能沿用 plain/AutoCon 默认值差异。
- 每个数据集/horizon 至少 5 个预注册 seed；validation 只用于 checkpoint/超参选择，test 只在最后一次读取。
- 报告 MSE、MAE、seed-paired delta、95% 区间、Holm/FDR 校正；不要只给平均排名。
- 同一 RTX 5090 D、同 dtype、同 batch、同输入 shape、统一 warmup/timed iterations，同时测单步 latency、吞吐、峰值显存和端到端训练时间。理论 `O(L)` 不等于实测更快。

## 研究判断

外部 Mamba 结构可以提供下一轮有价值的机制假设，但“引用了更多 Mamba 论文”本身不会形成高水平论文贡献。当前最有希望且仍与 AutoCon 主题一致的路线是：

1. 用 patch/TMB 修正时间输入表示；
2. 用低秩数据依赖 gate 补足 Traffic 的跨变量异质性；
3. 用训练集 signed ACF 选择尺度和合法 memory；
4. 用独立 projection 保留 AutoCon 的表示学习，并对 relation coverage 做可审计统计；
5. 在统一预算下证明精度至少不劣于 PatchTST，同时用端到端资源而不是理论复杂度证明速度优势。

在上述统一矩阵和多 seed 结果完成前，不能把本项目定位为已达到高水平投稿条件；本报告只是下一轮实现和实验的证据边界。
