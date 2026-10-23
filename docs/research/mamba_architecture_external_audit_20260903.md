# Mamba 时间序列结构外部核验与 AutoCon 适配审计

**核验日期：** 2026-09-03  
**范围：** 公开论文及其作者/组织官方 GitHub 仓库；只做源码和论文审计，不把外部数字当作本仓库结果。  
**本仓库接口：** `models/MambaACF.py`、`exp/exp_long_term_forecasting_with_mamba.py`、`layers/losses.py`。

## 结论先行

当前仓库最需要补的是高维多变量的跨通道信息流，而不是再堆叠更多独立的时间 Mamba。外部工作可以按下列优先级借鉴：

1. **CMamba 风格的单次低秩、数据依赖跨通道 mixer：优先候选。** 它直接针对 vanilla Mamba 的跨通道短板，且可以保持参数量对通道数近似线性/受控。
2. **MambaTS 风格的变量顺序学习：第二候选。** 训练中学习变量转移代价，推理时冻结变量顺序；必须把顺序作为 train-only sidecar，不能在测试窗口重新估计。
   当前树另提供 `mamba_acf_channel_order=similarity` 作为独立低成本
   ablation：它使用完整 signed selected-lag profile 的 cosine-seriation，
   不是 MambaTS 的 loss-driven pairwise adjacency，也不是 cross-correlation 图。
3. **固定多尺度 SSM 时间尺度（ms-Mamba 风格）：第三候选。** 可与当前多尺度状态融合兼容，但先使用固定/少量 learnable scales，不采用输入依赖的 dynamic scales。
4. **TimeMachine 的两阶段压缩、残差和局部/全局互补：可拆分借鉴。** 不建议原样复制四个 Mamba，特别是在 Traffic (`C=862`) 上。
5. **MambaMixer 的 dual token/channel selection：概念参考。** 官方仓库在固定 commit 仍写明代码和模型“will be available soon”，因此不能作为可复现实现依赖。
6. **Mamba4Cast：暂不纳入当前 supervised 主线。** 它是 synthetic pretraining/zero-shot foundation model，训练协议与当前 train-only ACF + AutoCon 目标不同。

这份审计不支持“外部论文已经证明某个结构会提高本仓库精度”。每个候选都必须在相同数据切分、seed、训练预算、参数量/吞吐测量和现有 AutoCon 目标下重新验证。

## 证据等级

| 等级 | 含义 |
| --- | --- |
| A | 已发表/正式接收论文 + 作者官方代码，且源码路径可核对 |
| B | arXiv/会议论文 + 作者官方代码，设计和实现可核对，但仍需独立复现实验 |
| C | 论文或项目页有明确设计，但官方实现缺失、未冻结或无法直接复现 |
| D | 二手综述/非官方实现，只能用于发现候选，不作为设计证据 |

## 外部结构逐项核验

### 1. MambaTS：VAST、TMB、variable permutation

- **论文：** [MambaTS, arXiv:2405.16440](https://arxiv.org/abs/2405.16440)
- **官方仓库：** [XiudingCai/MambaTS-pytorch](https://github.com/XiudingCai/MambaTS-pytorch)
- **冻结源码：** commit [`3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d`](https://github.com/XiudingCai/MambaTS-pytorch/tree/3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d)
- **入口：** [`models/MambaTS.py`](https://github.com/XiudingCai/MambaTS-pytorch/blob/3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d/models/MambaTS.py)、[`layers/mamba_ssm/mixer2_seq_simple.py`](https://github.com/XiudingCai/MambaTS-pytorch/blob/3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d/layers/mamba_ssm/mixer2_seq_simple.py)
- **证据：** B（论文 + 官方代码）。

源码核对到的执行路径：

1. `MambaTS.py` 将每个通道切成 patch，嵌入后按 `B (C*L) D` 做 variable-scan。
2. `MixerTSModel` 的 `VPT_mode=1` 在训练时随机打乱变量 token，并记录相邻变量转移。
3. `batch_update_state` 用训练 batch 代价累积 adjacency/count；评估时用 GD 或 `python-tsp` 的 SA/LS/LK 解变量访问顺序。
4. 推理后对 hidden states 做 `unshuffle`，恢复输出的原始通道顺序。
5. 官方改造版 Mamba 支持关闭 causal convolution，并在 selective projection 后加入 dropout。

**适配建议：**

- 只引入“训练期变量置乱 + 训练后冻结变量顺序”这一机制，先做独立 ablation。
- 顺序应写入与 `acf_prior.npz` 同级的 train-only sidecar，包含 `order`、`inverse_order`、训练数据哈希、算法和 solver 版本；评估禁止在线 ATSP。
- 当前 `MambaACF` 的 variable branch 已有 `mamba_acf_channel_order` 和 inverse reorder 语义（见 [`models/MambaACF.py:2394`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:2394) 和 [`models/MambaACF.py:3321`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:3321)），可把 VPT order 接在这里，而不改变 ACF lag 的时间坐标。
- AutoCon 表示必须在恢复原始 channel order 后生成；否则同一 channel 的正样本关系会因变量排列而改变。
- Traffic 上不要默认使用 `python-tsp` 的全量 CPU 求解；先比较固定输入顺序、ACF strength/similarity 顺序和离线 VPT 顺序的成本与收益。`similarity` 只在先验挂载时做 `O(C^2K)` 点积，不应被误报为每步 forward 成本。

**风险：** VAST 的排序目标来自训练损失，不是因果图；它可能把数据集/seed 特性编码进顺序。若不冻结 sidecar，评估会发生模型身份漂移。

### 2. TimeMachine：四路 Mamba、两阶段压缩和残差

- **论文：** [TimeMachine, arXiv:2403.09898](https://arxiv.org/abs/2403.09898)（ECAI）
- **官方仓库：** [Atik-Ahamed/TimeMachine](https://github.com/Atik-Ahamed/TimeMachine)
- **冻结源码：** commit [`5bf17a728349c674eceafc28a90f22b904e38c2d`](https://github.com/Atik-Ahamed/TimeMachine/tree/5bf17a728349c674eceafc28a90f22b904e38c2d)
- **入口：** [`TimeMachine_supervised/models/TimeMachine.py`](https://github.com/Atik-Ahamed/TimeMachine/blob/5bf17a728349c674eceafc28a90f22b904e38c2d/TimeMachine_supervised/models/TimeMachine.py)
- **证据：** B。

源码/论文共同确认的结构：

- `Linear(seq_len -> n1)` 和 `Linear(n1 -> n2)` 形成两级时间压缩；中间有 dropout 和 residual。
- 两个 Mamba 在高/低分辨率路径上处理时间上下文；另外两个路径通过转置承担 channel-mixing/global context 或 channel-independent/local context。
- `Linear(n2 -> n1)` 后与外层路径拼接，再由 `Linear(2*n1 -> pred_len)` 输出；预测端使用 RevIN/实例归一化。
- 论文报告 `d_conv=2`、`expand=1`，并对 dropout、残差、state expansion 做了消融；这些是该论文的实验合同，不是当前仓库的默认值。

**适配建议：**

- 借鉴“两级时间尺度 + residual”，但保留当前 `MambaACF` 的单一 shared temporal state 和 direct multi-horizon head。
- 可把各尺度 hidden 做 masked mean/last-token pooling，拼成一个表示后再进入现有 `_contrastive_repr`；预测头和对比头继续解耦。
- 不建议原样复制四路 Mamba：它会把扫描和参数开销近似成多份，并且让 AutoCon 的表示来源难以归因。

**风险：** 论文明确把 channel-mixing 与 channel-independent 策略按数据集切换；不能把同一套四路结构宣称对所有高维数据都有效。作者在论文结尾将 self-supervised setting 列为 future work，因而没有直接证据支持与 AutoCon 联合。

### 3. CMamba：M-Mamba + GDD-MLP + Channel Mixup

- **论文：** [CMamba, arXiv:2406.05316](https://arxiv.org/abs/2406.05316)
- **官方仓库：** [zclzcl0223/CMamba](https://github.com/zclzcl0223/CMamba)
- **冻结源码：** commit [`78166596a235fa1565c5b84d034216dc8242c84b`](https://github.com/zclzcl0223/CMamba/tree/78166596a235fa1565c5b84d034216dc8242c84b)
- **入口：** [`models/CMamba.py`](https://github.com/zclzcl0223/CMamba/blob/78166596a235fa1565c5b84d034216dc8242c84b/models/CMamba.py)、[`layers/CMambaEncoder.py`](https://github.com/zclzcl0223/CMamba/blob/78166596a235fa1565c5b84d034216dc8242c84b/layers/CMambaEncoder.py)、[`layers/GDDMLP.py`](https://github.com/zclzcl0223/CMamba/blob/78166596a235fa1565c5b84d034216dc8242c84b/layers/GDDMLP.py)
- **证据：** B。

源码核对：每层先对 `[B*C, patch, D]` 做 M-Mamba，再 reshape 为 `[B,C,patch,D]`；可选 GDD-MLP 对 channel 轴做 pooled average/max 的 data-dependent scale/shift；最后 residual 加回。Channel Mixup 在训练侧用于抑制过拟合。

**适配建议：**

- 这是当前 Traffic 失败最直接的结构候选：在 shared temporal state 后增加一次**低秩、共享权重、数据依赖的 channel mixer**，避免 dense `C x C` 参数。
- ACF 只提供 train-only gate/conditioning（例如每通道 signed peak summary），不能把 ACF 值直接当成 channel correlation；实际 mixer 权重仍由窗口 hidden 产生。
- AutoCon 表示取 mixer 后的 hidden，并维持现有 projection head；预测和表示两条路径仍须单独记录。
- Channel Mixup 可以作为正则化 ablation，但不能与 variable permutation 同时改变，避免无法归因。

**风险：** 官方实现的 GDD-MLP 仍有 `n_vars // reduction` 的容量/整除假设；对 `C=7` 与 `C=862` 需要单独定义低秩宽度。论文结果不自动转移到当前数据预处理和损失合同。

### 4. ms-Mamba：并行内部时间尺度

- **论文/期刊页面：** [ms-Mamba: Multi-scale Mamba for Time-Series Forecasting](https://www.sciencedirect.com/science/article/pii/S0925231226006235)
- **官方仓库：** [meric-karadag/ms-mamba](https://github.com/meric-karadag/ms-mamba)
- **冻结源码：** commit [`24c5e3eb5eff2bc968f37814a0e44e837d8302c1`](https://github.com/meric-karadag/ms-mamba/tree/24c5e3eb5eff2bc968f37814a0e44e837d8302c1)
- **入口：** [`layers/AdaptiveHierarchicalMamba.py`](https://github.com/meric-karadag/ms-mamba/blob/24c5e3eb5eff2bc968f37814a0e44e837d8302c1/layers/AdaptiveHierarchicalMamba.py)、[`layers/hierarchicalMamba.py`](https://github.com/meric-karadag/ms-mamba/blob/24c5e3eb5eff2bc968f37814a0e44e837d8302c1/layers/hierarchicalMamba.py)
- **证据：** B/C。仓库很新，需锁定依赖和脚本后才能独立复现论文数字。

该方向不是输入下采样，而是多个 Mamba 分支使用不同内部 `Delta`/sampling-rate；官方 README 区分 fixed、learnable、dynamic 三种 scale，并说明并行分支输出长度相同后融合。

**适配建议：**

- 当前已有 `mamba_scales` 和多尺度融合，优先只试固定的 2 个或 3 个内部 scale，分支 hidden 做加权平均。
- AutoCon 表示使用 scale-aware pooling，并把 scale 配置写入 runtime contract。
- dynamic scale 暂不作为首轮方案：它与当前 ACF scale conditioning 同时存在时会把“周期先验”和“输入依赖步长”混在一起，解释和归因都较弱。

### 5. S-Mamba：双向/通道 token baseline

- **论文：** [Is Mamba Effective for Time Series Forecasting?, arXiv:2403.11144](https://arxiv.org/abs/2403.11144)；README 标注 Neurocomputing 已接收。
- **官方仓库：** [wzhwzhwzh0921/S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba)
- **冻结源码：** commit [`e7e8bf04066135afa43d85b0a87afa97cda16e3f`](https://github.com/wzhwzhwzh0921/S-D-Mamba/tree/e7e8bf04066135afa43d85b0a87afa97cda16e3f)
- **入口：** [`model/S_Mamba.py`](https://github.com/wzhwzhwzh0921/S-D-Mamba/blob/e7e8bf04066135afa43d85b0a87afa97cda16e3f/model/S_Mamba.py)
- **证据：** A/B（正式发表信息 + 官方代码；具体依赖仍需锁定）。

S-Mamba 使用 inverted/channel-token embedding，并在每层将 forward 与 reverse Mamba 输出相加，再接 FFN。它适合作为强 baseline 或非严格因果 encoder 对照。

**适配建议：** 不把它作为当前主架构改造；可用相同 AutoCon projection 接在 encoder token 上做 baseline。若论文宣称严格 causal forecasting，必须明确双向扫描只读取历史窗口且不读取 decoder/future，不能笼统写成 causal Mamba。

### 6. MambaMixer：dual token/channel selection

- **论文：** [MambaMixer, arXiv:2403.19888](https://arxiv.org/abs/2403.19888)
- **项目页：** [mambamixer.github.io](https://mambamixer.github.io/)
- **官方仓库：** [MambaMixer/M2](https://github.com/MambaMixer/M2)
- **冻结源码：** commit [`250a86b9ff8bbb986b132c4e07d7aa8fd8b83f67`](https://github.com/MambaMixer/M2/tree/250a86b9ff8bbb986b132c4e07d7aa8fd8b83f67)
- **证据：** C。该 commit 的 README 明确写着 “Codes and models will be available soon”。

论文的 TSM2 设计包含 token mixer、channel mixer、weighted averaging 和跨层直连；时间维使用单向 S6，另一路可混合 auxiliary information。设计上与当前“shared temporal + optional variable branch”有启发关系，但没有可冻结官方 TS 实现，不能直接声称复现。

**适配建议：** 只把 weighted skip 作为后续小型 ablation；不要复制论文图中的未公开模块，也不要把项目页的“outstanding performance”当成可引用的本仓库证据。

### 7. Mamba4Cast：zero-shot synthetic pretraining

- **论文：** [Mamba4Cast, arXiv:2410.09385](https://arxiv.org/abs/2410.09385)
- **官方仓库：** [automl/Mamba4Cast](https://github.com/automl/Mamba4Cast)
- **冻结源码：** commit [`f11d0df579fdcecdc69f463fdf7a8091eb0cf64d`](https://github.com/automl/Mamba4Cast/tree/f11d0df579fdcecdc69f463fdf7a8091eb0cf64d)
- **证据：** B。

Mamba4Cast 以 synthetic data 预训练 PFN 风格模型，使用 Mamba-2 单次输出完整 horizon，目标是 zero-shot 泛化和推理效率，不是当前仓库的 dataset-specific supervised forecasting。它没有直接的 ACF sidecar 或 AutoCon relation objective。

**适配建议：** 当前论文主线不纳入；未来若做 foundation-model 分支，应另立数据和预训练协议，不能与 supervised AutoCon screen 混成一个实验矩阵。

## 与当前 AutoCon/ACF 的接口映射

| 外部思想 | 当前可挂接位置 | AutoCon 表示策略 | 必须冻结/记录的内容 |
| --- | --- | --- | --- |
| CMamba channel mixer | `MambaACF._encode_channels` 后、forecast head 前 | mixer 后 hidden -> 独立 projection | mixer rank、pooling、gate、channel width |
| MambaTS VPT/VAST | variable branch token order | restore 原始 channel order 后再 `_contrastive_repr` | order/inverse、训练数据哈希、solver、随机种子 |
| ms-Mamba fixed scales | temporal encoder block 内并行分支 | scale-aware pooled hidden 拼接/平均 | scale 数量、初始化、融合权重 |
| TimeMachine 两级压缩 | temporal hidden 的轻量旁路 | 每级 pooled summary 拼接 | `n1/n2`、residual、dropout |
| S-Mamba bidirectional | baseline encoder | encoder output projection | forward/reverse 组合、是否允许严格 causal claim |
| MambaMixer weighted skip | 多层 block 输出融合 | 最终融合 hidden | skip 权重参数化和额外 FLOPs |

当前代码已有的关键边界：

- 只要 live forecast/representation 分支消费 train-only ACF prior，sidecar 就在实验入口先加载，缺失时不能静默从 val/test 拟合；所有 ACF 分支关闭的 `is_training=0` plain replay 则明确跳过 sidecar（[`exp/exp_long_term_forecasting_with_mamba.py:35`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/exp/exp_long_term_forecasting_with_mamba.py:35)）。
- `MambaACF` 将预测 state 与 AutoCon projection 分离，AutoCon 模型返回 `(prediction, representation)`（[`models/MambaACF.py:4180`](/home/zhong/Documents/Self-Supervised-Contrastive-Forecsating/models/MambaACF.py:4180)）。
- ACFRelationLoss v3 的候选/正样本、符号项、local/global scope 是现有训练合同；外部架构 screen 不得同时改变 relation target。
- 当前仓库已有 channel lag-bank；外部结构只能在相同 lag-bank 与 sidecar 合同下比较，不能把 ACF 分支和 backbone 变更捆绑成一个“总提升”。

## 推荐的最小增量研究路线

### Phase A：跨通道 mixer（最高优先级）

- backbone：当前 `MambaACFVariableAutoCon` 或 shared MambaACF，固定已有 ACF target、sidecar 和 AutoCon lambda。
- 唯一差异：`channel_mixer=off` vs. 一次低秩 data-dependent mixer。
- 先做 ETTh1、Electricity、Traffic，H96/H336，至少 3 个 seed；Traffic 单独报告每通道/整体指标。
- 同时记录参数量、峰值显存、forward latency、训练吞吐；不能只报 MSE。

### Phase B：冻结变量顺序

- 只在 Phase A 的最佳或预注册基础配置上加 VPT order；不同时改 mixer、lag-bank、relation target。
- order 必须在 train split 生成并持久化；standalone evaluation 只读取 sidecar。
- 做 input order / ACF strength / ACF profile similarity / VPT order 四臂，并报告 permutation/reverse robustness；若顺序敏感性很大，优先处理 invariant training，再解释任何排序收益。

### Phase C：固定多尺度 Delta

- 仅 2 个固定 scale，保持总参数和 Mamba layer 数可比；dynamic scale 暂缓。
- AutoCon 表示用所有 scale 的同一 pooling 规则，避免某一分支获得额外监督。

## 不能直接采用的做法

- 把外部论文表格中的 SOTA 数字填入本仓库 baseline 表。
- 在测试窗口计算 VAST/ACF 顺序或重新选择 lag。
- 用四路 Mamba、channel mixer、ACF memory、relation loss 同时开启后只与 plain 模型比较。
- 用 Mamba4Cast synthetic pretraining 结果证明 supervised AutoCon 的有效性。
- 把 MambaMixer 项目页的性能描述当成已公开、可复现的官方代码证据。

## 外部来源快照

本报告中的仓库 commit 是检索日固定的源码锚点：

```text
MambaTS       3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d
TimeMachine   5bf17a728349c674eceafc28a90f22b904e38c2d
S-Mamba       e7e8bf04066135afa43d85b0a87afa97cda16e3f
CMamba        78166596a235fa1565c5b84d034216dc8242c84b
Mamba4Cast    f11d0df579fdcecdc69f463fdf7a8091eb0cf64d
ms-Mamba      24c5e3eb5eff2bc968f37814a0e44e837d8302c1
MambaMixer/M2 250a86b9ff8bbb986b132c4e07d7aa8fd8b83f67
```

这些 commit 只用于架构和源码核验，不意味着依赖、数据、GPU kernel 或外部结果已经在本仓库复现。
