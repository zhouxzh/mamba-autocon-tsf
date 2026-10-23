# MambaACF Memory-Gate Mode Screen Review

**状态：** `exploratory_mamba_acf_memory_gate_mode_screen_draft`，待人工复核，未授权训练。

**计划：** [`mamba_acf_memory_gate_mode_screen_plan_20260902.json`](./mamba_acf_memory_gate_mode_screen_plan_20260902.json)

## 审计对象

这是一项严格的 two-arm mechanism screen，比较同一个
`MambaACFVariableAutoCon`（runner alias）/`MambaACFAutoCon`（executable）模型中
ACF memory gate 的输入来源：

| Arm | Gate source | Runtime shape | What changes |
|---|---|---|---|
| `memory_gate_hidden_h96` | final shared temporal hidden state | `[B,H,1]`, broadcast over channels | Reference |
| `memory_gate_variable_token_h96` | restored variable-token states | `[B,H,C]`, channel-specific | Only `mamba_acf_memory_gate_mode` |

两臂都固定 variable-token mixer、输入 channel order、raw train-only ACF
sidecar、`train_peaks_available` lag selector、H96、global hard-selected signed
relation objective (`lambda=0.03`)、ETTh1/Traffic、seeds 2021--2023、batch 32、
最多 8 epochs 和相同优化器/调度器。两臂不是 variable-mixer on/off 比较；
它们的 estimand 是 gate source 的单因素差异。

## 关键实现边界

hidden 模式由共享 temporal hidden 的最后一个时间位置产生一个 horizon gate，
随后在所有 channel 上广播。variable-token 模式在 variable branch 完成
permutation/channel-order restore 后，从每个 channel token 产生独立 horizon gate。
因此 variable-token 模式增加的是 channel-specific modulation，不增加 gate
层参数；它改变的是输出张量规模和计算/激活成本。
由于 gate source 位于 variable-token forecast branch 内，梯度还会通过 gate
回传到该 branch 的 token 表示；这不是把同一 gate 简单换一个广播轴，而是
functional-capacity/optimization-path ablation，结果解释必须保留这一点。

两臂都要求静态 train-only ACF memory（`mamba_use_acf_memory=true`、
`mamba_dynamic_acf=false`）。动态窗口 ACF、strength order、channel conditioning、
representation fusion 和 horizon calibration 均关闭，避免把 gate-source 效应
与其它机制混淆。

## 作业矩阵与证据边界

- ETTh1（7 channels）和 Traffic（862 channels），仅 H96。
- 每个 dataset 使用 seeds 2021、2022、2023；推断/配对单位是
  `dataset × horizon × seed`，不是窗口或 channel 数。
- 两臂共 12 个训练作业，输出根目录为
  `artifacts/paper_final/mamba_acf_memory_gate_mode_screen_20260902/`，当前不应
  创建或复用训练 artifacts。
- 每个 paired coordinate 必须共享完全相同的 `acf_prior.npz` hash、effective
  lags、relation candidate lags、config invariants 和 immutable provenance。
- 只做 descriptive paired deltas；禁止 test-based arm selection、跨 plan 池化、
  事后补 seed/horizon 或把结果写成 publication claim。

## 资源成本合同

`acf_gate` 是 `Linear(d_model, pred_len)`，在本计划宽度下两臂都为
`32*96+96 = 3,168` 个参数。差异在输出规模：

| Dataset | Hidden output / item | Variable-token output / item | Variable MACs / item | Variable FP32 activation |
|---|---:|---:|---:|---:|
| ETTh1 (`C=7`) | 96 | 672 | 21,504 | 2,688 bytes |
| Traffic (`C=862`) | 96 | 82,752 | 2,648,064 | 331,008 bytes |

Hidden gate 为 3,072 MAC、384 FP32 bytes；variable-token gate 的 gate-only
MAC/activation 分别放大 `C` 倍（ETTh1 7 倍、Traffic 862 倍）。这只是层级
估算，不等价于端到端延迟。授权训练前必须用相同 GPU、batch、warmup/timed
forward 协议做独立 resource smoke；资源作业不计入 12 个训练作业，也不能
用 MAC 比例直接声称训练加速/减速。

预设工程预算为 variable-token 相对 hidden 的 median latency 和 peak allocated
memory 均不超过 1.25 倍；超出即停止该探索，不在看过结果后放宽预算。

## 停止与升级规则

1. 任意 manifest、artifact、source 或 sidecar invariant 失败，立即使对应
   phase 无效，不允许用补跑或替换文件掩盖问题。
2. variable-token gate 只有在六个 paired MSE 坐标至少赢 4 个、且每个 dataset
   至少赢 2/3 个 seed，并且两个 dataset 的平均相对 MSE 均不劣于 1% 时，才可
   进入另一个独立冻结的 replication 设计。
3. 任一 dataset 平均 MSE 恶化超过 1%，或 resource smoke 超过 1.25 倍，均停止
   promotion；不增加交互臂、长 horizon 或额外 seeds 来追逐结果。
4. 即使达到门槛，本 screen 仍只能支持“在该实现、数据、H96 和预算下的 gate
   source 探索性信号”，不能支持普适准确率/效率或论文发表结论。

## 人工复核清单

- 确认 `[B,H,1]` 与 `[B,H,C]` 的语义确实对应当前 `MambaACF.forward` 路径，
  且 variable-token gate 在 channel restore 后计算。
- 核对两臂 alias、executable、所有 Mamba/AutoCon/ACF 配置完全一致，只有
  `mamba_acf_memory_gate_mode` 不同。
- 核对数据指纹、train-only sidecar 规则、fresh output root 和 source snapshot
  记录；不要把已有 gate resource benchmark 或其它 screen 的 artifacts 直接
  合并进本矩阵。
- 先审阅资源预算和停止规则，再决定是否创建 launch sidecar；本文件本身不
  授权训练。

## 结论边界

这是一个可审计的未来 screen 草案，不是结果报告。即使 12 个作业全部完成，
也只能回答 gate source 在两个 hourly 数据集、H96、三 seed 和当前实现中的
描述性差异；任何更广泛的 Mamba/ACF 优势、速度结论或论文主张都需要独立的
确认协议。
