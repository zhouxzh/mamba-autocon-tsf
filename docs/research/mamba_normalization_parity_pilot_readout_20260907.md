# Mamba / AutoCon normalization parity pilot 读数

日期：2026-09-07  
状态：20 个作业全部完成；两数据集、两 seed、H96、4 epoch、validation-only；不是确认性论文结果。

## 合同核对

所有作业均显式使用 `AutoCon_wnorm=MeanStd`、`skip_test=true`、seed `2021/2022`。
PatchTST、Mamba-1、Mamba-2、plain MambaACF 和 MambaACFAutoCon 的输入切分与训练预算
一致。Mamba-1/2 采用 `d_state=16`、`d_conv=4`、`expand=2`、`headdim=16`、单尺度、
单位 stride、causal embedding。MambaACF 使用 train-only `train_peaks_available` ACF；
AutoCon arm 使用 `raw + hard_selected + lag_mask=true + global + encoder` relation。

所有作业返回码为 0；每个 arm 都有两个 `config.json`、两个 `loss_logs.json` 和一份
独立 resource JSON/CSV。资源测量显式为 prediction-only，`relation_loss_included=false`
且没有请求 auxiliary representation，因此不能当作 AutoCon 训练吞吐。

## 最佳 validation MSE

`best` 是每个 seed 的最低 validation loss；括号内为两个 seed 的相对变化。

| 数据集 | PatchTST | Mamba-1 | Mamba-2 | MambaACF | MambaACFAutoCon |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.690674 / 0.688481 | 0.719921 / 0.719351 | 0.733493 / 0.723162 | 0.755659 / 0.723029 | 0.744944 / 0.716314 |
| Traffic | 0.438770 / 0.447470 | 0.494447 / 0.491928 | 0.496279 / 0.484519 | 0.408935 / 0.401918 | 0.406096 / 0.404602 |

均值及相对 PatchTST 的差异：

| 数据集 | PatchTST mean | Mamba-1 | Mamba-2 | MambaACF | MambaACFAutoCon |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.689578 | 0.719636 (+4.36%) | 0.728327 (+5.62%) | 0.739344 (+7.22%) | 0.730629 (+5.95%) |
| Traffic | 0.443120 | 0.493187 (+11.30%) | 0.490399 (+10.67%) | 0.405427 (-8.51%) | 0.405349 (-8.52%) |

在同一 MambaACF 合同内，AutoCon relation 相对 plain MambaACF 的均值变化为
ETTh1 `-1.18%`（两个 seed 同方向），Traffic `-0.019%`（一个 seed 改善、一个 seed
退化）。因此 relation path 在 Traffic 的收益主要来自 MambaACF/ACF 结构，当前短预算
没有证明 AutoCon relation 本身带来额外平均增益；但 telemetry 证明关系目标确实被执行，
而不是退化为空目标（contrastive-active 约 ETTh1 `84--85%`、Traffic `81%`）。

## 训练与 GPU 资源

4 epoch 训练/验证 wall time 总和（秒）及统一 prediction-only GPU median latency（毫秒）：

| 数据集 | 模型 | wall time | median / p95 | trainable params | peak increment MB |
| --- | --- | ---: | ---: | ---: | ---: |
| ETTh1 | PatchTST | 3.815 | 0.278 / 0.298 | 46,080 | 1.514 |
| ETTh1 | Mamba-1 | 3.846 | 0.418 / 0.433 | 31,975 | 1.812 |
| ETTh1 | Mamba-2 | 27.230 | 0.708 / 0.730 | 29,907 | 3.304 |
| ETTh1 | MambaACF | 27.221 | 0.719 / 0.739 | 38,644 | 4.034 |
| ETTh1 | MambaACFAutoCon | 31.275 | 0.715 / 0.734 | 40,756 | 4.034 |
| Traffic | PatchTST | 44.313 | 0.890 / 0.901 | 46,080 | 95.188 |
| Traffic | Mamba-1 | 24.371 | 0.461 / 0.479 | 169,630 | 23.156 |
| Traffic | Mamba-2 | 50.729 | 0.753 / 0.775 | 167,562 | 23.156 |
| Traffic | MambaACF | 51.035 | 0.733 / 0.752 | 148,939 | 15.302 |
| Traffic | MambaACFAutoCon | 56.245 | 0.735 / 0.762 | 151,051 | 15.302 |

资源相对 PatchTST：ETTh1 上 MambaACF/AutoCon median latency 约 `+159%/+157%`；Traffic
上约 `-17.7%/-17.4%`。这表明 MambaACF 的速度优势具有数据维度依赖性，不能写成普遍
加速。AutoCon 预测-only 路径不执行 relation loss，训练 wall time 仍比 plain MambaACF
高约 ETTh1 `14.9%`、Traffic `10.2%`。

## 结论与下一步

1. `MeanStd` 消除了 LastVal 与 PatchTST 之间的主要归一化混杂；在 Traffic，plain
   MambaACF 和 AutoCon 都比 PatchTST 的短预算 validation loss 低约 `8.5%`，并且两 seed
   方向一致。
2. ETTh1 仍是负结果：MambaACF/AutoCon 比 PatchTST 高约 `7.2%/6.0%`。这否定了“跨数据集
   普遍提升”的表述，也说明 ACF memory 与 AutoCon 关系目标存在数据集依赖。
3. AutoCon 关系目标已实际运行且 telemetry 非退化，但两 seed 的平均增益接近零；因此
   当前不能把 AutoCon relation 单独写成主要贡献。后续必须把它与 plain MambaACF 做独立
   null、补足 3--5 seed，并报告 paired effect。
4. 下一优先候选是 `variable-token + AutoCon`：它在另一个 paired pilot 中 Traffic 两 seed
   都改善，但 ETTh1 方向不一致，且 median latency 增加约 `71--72%`。它只能作为精度候选，
   不能作为当前效率主线；低秩/共享输出投影应单独注册后再优化。

原始训练与资源工件：
[`mamba_normalization_parity_pilot_20260907`](../../artifacts/exploratory/mamba_normalization_parity_pilot_20260907)。
