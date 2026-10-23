# MambaTS v4 AutoCon PEMS Screen

**日期：** 2026-09-10  
**状态：** validation-only exploratory；测试集保持关闭。  
**用途：** 人工复核 v4 的 train-only ACF memory 是否在不破坏 v3 短期优势的情况下改善长期预测。

## 研究问题

v4 在 MambaTS v3 的 shared patch-token scan、低秩跨变量 context 和 per-channel persistence gate 上，增加一个由训练集 ACF 初始化的静态 residual memory。ACF 分支使用 ReZero gate，从零开始；预测时只索引已经观测到的历史位置，不能读取验证或测试窗口的统计量。AutoCon 仍以 `lambda=0.001` 运行。

这项 screen 只回答“结构是否可运行、ACF 分支是否被优化、短/长期 validation 方向如何”，不回答确认性精度、测试集泛化或论文主表结论。

## 冻结合同与工件

- 计划：[`mamba_pems_ts_v4_autocon_screen_20260910.json`](mamba_pems_ts_v4_autocon_screen_20260910.json)
- 工件根目录：[`mamba_pems_ts_v4_autocon_screen_20260910`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_screen_20260910/)
- 读出：[`readout.md`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_screen_20260910/readout/readout.md)
- 精确配对 CSV：[`mambatsv3autocon_deltas.csv`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_screen_20260910/readout/mambatsv3autocon_deltas.csv)、[`timesnet_deltas.csv`](../../artifacts/exploratory/mamba_pems_ts_v4_autocon_screen_20260910/readout/timesnet_deltas.csv)

矩阵包含 PEMS03/04/07/08、H=12/96、seed=2022/2023，共 16 个坐标、16 个作业。16/16 作业成功，phase validator 通过，测试指标没有写入读出。ACF prior 按训练 split 拟合，不能与验证或测试窗口混合。

## 结果

| 比较 | H=12 | H=96 |
|---|---:|---:|
| v4 vs MambaTSv3AutoCon | 3/8 胜，平均 `+0.074%` | 5/8 胜，平均 `-0.057%` |
| v4 vs TimesNet | 8/8 胜，平均 `-19.449%` | 2/8 胜，平均 `+6.678%` |

ACF gate 在 16/16 个 checkpoint 中为非零。平均绝对 gate 为 H=12 的 `0.029758`、H=96 的 `0.032578`，说明 ACF residual 在训练中没有被完全关闭；但相对 v3 的变化接近零，不能说 ACF memory 已经带来独立精度增益。H=96 相对 TimesNet 的差距仍然存在，主要失败坐标是 PEMS03/04。

## 人工复核结论

1. **实现层面通过。** v4 的模型注册、AutoCon 返回表示、ACF prior 绑定、因果历史索引和资源配置合同均有覆盖；新回归测试验证了 shape、variant、lag 压缩和初始 zero-gate。
2. **机制层面部分通过。** gate 在所有坐标中被训练到非零，说明 ACF 路径确实参与优化；但当前 screen 没有显示相对 v3 的稳定改进。
3. **预测层面不通过晋级门槛。** H=12 的 TimesNet 优势保持，H=96 仍有 `+6.678%` 的平均相对误差；v4 不能作为“跨 horizon 不低于强基线”的论文主模型。
4. **证据等级仍为 exploratory。** 训练预算为 4 epochs，指标为 validation-only，尚未完成正式预算、更多 seed、测试集锁定后的单次评估和完整效率统计。

## 可复现性限制

当前 v4 在评估时会从 train split 重新构造 ACF prior，而不是读取独立的 `acf_prior.npz` sidecar。这样可以避免验证/测试泄漏，但正式论文实验前仍应补齐 prior sidecar 的保存、hash 和 replay contract；否则“同一 checkpoint + 同一 prior”的独立回放仍不完整。该限制不改变本 screen 的 train-only 数据边界。

## 下一步门槛

- 在不重新选择测试集的前提下，固定 v4 配置，完成正式训练预算和至少 3 个 seed 的 H=12/H=96 配对复验。
- 单独报告 PEMS03/04 H=96 的失败分析，不把 H=12 的胜出外推到长期预测。
- 完成 ACF sidecar/replay contract、训练吞吐和峰值显存的独立测量。
- 只有当 v4 在预注册坐标上相对 v3 稳定改善、且不牺牲 H=12 或效率时，才考虑进入确认性矩阵；本 screen 本身不能支持投稿主张。

