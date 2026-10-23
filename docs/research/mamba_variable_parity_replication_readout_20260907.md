# Variable-token parity 三 seed replication 读数

日期：2026-09-07  
性质：探索性 paired replication；不是预注册确认性结果，也没有读取 test split。

## 目的与固定合同

本轮只补充 `seed=2023`，把既有 ETTh1/Traffic、H96、4 epoch 的
`MambaACFAutoCon` variable-token paired pilot 从 2 seed 扩展到 3 seed。两臂只在
`mamba_variable_mixer` 上不同：shared arm 为 `false`，variable arm 为 `true`。
两臂均固定 `MeanStd`、causal embedding、train-only `train_peaks_available` ACF、
`raw + hard_selected + lag_mask=true`、global encoder relation、balanced lag pairs、
`min_lag=96`、`AutoCon_lambda=0.03`、batch 32、`d_model=32`、单尺度和相同早停/学习率合同。

四个第三 seed 作业均返回码 0，均为 `skip_test=true`。原始日志、配置、sidecar、
checkpoint 和资源记录位于
`artifacts/exploratory/mamba_variable_parity_replication_20260907/`。

## 三 seed paired validation

表中 delta 是 `(variable/shared - 1) * 100%`，负值表示 variable-token 验证损失更低。

| 数据集 | seed | shared MSE | variable MSE | delta | shared wall(s) | variable wall(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 2021 | 0.744944014 | 0.703650127 | -5.543% | 14.907 | 16.711 |
| ETTh1 | 2022 | 0.716314397 | 0.719992324 | +0.514% | 16.368 | 16.749 |
| ETTh1 | 2023 | 0.717758756 | 0.704365778 | -1.866% | 14.873 | 16.765 |
| Traffic | 2021 | 0.406096252 | 0.379617060 | -6.520% | 27.818 | 31.868 |
| Traffic | 2022 | 0.404602280 | 0.383597873 | -5.191% | 28.426 | 31.645 |
| Traffic | 2023 | 0.407460041 | 0.386934173 | -5.038% | 27.468 | 31.896 |

三 seed 汇总：

| 数据集 | shared mean +/- sd | variable mean +/- sd | 平均 delta | 改善 seed 数 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.726339 +/- 0.016129 | 0.709336 +/- 0.009236 | -2.299% | 2/3 |
| Traffic | 0.406053 +/- 0.001429 | 0.383383 +/- 0.003663 | -5.583% | 3/3 |

第三 seed 的 relation telemetry 没有出现退化：ETTh1 shared/variable 的
contrastive-active anchor 为 `84.28%/84.28%`，degenerate-positive 为
`6.89%/6.89%`；Traffic 为 `80.88%/80.88%` 和 `10.45%/10.45%`。同一输入/lag 合同
下两臂 telemetry 相同，说明本轮差异来自 variable branch，而不是 ACF sidecar 变化。

## 计算代价

训练 wall time 是每个作业日志中 epoch cost 的总和，不是吞吐基准。三 seed 平均为：

| 数据集 | variable 相对 shared 的平均训练 wall | prediction-only median latency | synthetic sequences/s | trainable 参数 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | +8.84% | 0.718 -> 1.217 ms（+69.59%） | 11112 -> 6560（-40.97%） | 40756 -> 62850（+54.21%） |
| Traffic | +13.97% | 0.743 -> 1.252 ms（+68.59%） | 10751 -> 6378（-40.67%） | 151051 -> 173145（+14.63%） |

Traffic 的 steady-state peak increment 为 `15.302 -> 37.658 MB`（+146.10%）；ETTh1
两臂均为 `4.034 MB`。资源协议为 RTX 5090 D、device-resident synthetic input、batch 8、
10 warm-up/50 timed forwards；`relation_loss_included=false`，因此不能把该表解释为
训练端到端速度或包含反向传播的效率结果。

## 解释与停止判断

Traffic 的三 seed 方向一致，说明 variable-token branch 是一个值得进入更大预算复验的
精度候选；ETTh1 只有 2/3 改善，仍属数据集依赖。与此同时，两数据集的预测 latency
都增加约 69%，吞吐下降约 41%，所以该分支不能支持“更快”或“效率-精度同时提升”的主张。

这组结果仍不足以写成确认性论文主表：没有 test 指标，只有两个数据集、一个 horizon、
短训练预算和三 seed；shared AutoCon 本身也没有证明 relation loss 的独立平均收益。
下一步若继续，只应在预先冻结的 3--5 seed、完整 epoch、同一强基线和独立 test gate 下复验，
并把 variable branch 的容量/延迟代价作为主要审计项。

