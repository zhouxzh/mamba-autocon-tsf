# Mamba patch-token pilot 读数

日期：2026-09-07  
状态：已完成工程筛选；未注册、未授权，不是论文结果。

## 结论先行

仓库中已有的 `MambaACFAutoCon` patch-token 分支值得进入下一轮正式筛选：在同一模型、同一 AutoCon/ACF 配置、同一 seed 和短训练预算下，ETTh1 与 Traffic 的最佳验证损失均低于关闭 patch-token 的配对臂。

但当前结果不能支持“精度和速度同时优于基线”的结论：

- 这只是 1 个 seed、3--4 个 epoch、`skip_test=true` 的 validation-only pilot；
- patch-token 增加了参数和预测前向开销，Traffic 的显存峰值读数还有明显测量波动；
- `AutoCon` relation telemetry 显示 global contrastive active-anchor fraction 为 `0`，global degenerate-positive fraction 约为 `99.6%`（ETTh1）或 `100%`（Traffic），因此不能把这次收益归因于有效的跨样本对比学习。

## 配对训练协议

两臂均使用 `MambaACFAutoCon`，只改变 `mamba_patch_mixer`：

- 数据集：ETTh1、Traffic；多变量（`features=M`）；`seq_len=96`、`label_len=48`、`pred_len=96`；
- seed：`2021`；batch size：`32`；训练上限：4 epoch；early stopping patience：2；不运行 test；
- 主干：`d_model=32`、`d_ff=64`、1 个 Mamba block、Mamba-2 state width `16`；
- AutoCon：`--AutoCon --AutoCon_lambda 0.03 --AutoCon_wnorm MeanStd --AutoCon_acf_source residual`；
- ACF relation：`--mamba_acf_relation_loss`，其余 relation 参数保持默认；
- 变量 token、causal embedding 均开启；patch 端使用 `patch_len=16`、`patch_stride=8`、`time_major` 顺序；
- 输出根目录：[`artifacts/exploratory/mamba_patch_token_pilot_20260907`](../../artifacts/exploratory/mamba_patch_token_pilot_20260907)。

该分支采用 MambaTS 风格的时间 patch token：每个通道先形成短时间 patch，随后按显式的 time-major 或 channel-major 顺序进行共享 Mamba 扫描，再通过保留通道身份的 horizon head 产生加性 forecast residual。实现位于 [`models/MambaACF.py`](../../models/MambaACF.py) 的 `_patch_forecast`；外部结构参考仅用于工程迁移，不能替代本仓库的训练证据。[MambaTS 官方仓库](https://github.com/XiudingCai/MambaTS-pytorch)

## 验证损失

表中 `best val` 是训练过程中的最低 validation loss，不是测试集 MSE；`delta` 定义为 `(on / off - 1) * 100%`，负值表示 patch-token 更低。

| 数据集 | patch-token | 最佳 epoch | best val | 相对 delta | 参数量（可训练） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | off | 1 | 0.741807 | -- | 55,746 |
| ETTh1 | on | 1 | 0.728295 | -1.82% | 102,448 |
| Traffic | off | 4 | 0.386513 | -- | 166,041 |
| Traffic | on | 2 | 0.373491 | -3.37% | 212,743 |

完整 validation trajectory 保留在四个 checkpoint 目录的 `loss_logs.json` 中。由于两侧只跑一个 seed，且 ETTh1 的最佳点在第一 epoch，不能进行显著性检验，也不能把这组差异视为稳定泛化提升。

## 资源读数

资源测试使用合成、常驻 GPU 的输入，batch size `8`，warmup `10` 次、计时 `30` 次；只测 prediction-only forward，不含数据加载、CPU-GPU 传输、反向传播或 checkpoint I/O。设备为 RTX 5090 D。原始 JSON/CSV 在各数据集的 `resources/{etth1,traffic}_{off,on}` 目录。

| 数据集 | patch-token | 总参数 | median forward | p95 forward | peak increment |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | off | 58,947 | 1.229 ms | 1.258 ms | 3.28 MB |
| ETTh1 | on | 105,649 | 1.726 ms | 1.785 ms | 3.28 MB |
| Traffic | off | 169,242 | 1.250 ms | 1.278 ms | 32.17 MB |
| Traffic | on | 215,944 | 2.073 ms | 2.094 ms | 208.63 MB |

相对变化为：可训练参数 ETTh1 `+83.78%`、Traffic `+28.13%`；median forward ETTh1 `+40.49%`、Traffic `+65.88%`。Traffic 的 patch-on 峰值增量显著高于 off，应在正式 screen 中重复测量并拆分 allocator warm-up 与稳定区间，不能直接把该单次值当成严格显存上界。

训练日志中的 epoch 总耗时也上升：ETTh1 约 `17.5 -> 19.6 s`（`+11.7%`），Traffic 约 `35.7 -> 51.1 s`（`+43.2%`）。这仍是短 pilot 的 wall-clock 读数，不是端到端吞吐结论。

## AutoCon/ACF 诊断

relation objective 确实进入训练日志，`effective_lambda=0.03` 且 global valid fraction 约为 ETTh1 `0.162`、Traffic `0.133--0.135`。但是：

- `global_contrastive_active_anchor_fraction=0.0`；
- `global_degenerate_positive_anchor_fraction` 为 ETTh1 `0.9962`、Traffic `1.0`；
- global sign accuracy 约 `0.49--0.50`，接近随机；
- 当前 sampler 实际记录为 `batching_effective=shuffle`、`reason=all_candidate_targets`。

因此这次 pilot 更像是“patch representation 分支 + 当前 relation objective 的联合训练”筛选，不是对比学习已经有效的证明。下一轮必须先修复或重新定义 positive/negative anchor 支持，至少让 active-anchor telemetry 非零，再解释 AutoCon 的增益。

## 是否进入正式实验

建议进入一个小规模、已注册的 follow-up screen，但不直接进入论文主表：

1. 固定同一 `MambaACFAutoCon` 和 AutoCon/ACF 合同，先做 patch off/on 的 3 seed 配对复验；
2. 数据集扩展到 ETTh1、ETTm1、Electricity、Traffic，至少覆盖 H96，若资源允许再加 H336；
3. 同时保留 DLinear、NLinear、PatchTST、TimesNet、Mamba1、Mamba2 六个 plain baseline，并报告参数、训练步耗时和 prediction-only latency；
4. 在启动前冻结 relation target、lag sampler 和 active-anchor 通过门槛；若 telemetry 仍退化，应把该机制作为负结果单独报告，而不是调参到看起来有效；
5. 只有在多 seed test 指标、配对统计、容量公平性和端到端资源协议全部通过后，才考虑进入论文主表。

当前最稳妥的定位是：patch-token 是一个有精度信号、但明显增加计算成本且尚未证明 AutoCon 对比学习有效的候选结构。它不能支持“更快”或“已达到论文发表要求”的表述。
