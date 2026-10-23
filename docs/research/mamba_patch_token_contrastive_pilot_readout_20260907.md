# Mamba patch-token contrastive pilot 读数

日期：2026-09-07  
状态：已完成工程筛选；未注册、未授权，不是论文结果。

## 这次修正了什么

上一份 [`mamba_patch_token_pilot_readout_20260907.md`](mamba_patch_token_pilot_readout_20260907.md) 记录了 `mamba_acf_relation_lag_mask=false` 的结构 pilot。该配置在 `target_mode=hard_selected` 下按当前代码语义会清空 selected-lag mask，relation objective 退化为 all-candidate，因此它不能证明有效的 selected-peak contrastive learning。

本次只把 `mamba_acf_relation_lag_mask` 打开，其他模型、数据、seed、训练超参和 patch off/on 对照保持不变。这样 sampler 会把相隔 `24/168/336` 的窗口配入同一训练 batch，loss 仍保留同一 ACF 表范围内的其他窗口作为 denominator negatives。

## 结果摘要

`best val` 是训练过程中最低 validation loss，不是 test MSE；`delta` 定义为 `(patch-on / patch-off - 1) * 100%`。

| 数据集 | patch | best epoch | best val | 可训练参数 |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | off | 1 | 0.734562 | 55,746 |
| ETTh1 | on | 1 | 0.714073 | 102,448 |
| Traffic | off | 3 | 0.387908 | 166,041 |
| Traffic | on | 4 | 0.375323 | 212,743 |

在这个修正后的单 seed、短训练筛选中，patch-token 相对 off 的 validation loss 变化是：ETTh1 `-2.79%`，Traffic `-3.24%`。两个数据集方向一致，但没有 test 指标、没有多 seed 统计，不能外推到泛化或论文结论。

## Relation telemetry

修正后 relation objective 不再退化：

| 数据集 / patch | global valid fraction | contrastive-active anchors | degenerate positives | positive count / anchor | sign accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 / off | 3.02% | 87.00% | 5.21% | 0.94 | 92.45% |
| ETTh1 / on | 3.02% | 87.00% | 5.21% | 0.94 | 92.45% |
| Traffic / off | 3.02% | 84.59% | 7.87% | 0.94 | 92.57% |
| Traffic / on | 3.02% | 84.09% | 8.20% | 0.93 | 92.57% |

这里的 `contrastive-active` 要求 anchor 同时存在正目标和至少一个 denominator negative；它已经从上一份 pilot 的 `0` 变为约 `84--87%`。这支持“本次确实执行了有负例的关系学习目标”，但不支持“该目标是 patch 增益的唯一来源”。局部 relation 的 target fraction 约 `1.58%`，仍然很稀疏；正式实验必须继续同时报告 global/local telemetry。

lag-pair exposure 也被写入各 run 的 `loss_logs.json`：三条 configured lags 的 epoch-level exposure 大致在 `0.30--0.36`，没有某一 lag 完全缺失。训练日志同时记录 `batching_effective=lag_pairs`、`reason=hard_selected_lag_mask`。

## 训练开销

短 pilot 的 epoch 总耗时（包含训练和当轮验证）如下：

| 数据集 | patch off | patch on | 相对变化 |
| --- | ---: | ---: | ---: |
| ETTh1 | 28.94 s | 30.81 s | +6.47% |
| Traffic | 62.24 s | 70.69 s | +13.57% |

这比未开启 lag mask 的旧结构 pilot 更接近真实 relation 训练合同，但仍是单 GPU、单 seed、4 epoch 的 wall-clock 读数，不等同于完整端到端效率。

prediction-only synthetic resource smoke 的结构读数不受 lag mask 影响，继续引用上一份 readout：patch-on 的 median forward 相对 off 在 ETTh1 为 `+40.49%`、Traffic 为 `+65.88%`，参数量分别增加 `+83.78%`、`+28.13%`。原始 JSON/CSV 位于 [`artifacts/exploratory/mamba_patch_token_pilot_20260907/resources`](../../artifacts/exploratory/mamba_patch_token_pilot_20260907/resources)。因此 patch-token 当前没有“更快”证据。

## 证据边界和下一步

- 本次仍是 validation-only、`seed=2021`、最多 4 epoch、`skip_test=true`；不进入论文主表。
- `patch-on` 与 `patch-off` 使用同一 relation 合同和同一 lag-pair sampler，因而可以作为下一轮多 seed 配对 screen 的起点。
- 下一轮应先用 3 个 seed 在 ETTh1、ETTm1、Electricity、Traffic 做 H96 配对复验；资源允许时再加 H336。plain baseline 至少保留 DLinear、NLinear、PatchTST、TimesNet、Mamba1、Mamba2。
- 正式 screen 要在启动前冻结 `hard_selected + lag_mask=true` 的语义、lag exposure、active-anchor 门槛和 test 指标规则；若多 seed 下 patch 增益不稳定，就把它降级为结构消融，而不是继续堆叠模块。

当前最谨慎的定位是：patch-token 在有效 relation learning 配置下出现了两个数据集一致的短期验证信号，且明显增加计算成本；它是值得复验的候选结构，还没有达到论文发表所需的确认性证据。
