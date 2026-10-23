# MambaACF Lite AutoCon Resource Preflight

日期：2026-09-09。状态：已完成结构资源 smoke；训练计划仍未授权，本文不是精度结果。

## 候选定义

`MambaACFLiteAutoCon` 是训练计划层的固定公开别名，执行同一份
`MambaACFAutoCon` 模块，但锁定以下配置：`mamba_scales=[1]`、
`mamba_causal_embedding=true`、`mamba_low_rank_head=true`、
`mamba_temporal_rank=8`，并关闭 variable-token、patch、local 和 ACF forecast-memory
分支。AutoCon 的 ACF lag-pair relation loss 保留在训练路径；预测-only 资源测试显式关闭辅助表示返回，
因此不会把训练时的 relation projection 误计入部署前向。

计划文件为 [`mamba_acf_lite_autocon_pilot_20260909.json`](mamba_acf_lite_autocon_pilot_20260909.json)，
包含 ETTh1/Traffic、H96、3 seeds、24 个 validation-only 作业。`execution.authorized=false`，
目前只完成计划校验和资源 smoke，没有启动训练或读取 test split。

随后代码审计补齐了资源工具的公开别名归一化：无论显式 CLI 还是带
`plan_job_id` 的可执行模型配置，`MambaACFLiteAutoCon` 都会强制回到上述固定配方。
这只收紧输入配置边界，不构成新的 GPU 延迟结果；下表已有记录中的 architecture
字段已经符合该配方。

## 同形状资源读数

RTX 5090 D；device-resident synthetic FP32 input；10 次 warm-up、50 次 timed forward、batch=8；
报告 median latency 和 p95，资源工具版本与配置保存在对应输出目录。数值单位为毫秒、MB 和 sequences/s。

| 数据集 / arm | 可训练参数 | median ms | p95 ms | sequences/s | peak increment MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 PatchTST | 46,080 | 0.2722 | 0.2939 | 28,986.7 | 1.51 |
| ETTh1 Mamba-2 | 29,907 | 0.6703 | 0.6912 | 11,896.9 | 3.30 |
| ETTh1 MambaACF full AutoCon | 49,409 | 1.2128 | 1.2432 | 6,571.8 | 3.28 |
| ETTh1 MambaACF Lite AutoCon | 21,844 | 0.7224 | 0.7392 | 11,046.8 | 3.28 |
| Traffic PatchTST | 46,080 | 0.8880 | 0.8997 | 8,983.8 | 95.19 |
| Traffic Mamba-2 | 167,562 | 0.7075 | 0.7290 | 11,281.9 | 23.10 |
| Traffic MambaACF full AutoCon | 159,704 | 1.2967 | 1.3182 | 6,160.8 | 15.30 |
| Traffic MambaACF Lite AutoCon | 132,139 | 0.7436 | 0.7619 | 10,721.7 | 15.30 |

相对同数据集的 full AutoCon，Lite median latency 降低约 `40.4%`（ETTh1）和 `42.6%`
（Traffic），可训练参数降低约 `55.8%` 和 `17.3%`。这只证明当前合成前向协议下的结构负载变化：
Lite 仍略慢于普通 Mamba-2，ETTh1 上也明显慢于 PatchTST；Traffic 上则比 PatchTST 更快且峰值增量更低。

## 人工复核边界

- 低秩 head 的历史三 seed screen 曾出现约 `+1.095%` 的 MSE 退化，因此本次 rank-8 只是待证伪候选，不能预先解释为“压缩后不掉点”。
- 单尺度扫描有历史资源优势，但尚没有与本 Lite 配置配对的训练精度证据。
- AutoCon relation 的有效性必须由训练日志中的 active-anchor、degenerate-positive、lag exposure 和 sign accuracy 共同检查；资源 smoke 不包含这些指标。
- 只有 Lite 在 ETTh1/Traffic 的三 seed validation loss 不明显退化，且 train-step / end-to-end 资源同步改善，才值得进入更长 horizon、独立 test release 的确认计划。

原始资源目录：[`artifacts/exploratory/mamba_acf_lite_autocon_pilot_20260909/resources`](../../artifacts/exploratory/mamba_acf_lite_autocon_pilot_20260909/resources)。
