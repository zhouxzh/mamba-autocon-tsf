# 当前基准库存（修正版）

**用途：** 人工复核仓库状态。本文档只汇总审计器读到的文件，不授权训练，也不把不同 provenance 的结果合并成论文结论。

## 一眼结论

- 2026-09-03 收尾审计发现 `37` 个计划、`110` 个 phase；此前只扫描无日期后缀文件名，旧快照的 `24/74` 不再代表当前库存。
- 工件目录中有 `5,882` 个配置和 `2,977` 个可用指标；其中 `226` 个指标位于注册 phase 根目录之外，均保留为历史或探索性工件。
- 注册 phase 展开为 `7,141` 个坐标，已有 `2,751` 个坐标匹配指标，覆盖率为 `38.52%`。P0、P1、P2 分别登记为 `0/36`、`0/12`、`0/180`；缺失数上升是将未授权草案纳入审计范围，不是结果丢失。
- 当前仍没有可以支持“所有数据集、所有 horizon、所有基线都不劣且更快”的证据。Traffic 上已有的 Mamba2 后验读数明显落后 PatchTST；MambaACF 候选尚未完成同一冻结协议下的确认性训练。

`mamba_acf_channel_order_conditioning_screen_plan.json` 是一个已执行的历史探索计划：其 JSON 保留启动前的 `training_started=false` 快照，不能在事后回填而破坏 launch sidecar 的计划哈希。该 screen 的实际执行状态应以授权 sidecar、各 phase 的 `phase_validation.json` 和 readout 为准（18/18）；当前新计划仍以各自的 `execution` 字段和授权门为准。

同样，`mamba_acf_variable_token_fair_screen_plan.json` 的 `execution` 快照与其 `registered_job_counts.training_started` 不一致，`mamba_acf_variable_memory_relation_screen.json` 则是旧的未授权快照但目录已有 18/18 历史工件。审计器在 current artifact 的 `Execution-state Consistency` 表中保留这些差异，要求结合各自 sidecar/provenance 阅读；这些历史状态不会被自动升级为当前源码重训或确认性证据。

## 已有结果能说明什么

| 范围 | 状态 | 可写结论 |
| --- | --- | --- |
| corrected / high-dimensional 历史矩阵 | 各自 phase 内覆盖完整，但模型列表缺 NLinear | 可作为历史背景和数据集依赖性诊断，不是完整六基线确认 |
| MambaACF memory、relation、variable-token 等 screen | 多为 H96、3 seeds 或单 seed | 只能说明机制方向，不能替代五 seed、多 horizon 主实验 |
| 日期后缀 screen plan | 已登记但大多 `authorized=false`、无输出根 | 是待复核的实验合同，不是已完成结果 |
| 资源 smoke | 有参数、显存和 device-resident latency 读数 | 不是端到端训练/数据加载速度，也不能由资源读数推断精度 |

已知的高维读数（scaled MSE）仍应按数据集分开报告：Electricity 上 Mamba2 与 PatchTST 接近且随 horizon 改变；Traffic H96/H336/H720 上 Mamba2 相对 PatchTST 分别约高 `29.2%/20.6%/19.8%`。这组工件标记为后验 replay，不能写成预注册确认性结果。

## 当前候选晋级顺序

1. 先完成构造级审计：Mamba-1/Mamba-2、AutoCon 表示路径、ACF prior/sidecar 和候选通道混合必须在同一 config 中可重放。
2. 再做小规模配对 screen：plain shared、channel-state mixer、variable-token、PatchTST 与 NLinear，固定数据切分、预算和 seeds；没有结果前不选“最佳”结构。
3. 只有候选在 screen 中同时满足精度非劣的预登记门槛和资源门槛，才复制到四数据集、三 horizon、五 seed 的确认计划。

## 复核命令

```bash
conda run --no-capture-output -n mamba python scripts/audit_benchmark_matrix.py \
  --plan-dir docs/research --artifact-root artifacts \
  --output-dir artifacts/benchmark_audit_20260903_post_contract

conda run --no-capture-output -n mamba python -m pytest tests/test_benchmark_matrix.py -q
```

原始 JSON/CSV/Markdown 审计输出位于 [`artifacts/benchmark_audit_20260903_post_contract/`](../../artifacts/benchmark_audit_20260903_post_contract/)。详细的 provenance、NLinear 缺口和历史限制见 [`dataset_baseline_inventory_20260903.md`](dataset_baseline_inventory_20260903.md) 与 [`mamba_current_status.md`](mamba_current_status.md)。
