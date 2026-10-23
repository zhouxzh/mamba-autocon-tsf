# 实验运行成本估计

**更新时间：** 2026-09-03  
**用途：** 人工决定训练批次和资源窗口；不是运行承诺，也不构成训练授权。

## 当前可执行规模

| 计划 | 作业数 | 当前状态 | 估计串行耗时 |
|---|---:|---|---:|
| `extension_baseline_plan.json` | 180 | `validate_only`；`0/180` 结果 | 约 13.9 小时（按历史中位数） |
| `mamba_acf_step_reliability_screen_plan_20260903.json` | 12 | `validate_only`；`0/12` 结果 | 约 8 分钟（按已有 H96 MambaACFAutoCon 记录） |
| `mamba_data_dependent_channel_mixer_screen_plan_20260903.json` | 60 | `validate_only`；`0/60` 结果 | 需单独按当前候选资源/历史记录估计 |
| `mamba_acf_confirmation_plan.json` | 884 | `validate_only`；`0/884` 结果 | 不建议在一次 GPU 会话中启动 |

## 估计方法

估计脚本以仓库现有 `artifacts/**/logs/job_manifest.jsonl` 为只读来源，先按 `job_id` 去重，只保留成功记录，再按 dataset、model、horizon 和 features 匹配历史 `elapsed_sec`。对扩展基线的 180 个计划作业，150 个坐标有同 dataset/model/horizon 的历史记录；30 个 NLinear 坐标没有同合同历史记录，使用同数据集历史中位数作保守替代。因此 `13.9` 小时是工程排程估计，不是当前源码的统计预测。

使用历史组内的 p90 替代中位数时，180 个作业合计约 `17.9` 小时。实际 wall-clock 还会受 GPU 状态、文件系统、早停轮数、失败重试和并发策略影响。Traffic/PatchTST 等高维作业通常是长尾，不能简单用所有作业的全局平均乘以作业数。

逐步可靠性计划的已有 H96 `MambaACFAutoCon` 成功记录中，ETTh1 median `21.52 s`、traffic median `59.89 s`；12 个作业按这两个数据集各 6 个估算约 `8.1 min`。该数字不包含新候选的编译抖动、资源 smoke 或失败重试，也不意味着可以跳过授权。

## 资源安排建议

- 先由人工复核并授权 12-job stepwise screen，作为低成本机制检查；它不会填补四数据集确认性矩阵。
- 高维 plain baseline 180 jobs 应按数据集或 horizon 分批，每批结束运行 `validate_training_phase.py`，确认 manifest、stream metrics、prior/provenance 没有漂移，再继续下一批。
- 不要同时在同一张 GPU 上堆叠多个 Traffic/PatchTST 作业；当前资源探针只证明单模型结构可承载，不证明并发安全或吞吐线性扩展。
- 884-job confirmation 不应直接启动。必须先冻结主比较边界、NLinear 处理方式、E2E 计时协议和 launch sidecar；如需分批，需为每批建立新的、哈希绑定的计划，而不是修改运行中的计划。

## 不应从估计推出的结论

耗时中位数不能推出“某模型更快”；prediction-only synthetic latency 也不能替代含数据加载、反向、验证和 checkpoint 的 E2E 测量。任何速度结论都必须来自同一源码、同一设备、同一 batch/AMP/loader 合同下的独立重复记录，并在准确率门通过后再解释。

## 复核入口

- 扩展基线：[`extension_baseline_plan.json`](extension_baseline_plan.json)
- 逐步可靠性：[`mamba_acf_step_reliability_screen_plan_20260903.json`](mamba_acf_step_reliability_screen_plan_20260903.json)
- 当前证据缺口：[`mamba_acf_evidence_gap_audit_20260903.md`](mamba_acf_evidence_gap_audit_20260903.md)
- 资源协议：[`resource-benchmark.md`](resource-benchmark.md)
