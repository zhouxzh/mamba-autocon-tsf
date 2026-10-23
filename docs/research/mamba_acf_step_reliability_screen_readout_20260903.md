# MambaACF 逐预测步可靠性筛选：人工复核稿

**状态：** `exploratory_step_reliability_screen_draft`；未授权，未启动训练。

计划文件为 [`mamba_acf_step_reliability_screen_plan_20260903.json`](mamba_acf_step_reliability_screen_plan_20260903.json)。当前只完成了 JSON 解析和 runner `validate_only`（12 个计划作业）；没有新的精度、训练吞吐或端到端速度结果。

## 为什么做这个消融

当前静态 ACF memory 有两种可靠性路径：默认把合法性和 ACF 强度汇总成每通道一个系数，再广播到所有预测步；`mamba_acf_step_reliability=true` 则为每个预测步使用其实际可观测 lag 集合的可靠性。输入长度和预测长度相同、且候选 lag 包含长于输入窗口的周期时，后段 horizon 的合法 lag 会减少，因此两种路径的归因是清楚的。

这个候选只改变 forecast-memory 的 gate。它不改变 Mamba 主干、AutoCon relation objective、ACF prior、lag selector、数据划分、优化器或训练预算，也不解除当前全窗口归一化对“严格表示因果性”主张的限制。

## 注册的比较

| 项目 | 固定值 |
|---|---|
| 数据 | ETTh1（7 通道）、traffic（862 通道） |
| horizon / input | H96 / L96 |
| 模型 | `MambaACFAutoCon` |
| seeds | 2021、2022、2023 |
| memory | static、train-only、raw ACF、`train_peaks_available` |
| relation | global、hard-selected、balanced lag pairs、lambda=0.03 |
| Mamba | `scales=[1]`、unit stride、causal embedding、非双向 |
| 训练 | batch 32、8 epochs、patience 3、cosine、同一 split |
| 唯一差异 | `mamba_acf_step_reliability=false` vs `true` |

## 先验行为检查

在正式训练前应保留一个不依赖数据集的单元测试：若某通道的 prior 只有一个强峰（例如 `[0.9, 0, 0]`），逐步可靠性应接近 `0.9`；聚合路径会把该强峰按候选槽位平均，约为 `0.3`。这只是公式检查，不是预测结果，也不能替代 paired retraining。

## 预先声明的判定

- 主指标是 validation scaled MSE；validation scaled MAE、elapsed、active trainable parameters 和 ACF telemetry 是次要指标。
- 每个数据集必须分别报告三个 seed、配对 delta、均值/中位数比值和置信区间；ETTh1 与 traffic 不能当作六个独立重复合并检验。
- 筛选通过要求每个数据集的 stepwise/aggregate 平均 MSE 比值不超过 1.000、至少 3 个 seed 中 2 个胜出、且单个 seed 比值不超过 1.05。
- 每个完成 epoch 必须有有限的 `acf_runtime_gate_mean`、`acf_seasonal_rms`、`acf_effective_memory_rms` 和 `acf_effective_memory_to_forecast_rms`；缺任何一项，机制读out无效。
- stepwise arm 的 median latency 和 steady-state peak allocated memory 不得超过 aggregate arm 的 1.10 倍。这是成本预算，不是“更快”结论。

## 当前没有的证据

本稿不能支持以下表述：逐步可靠性已经提高精度、ACF 已经带来稳定收益、MambaACF 不劣于 PatchTST/Mamba2、MambaACF 更快，或该消融具有严格表示因果性。计划通过只说明配置可以被展开且没有违反静态合同。

## 人工复核后才可做的事

1. 确认是否接受两数据集、H96、三 seed 的探索性范围，以及上述停止规则。
2. 核对当前源码、环境和数据哈希，并为两臂做同一硬件协议的资源 smoke。
3. 冻结 JSON、生成与计划哈希绑定的 approved launch sidecar，并使用新的输出根。
4. 训练完成后先运行 phase validator，再由只读 analyzer 生成配对 readout；不得用 test 指标选臂或改配置。

## 相关证据

- 当前 ACF 代码与诊断合同：[`models/MambaACF.py`](../../models/MambaACF.py)
- 训练日志收集：[`exp/exp_long_term_forecasting_with_AutoCon.py`](../../exp/exp_long_term_forecasting_with_AutoCon.py)
- 逐步可靠性已有公式/合同测试：[`tests/test_mamba_acf_contract.py`](../../tests/test_mamba_acf_contract.py)
- 总体证据边界：[`mamba_acf_evidence_gap_audit_20260903.md`](mamba_acf_evidence_gap_audit_20260903.md)
