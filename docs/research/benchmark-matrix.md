# Benchmark Matrix Inventory

这份清单描述仓库当前“注册了什么、实际跑了什么、还缺什么”。它由
[`scripts/audit_benchmark_matrix.py`](../../scripts/audit_benchmark_matrix.py) 生成，
只读取计划、配置和指标文件，不选择模型，也不把不同 provenance 的数字合并。

## 当前注册对象

仓库登记了 9 个可复现 CSV 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Electricity、
Traffic、Weather、Exchange-Rate 和 National Illness。现有文件的列数、时间列和
`OT` 目标列可由审计工具检查；Weather 和 National Illness 的频率/时间语义与
Electricity、Traffic 不同，不能只按相同的 horizon 数字混为同一类结论。

模型入口包括 DLinear、NLinear、PatchTST、TimesNet、iTransformer、AutoConNet、
Mamba-1、Mamba-2，以及 MambaACF/MambaACF1 的 plain 和 AutoCon 变体。最低可接受的
预测基线集合是：

`DLinear`, `NLinear`, `PatchTST`, `TimesNet`, `Mamba1`, `Mamba2`。

AutoConNet 和 MambaACF 系列属于方法候选/消融，不替代上述基线。MambaACF
候选还必须单独报告 ACF memory、variable-token、channel-independent 和关系损失
配置，不能从一个候选配置推断所有变体。

## 结果覆盖口径

一个 job 只有在其目录有可解析且有限的 `metrics.npy`（至少 MAE/MSE/RMSE）或
合法 `stream_metrics.json` 时才算“完成”。只有 `config.json` 的目录会列为
`config_only_rows`；位于任何注册阶段根目录之外的指标会列为
`unplanned_metric_count`，例如旧 pilot，不能自动进入论文主表。

运行审计：

```bash
conda run --no-capture-output -n mamba python scripts/audit_benchmark_matrix.py \
  --plan-dir docs/research \
  --artifact-root artifacts \
  --output-dir artifacts/benchmark_audit
```

输出文件：

- `benchmark_audit.json`：完整的计划、数据集、模型和 phase 结构化清单；
- `phase_summary.csv`：每个 phase 的期望/完成/缺失数量和基线缺口；
- `missing_jobs.csv`：可直接转换为下一轮训练队列的缺失坐标；
- `benchmark_audit.md`：人工复核版摘要。

## 2026-09-02 历史审计快照

本轮只读审计输出在
 [`../../artifacts/benchmark_audit_20260902/benchmark_audit.md`](../../artifacts/benchmark_audit_20260902/benchmark_audit.md)：扫描 `24` 个计划、`74` 个 phase、`5,839` 个配置和 `2,934` 个可解析指标；其中 `183` 个指标位于注册 phase 之外，仍属于历史/探索性工件。该段是历史快照，不代表当前库存。当前日期后缀计划已纳入新的 [`benchmark_inventory_current_20260903.md`](benchmark_inventory_current_20260903.md) 和 [`../../artifacts/benchmark_audit_20260903_final2/benchmark_audit.md`](../../artifacts/benchmark_audit_20260903_final2/benchmark_audit.md)：`35` 个计划、`102` 个 phase、`5,882` 个配置、`2,977` 个指标、`226` 个未注册根指标。`corrected_confirmatory_S/M` 和 `highdim_confirmatory_M_v3` 的覆盖率为 `100%`，但审计仍标出两个 corrected M 阶段和高维 confirmatory 阶段缺少 `NLinear` 必要基线；selection/extension 目录中的大量条目只有配置而没有指标，不能当作已完成训练。

因此当前文档中的 baseline 数字只能按各自注册矩阵和 provenance 解读。补齐 NLinear 或扩展计划前，不能把已有 DLinear/PatchTST/Mamba 数字写成“完整六基线”或跨协议排行榜。

## 当前缺口解释

已有 corrected/highdim 结果覆盖了若干 5-seed 矩阵，但它们属于各自的历史
或修正后协议。`extension_highdim_plain_M` 计划要求 Electricity、Traffic 上
6 个 plain 基线、3 个 horizon 和 5 个 seed，共 180 个坐标；在该计划指定的
输出根下没有可计入的指标时，审计结果应显示 0% coverage，而不是借用
`highdim_confirmatory_M_v3` 或 `fair_pilot` 的数字填充。后者的训练预算、源码
快照或模型头可能不同，跨根目录拼接会破坏配对比较。

如果要形成新的主结果矩阵，建议先完成以下最小闭环：

1. 固定一个新的 plan/provenance，显式写出 9 个数据集中的目标子集、features、
   horizon、seed、optimizer、模型宽度和 AutoCon lambda 选择规则；
2. 每个数据集至少包含六个最低基线；MambaACF 变体作为额外行，并保留 plain
   Mamba1/Mamba2 对照；
3. 先只用 validation 选择结构和 lambda，冻结后再读 test；
4. 以 dataset × horizon × seed 为配对单位收集 scaled MSE、原始量纲诊断、参数量、
   峰值显存和包含数据加载的 epoch 时间；
5. 对缺失 seed 或不一致配置停止汇总，重新开新 phase 根，不覆盖旧工件。

## 解释边界

外部 GitHub/论文中的 Mamba 时间序列结果只能作为架构动机，不能作为本仓库的
基线数字。只有在同一数据切分、特征模式、训练预算、硬件和指标合同下重新运行
并留下 provenance 的结果，才可放入本地比较表。审计工具因此只报告覆盖和缺失，
不输出“最佳模型”结论。
