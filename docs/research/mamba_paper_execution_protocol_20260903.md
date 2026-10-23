# Mamba + AutoCon 论文执行协议（人工复核版）

**日期：** 2026-09-03  
**状态：** 代码候选已实现；2026-09-07 新增的 parity/variable pilot 已完成，但仍是
exploratory validation-only；确认性实验未授权、未启动。本文档不是论文结果。

## 先给结论

当前证据还不足以支持“ Mamba + AutoCon 已经稳定优于强基线”或“训练/推理更快”的论文结论。最明确的失败信号仍是 Traffic：历史高维后验复核中，Mamba2 相对 PatchTST 的 scaled-MSE 差为 H96 `+29.205%`、H336 `+20.611%`、H720 `+19.833%`。这说明首要问题是高维跨通道信息流和归一化公平性，不是简单增加 Mamba 层数。

## 当前可复核的代码候选

1. `MeanStd`：在 Mamba、Mamba AutoCon、MambaACF 和 AutoConCI 中使用 PatchTST 相同的 population variance 合同 `sqrt(var + 1e-5)`；`LastVal`、`Mean` 和 sample-standard-deviation 的 `ReVIN` 保持为另一套明确合同。它是公平性控制，不是精度提升主张。
2. `mamba_variable_repr_fusion`：将已有的 `[B,C,D]` variable-token Mamba 状态做 softmax channel pooling，经 `D -> D -> D` adapter 后以小门控注入 AutoCon temporal representation。默认关闭，只允许 `MambaACFAutoCon/MambaACF1AutoCon + mamba_variable_mixer`。它让 relation loss 能直接到达 variable branch，但窗口级摘要不能声明严格 prefix-causal。
3. 训练期 ACF 仍只来自 train-only sidecar；推理不得从 validation/test 重新估计 lag、通道顺序或统计量。默认预测 head 与 relation representation 分离；`forecast` relation view 是单独注册的训练期消融，当前 global loss 仅约束预测窗口的跨通道均值关系，不是逐步 forecast dynamics。
4. 资源证据分为 prediction-only 与 prediction-plus-representation 两种协议：后者可用 `scripts/benchmark_model_resources.py --include-representation` 显式测量，但只覆盖表示前向，不包含 relation loss、反向传播或端到端训练时间；两种记录不能混合排名。

主干选择上，Mamba-2 应作为效率/主结果候选，Mamba-1 作为同预算结构基线；两者不能合并成一个“Mamba”结果。Mamba-2 的 fused selective-scan 路径更适合做速度比较，但高维 Traffic 的通道压缩问题仍需由 variable-token 或 channel-preserving 分支单独验证。

`ReVIN` 在 AutoConNet/AutoConCI/Mamba-1/2 head 与 MambaACF 统一使用 unbiased
sample standard deviation；正式跨家族公平比较固定显式 `MeanStd` population
合同。训练日志 telemetry 的批量搬运只减少同步开销，不改变 forecast/loss
数值，也不能单独支持 speed claim。

实现入口：[`models/MambaACF.py`](../../models/MambaACF.py)、[`run.py`](../../run.py)、[`utils/mamba_causality_contract.py`](../../utils/mamba_causality_contract.py)。外部结构核验见 [`mamba_architecture_external_audit_20260903.md`](mamba_architecture_external_audit_20260903.md)。

## 分阶段执行顺序

| 阶段 | 计划 | 目的 | 当前状态 |
|---|---|---|---|
| P0 | [`mamba_normalization_parity_screen_plan_20260903.json`](mamba_normalization_parity_screen_plan_20260903.json) | ETTh1/Traffic、H96、3 seed，比较 LastVal 与 PatchTST-compatible MeanStd | 36 个 validation-only jobs，`validate_only` 通过，未授权 |
| P1 | [`mamba_variable_repr_fusion_screen_plan_20260903.json`](mamba_variable_repr_fusion_screen_plan_20260903.json) | 在 MeanStd 公平尺度下比较同一 variable-token MambaACF AutoCon 的 representation fusion off/on | 12 个 validation-only jobs，`validate_only` 通过，未授权 |
| P1b | [`mamba_acf_forecast_relation_view_screen_plan_20260903.json`](mamba_acf_forecast_relation_view_screen_plan_20260903.json) | `encoder/forecast × lambda=0/.03`，检验窗口级 direct forecast relation routing | 24 个 validation-only jobs，`validate_only` 通过，未授权 |
| P1c | [`mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json`](mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json) | 在同一 train-only ACF bank 上比较逐通道 top-k memory support；只声明 forecast-memory indexing 因果性；指标取 `loss_logs.json:val_loss` | 12 个 validation-only jobs，`validate_only` 通过，未训练、未授权 |
| P2 | [`mamba_multidataset_plain_baseline_expansion_plan_20260903.json`](mamba_multidataset_plain_baseline_expansion_plan_20260903.json) | 在同一新 provenance 下补齐 ETTh1/ETTm1/Electricity/Traffic 的 DLinear/NLinear/PatchTST/TimesNet/Mamba1/Mamba2 三 horizon plain 基线 | 216 个 validation-only jobs，`validate_only` 通过，未训练、未授权 |
| P2-legacy | [`extension_baseline_plan.json`](extension_baseline_plan.json) | 独立高维 Electricity/Traffic NLinear 缺口矩阵 | 180 jobs 已由旧计划计数锁定，不能与 P2 拼接 |
| P3 | [`mamba_acf_confirmation_plan.json`](mamba_acf_confirmation_plan.json) | 4 数据集、多个 horizon、5 个确认 seed 的未来冻结矩阵 | 884 jobs，仍需 launch sidecar 和人工批准 |

2026-09-09 新增的 PEMS exploratory screen 不改变上述确认矩阵：

| PEMS exploratory | 合同 | 结果状态 |
|---|---|---|
| [`mamba_pems_ts_v3_fresh_seeds_20260909.json`](mamba_pems_ts_v3_fresh_seeds_20260909.json) | PEMS03/04/07/08、H12/H96、Mamba2 vs MambaTSv3、2 fresh seeds、4 epochs、validation-only | 32/32 完成；v3 为 16/16 配对改善，不能替代确认性结果 |
| [`mamba_pems_ts_v3_autocon_screen_20260909.json`](mamba_pems_ts_v3_autocon_screen_20260909.json) | 同一 PEMS 坐标，plain/AutoCon v3 与 Mamba-2 配对、seed 2022、4 epochs | 32/32 完成；v3 + AutoCon 相对 plain v3 仅 1/8 改善，不能宣称 AutoCon 增益 |
| [`mamba_pems_ts_v3_strong_baseline_screen_20260909.json`](mamba_pems_ts_v3_strong_baseline_screen_20260909.json) | DLinear/PatchTST/TimesNet/Mamba2/MambaTSv3、PEMS 四数据集、H12/H96、seed 2022、4 epochs | 40/40 完成；TimesNet 在部分 H96 坐标胜出，仍需 fresh seeds/长预算/test gate |
| [`mamba_pems_ts_v3_timesnet_fresh_seed_20260909.json`](mamba_pems_ts_v3_timesnet_fresh_seed_20260909.json) | TimesNet/MambaTSv3、PEMS 四数据集、H12/H96、fresh seed 2023、4 epochs | 16/16 完成；v3 为 6/8 胜，H96 为 2/4 胜；仍属 exploratory |

2026-09-07 的小规模补充 pilot 不追溯性改变上表计划：

| 补充 pilot | 合同 | 结果状态 |
|---|---|---|
| [`mamba_meanstd_baseline_pilot_20260907.json`](mamba_meanstd_baseline_pilot_20260907.json) | ETTh1/ETTm1/Electricity/Traffic，七个核心基线，H96、MeanStd、三 seed、四 epoch | 84/84 完成；Mamba-1/2 未在四个数据集上追平最强非 Mamba，且不普遍更快；不进确认主表 |
| [`mamba_normalization_parity_pilot_readout_20260907.md`](mamba_normalization_parity_pilot_readout_20260907.md) | ETTh1/Traffic、H96、PatchTST/Mamba1/Mamba2/MambaACF/MambaACFAutoCon、MeanStd、2 seeds、4 epochs | 20/20 完成；Traffic MambaACF 低于 PatchTST，AutoCon 独立均值增益接近零；不进主表 |
| [`mamba_variable_parity_pilot_20260907.md`](mamba_variable_parity_pilot_20260907.md) | 同一 MeanStd/AutoCon 合同，仅增加 variable-token branch，2 数据集、2 seeds | 4/4 完成；Traffic 两 seed 改善但 latency 增加约 71--72%；只作为待复验精度候选 |
| [`mamba_variable_parity_replication_20260907.md`](mamba_variable_parity_replication_20260907.md) | 沿用上述合同补充 seed 2023，shared/variable 配对，ETTh1/Traffic H96 | 4/4 完成；Traffic 三 seed 全部改善，prediction-only latency 仍增加约 69%；不进确认主表 |
| [`mamba_classical_baseline_extension_pilot_20260907.json`](mamba_classical_baseline_extension_pilot_20260907.json) | Weather/Exchange-Rate，12 个模型，H96、MeanStd、三 seed、四 epoch | 72/72 完成；Weather 最优为 Crossformer/TimesNet，Exchange-Rate 最优为 PatchTST/NLinear；Mamba 仍不具备跨域不劣性或速度主张 |
| [`mamba_classical_baseline_core_extension_pilot_20260907.json`](mamba_classical_baseline_core_extension_pilot_20260907.json) | ETTh1/ETTm1/Electricity/Traffic，五个经典模型，H96、MeanStd、三 seed、四 epoch | 60/60 完成；补齐经典模型覆盖；该阶段无 Mamba 臂，不与已有 84-job MeanStd pilot 直接合并排名 |

P0、P1、P1b 和 P1c 是独立的 validation-only 机制筛选，不能直接填入论文主表，也不能追溯性地选择或证明任何已有 P3/历史高维工件：已有历史高维设置使用 `LastVal` 且没有 variable representation fusion 或 forecast relation view。P1c 不声称 AutoCon representation 的 prefix-causality。P2 同样不能与历史高维 `_v3` 表拼接，它的尺度、scales、batch 和 ACF source 合同不同。若这些筛选有值得复制的候选，必须新建包含固定 config hash、fresh seeds、train-only sidecar、注册计数和独立 test gate 的 P3b 确认计划。

这批 MeanStd pilot 只解决了一个已知公平性缺口：DLinear/NLinear 现在可以显式使用与 PatchTST 相同的窗口 `MeanStd` 归一化。它仍是四 epoch 的 validation-only 筛选，不能替代 P2 的完整 H96/H336/H720、长预算和 test-release 合同；其结果只能用于决定哪些模型值得进入后续冻结矩阵，不能作为论文主表或非劣性结论。

## 每阶段的停止条件

- 任一 phase 缺少坐标、配置哈希、sidecar、checkpoint 或 manifest 完整性，停止汇总。
- 只能用 validation 选择 lambda、epoch、lag 或结构；test 只能在冻结确认阶段结束后一次性报告。
- paired unit 是同一 dataset、horizon、seed；窗口和通道不是独立重复。
- 精度和效率分开审计：参数量、峰值显存、device-resident latency、吞吐和端到端 epoch time 必须用相同硬件、batch、warmup、同步规则测量。
- P1 的 representation fusion 是单因素 off/on；不得同时改变 relation target、ACF lag、normalization、variable order 或训练预算。它只选择 validation 行为，不能产生 test 指标。
- P1b 以同一 route 内的 `lambda=.03 - lambda=0` 对比以及两者的差分中的差分为主；lambda=0 仍计算 relation path，不能用它比较“启用/关闭辅助路径”的训练吞吐。

## 论文最低证据门槛

在写“方法有效”前，至少要同时满足：

- 新的 P3b 强基线在同一数据切分、同一训练预算和不少于 5 个确认 seed 下完整可复核；P2 只补齐独立探索矩阵，不能替代该条件；
- P3b 预先指定的 PatchTST/Mamba2 primary non-inferiority 比较通过 multiplicity-adjusted 统计门槛；DLinear、NLinear、TimesNet、iTransformer、Mamba1 只能在门槛扩展后作为 headline comparator；
- ACF memory、AutoCon relation、variable representation fusion 和 relation representation choice 各自有独立 null/ablation，不能只比较一个全部打开的组合；
- Traffic 等高维数据不能只报整体平均，必须给出逐通道/分组诊断和容量信息；
- 没有数据或测试泄漏，所有 train-only sidecar、source snapshot、环境版本和命令均可复现；
- 只有当精度门槛先通过，且在预先指定的 nonlinear comparator 上达到注册的时间优势，才可以写 speed claim。

若这些条件未满足，论文定位应改为“失败分析/机制诊断/候选方法报告”，而不是 SOTA 或普适改进论文。

## 人工复核命令

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_normalization_parity_screen_plan_20260903.json --validate_only
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_variable_repr_fusion_screen_plan_20260903.json --validate_only
conda run --no-capture-output -n mamba python -m pytest -q
```

以上命令只验证代码和实验合同，不启动训练，也不会生成新的精度结果。真实 launch 还必须绑定人工批准的 `launch_sidecar`；当前两个新 screen 的 `execution.authorized=false`。
